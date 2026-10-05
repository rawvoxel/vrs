// The trace pass: from the frame's depth, at a fraction of the frame's resolution or all of
// it, for shading to upscale, see `shadowTaps` in `mesh.slang`.
//
// `cullLights` first lists the lights reaching each `LIGHT_TILE` wide tile of pixels. Then
// `traceSurfaces`, for each block of pixels, traces the sun's shadow and those of the block's
// brightest few lights, finds the voxel it shows, and traces its reflection when it's glossy.
// Below full resolution, it also finds the light reaching it from everywhere but the sun,
// with its ambient occlusion.

#include "common.slang"

// Match `DebugView::Trace` and `DebugView::Levels`, `DEBUG_TRACE` and `DEBUG_LEVELS` in
// `mesh.slang`.
static const uint DEBUG_TRACE_VIEW = 6;
static const uint DEBUG_LEVELS_VIEW = 7;

// Each texel's sun visibility, tinted by see-through voxels, and its surface's distance from
// the camera.
[[vk::binding(13, 0)]] [[vk::image_format("rgba32f")]] RWTexture2D<float4> shadowsStorage;
[[vk::binding(14, 0)]] Sampler2D<float> depthBuffer;
// For each tile, how many lights reach it, then which, `TILE_WORDS` words a tile.
[[vk::binding(18, 0)]] RWStructuredBuffer<uint> lightTiles;
// Each texel's shadowed lights, 8 bits each in x, `NO_LIGHT` for none, and how much of each
// gets through, 8 bits each in y.
[[vk::binding(19, 0)]] [[vk::image_format("rg32ui")]] RWTexture2D<uint2> lightShadowsStorage;
// Each texel's sharp reflection when its voxel is glossy, and its face's number, 0 for none.
[[vk::binding(21, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> reflectionsStorage;
// Each texel's light from everywhere but the sun, with its ambient occlusion, and its face's
// number, 0 for none.
[[vk::binding(23, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> indirectStorage;

groupshared uint tileNearest;
groupshared uint tileFarthest;
groupshared uint tileCount;
// Every light reaching the tile, and how bright it is at the tile's nearest point.
groupshared uint tileList[MAX_LIGHTS];
groupshared float tileStrengths[MAX_LIGHTS];

[shader("compute")]
[numthreads(LIGHT_TILE, LIGHT_TILE, 1)]
void cullLights(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID, uint index: SV_GroupIndex) {
    if (index == 0) {
        tileNearest = 0xffffffff;
        tileFarthest = 0;
        tileCount = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    // The tile's depth range, depths being positive floats that order as their bits do.
    uint2 frameSize = uniforms.frameSize;
    uint2 pixel = group.xy * LIGHT_TILE + thread.xy;
    if (all(pixel < frameSize)) {
        float depth = depthBuffer.SampleLevel((float2(pixel) + 0.5) / float2(frameSize), 0.0);
        if (depth < 1.0) {
            InterlockedMin(tileNearest, asuint(depth));
            InterlockedMax(tileFarthest, asuint(depth));
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // The lights whose range reaches the box around the tile's pixels in the world.
    if (tileFarthest != 0) {
        float2 low = float2(group.xy * LIGHT_TILE) / float2(frameSize) * 2.0 - 1.0;
        float2 high = float2(min((group.xy + 1) * LIGHT_TILE, frameSize)) / float2(frameSize) * 2.0 - 1.0;
        float3 boxMin = float3(1e30);
        float3 boxMax = float3(-1e30);
        for (uint corner = 0; corner < 8; corner++) {
            float3 ndc = float3(
                (corner & 1) != 0 ? high.x : low.x,
                (corner & 2) != 0 ? high.y : low.y,
                asfloat((corner & 4) != 0 ? tileFarthest : tileNearest));
            float4 offset = mul(uniforms.cameraOffsetFromClip, float4(ndc, 1.0));
            float3 world = uniforms.cameraPosition + offset.xyz / offset.w;
            boxMin = min(boxMin, world);
            boxMax = max(boxMax, world);
        }
        for (uint i = index; i < uniforms.trace.lightCount; i += LIGHT_TILE * LIGHT_TILE) {
            Light light = lights[i];
            float3 nearest = clamp(light.positionRange.xyz, boxMin, boxMax);
            float distance = length(nearest - light.positionRange.xyz);
            if (distance < light.positionRange.w) {
                uint slot;
                InterlockedAdd(tileCount, 1, slot);
                tileList[slot] = i;
                tileStrengths[slot] = dot(light.colorSpot.rgb, LUMINANCE) * lightFalloff(distance, light.positionRange.w);
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // With more than fit, the brightest are kept, ranked by how many are brighter.
    uint tile = (group.y * uniforms.trace.tilesAcross + group.x) * TILE_WORDS;
    uint total = tileCount;
    if (index == 0) {
        lightTiles[tile] = min(total, MAX_TILE_LIGHTS);
    }
    for (uint i = index; i < total; i += LIGHT_TILE * LIGHT_TILE) {
        uint rank = i;
        if (total > MAX_TILE_LIGHTS) {
            rank = 0;
            float strength = tileStrengths[i];
            for (uint j = 0; j < total; j++) {
                float other = tileStrengths[j];
                rank += (other > strength || (other == strength && j < i)) ? 1 : 0;
            }
        }
        if (rank < MAX_TILE_LIGHTS) {
            lightTiles[tile + 1 + rank] = tileList[i];
        }
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void traceSurfaces(uint3 threadId: SV_DispatchThreadID) {
    uint scale = uniforms.trace.pixelsPerTexel;
    uint2 size = uniforms.trace.imageSize;
    uint2 frameSize = uniforms.frameSize;
    if (any(threadId.xy >= size)) {
        return;
    }
    // The pixel at the block's center.
    uint2 pixel = min(threadId.xy * scale + scale / 2, frameSize - 1);
    float2 uv = (float2(pixel) + 0.5) / float2(frameSize);
    float depth = depthBuffer.SampleLevel(uv, 0.0);
    if (depth >= 1.0) {
        // Nothing drawn, the sky.
        shadowsStorage[threadId.xy] = float4(1.0, 1.0, 1.0, 1e9);
        lightShadowsStorage[threadId.xy] = NO_SHADOWED_LIGHTS;
        reflectionsStorage[threadId.xy] = float4(0.0);
        indirectStorage[threadId.xy] = float4(0.0);
        return;
    }
    float4 offset = mul(uniforms.cameraOffsetFromClip, float4(uv * 2.0 - 1.0, depth, 1.0));
    float3 fromCamera = offset.xyz / offset.w;
    float3 position = uniforms.cameraPosition + fromCamera;
    float distance = length(fromCamera);
    float3 toCamera = -fromCamera / max(distance, 1e-6);
    float3 view = -toCamera;
    float3 onSurface = toGrid(position);
    bool hasGrid = uniforms.lighting.gridLevels > 0;

    // The voxel the pixel shows and the face it's seen through, see-through ones too, found by
    // a short ray from in front of it. The depth buffer only places the surface roughly, more
    // so further away, so rays start just off this face instead, past no voxel. Near a
    // concave edge, a voxel back toward the camera can be inside the edge's other face, then
    // a quarter voxel back is the empty cell in front of this one.
    VoxelHit surface;
    float3 onFace = float3(0.0);
    bool found = false;
    if (hasGrid) {
        for (int attempt = 0; attempt < 2 && !found; attempt++) {
            float back = attempt == 0 ? 1.0 : 0.25;
            float3 start = onSurface - view * back;
            found = traceVoxels(start, view, back + 1.0, false, true, surface) && surface.distance > 0.0;
            onFace = start + view * surface.distance;
        }
    }
    // Where rays toward the sun and the lights start: just off the face, else just off the
    // surface toward the camera, which is always on its open side, skipping any voxels that's
    // still in.
    float3 rayOrigin = found
        ? onFace + surface.normal * SHADOW_RAY_OFFSET
        : onSurface + toCamera * SHADOW_RAY_OFFSET;

    float3 visible = float3(1.0);
    uint sunTraceEnd = TRACE_LEFT_TABLES;
    if (hasGrid) {
        VoxelHit hit;
        bool blocked = traceVoxels(rayOrigin, uniforms.lighting.sunDirection, uniforms.lighting.traceDistance, !found, false, hit);
        visible = blocked ? float3(0.0) : hit.transmittance;
        sunTraceEnd = traceEnd;
    }
    shadowsStorage[threadId.xy] = float4(visible, distance);

    // The light reaching the face from everywhere but the sun, and what it reflects when it's
    // glossy. Without the face, as on grass and leaves, which aren't voxels, the light
    // reaching the surface as if facing up, as they mostly do, rather than shading falling
    // back on the sky's, which indoors is far too bright. Always up, so it's the same however
    // the camera sees the surface.
    float4 reflection = float4(0.0);
    float4 indirect = float4(0.0);
    if (!found && hasGrid && uniforms.tracedIndirect != 0) {
        float3 up = float3(0.0, 1.0, 0.0);
        float3 light = indirectLight(onSurface, up, toCamera, false);
        indirect = float4(light, float(faceNumber(up)));
    }
    if (found) {
        float face = float(faceNumber(surface.normal));
        if (uniforms.tracedIndirect != 0) {
            float3 light = indirectLight(onFace, surface.normal, toCamera, false);
            // Only axis aligned faces are darkened, as when each pixel finds its own.
            float occlusionStrength = uniforms.lighting.ambientOcclusion;
            if (occlusionStrength > 0.0 && meshes[surface.mesh].kind == KIND_AXIS_ALIGNED) {
                float3 inLevel = onFace / float(1 << surface.level);
                light *= lerp(1.0, gridOpenness(surface.level, inLevel, int3(surface.normal)), occlusionStrength);
            }
            indirect = float4(light, face);
        }
        Material material = materialOf(surface.id);
        if (uniforms.trace.traceReflections != 0 && material.colorRoughness.w < GLOSSY_ROUGHNESS) {
            float3 reflected = radianceAlong(rayOrigin, reflect(view, surface.normal), uniforms.lighting.traceDistance);
            reflection = float4(reflected, face);
        }
    }
    // For `DebugView::Trace` and `DebugView::Levels`: whether the face was found, how much
    // sun reaches it, its level, and how the sun's ray ended, in place of the reflection.
    uint debugView = uniforms.lighting.debugView;
    if (debugView == DEBUG_TRACE_VIEW || debugView == DEBUG_LEVELS_VIEW) {
        float level = found ? float(surface.level) : 0.0;
        reflection = float4(found ? 1.0 : 0.0, dot(visible, LUMINANCE), level, float(sunTraceEnd));
    }
    reflectionsStorage[threadId.xy] = reflection;
    indirectStorage[threadId.xy] = indirect;

    // The block's tile's brightest lights here, as many as are shadowed, brightest first.
    uint shadowed = min(uniforms.trace.shadowedLights, MAX_SHADOWED_LIGHTS);
    BrightestLights brightest = noBrightestLights();
    uint tile = tileOf(pixel) * TILE_WORDS;
    uint count = lightTiles[tile];
    for (uint i = 0; i < count; i++) {
        uint lightIndex = lightTiles[tile + 1 + i];
        keepBrightest(brightest, shadowed, lightIndex, lightStrength(lights[lightIndex], position), float3(0.0));
    }
    uint2 result = NO_SHADOWED_LIGHTS;
    for (uint k = 0; k < shadowed && brightest.lights[k] != NO_LIGHT; k++) {
        float lit = lightShadow(lights[brightest.lights[k]], rayOrigin);
        result.x = (result.x & ~(NO_LIGHT << (k * 8))) | (brightest.lights[k] << (k * 8));
        result.y |= uint(round(saturate(lit) * 255.0)) << (k * 8);
    }
    lightShadowsStorage[threadId.xy] = result;
}

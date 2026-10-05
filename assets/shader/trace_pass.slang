// The trace pass: from the frame's depth, at a fraction of the frame's resolution or all of
// it, for shading to upscale, see `shadowTaps` in `mesh.slang`.
//
// `cullLights` first lists the lights reaching each `LIGHT_TILE` wide tile of pixels. Then
// `traceSurfaces`, for each block of pixels, traces the sun's shadow and those of the block's
// brightest few lights, finds the voxel it shows, and traces its reflection when it's glossy.
// Below full resolution, it also finds the light reaching it from everywhere but the sun,
// with its ambient occlusion.

#include "common.slang"

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
            float4 world = mul(uniforms.inverseViewProj, float4(ndc, 1.0));
            boxMin = min(boxMin, world.xyz / world.w);
            boxMax = max(boxMax, world.xyz / world.w);
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
    float4 world = mul(uniforms.inverseViewProj, float4(uv * 2.0 - 1.0, depth, 1.0));
    float3 position = world.xyz / world.w;
    float3 camera = uniforms.cameraPosition;
    float distance = length(position - camera);
    // Rays start just off the surface toward the camera, which is always on its open side.
    float3 toCamera = (camera - position) / max(distance, 1e-6);
    float3 onSurface = toGrid(position);
    float3 gridPosition = onSurface + toCamera * SHADOW_RAY_OFFSET;

    float3 visible = float3(1.0);
    if (uniforms.lighting.gridLevels > 0) {
        visible = sunVisibility(gridPosition, uniforms.lighting.sunDirection);
    }
    shadowsStorage[threadId.xy] = float4(visible, distance);

    // The voxel the pixel shows and the face it's seen through, see-through ones too, found by
    // a short ray from just in front of it, then the light reaching that face from everywhere
    // but the sun, and what it reflects when it's glossy.
    float4 reflection = float4(0.0);
    float4 indirect = float4(0.0);
    bool lightsIndirect = uniforms.tracedIndirect != 0;
    if ((uniforms.trace.traceReflections != 0 || lightsIndirect) && uniforms.lighting.gridLevels > 0) {
        float3 view = -toCamera;
        float3 start = onSurface - view;
        VoxelHit surface;
        if (traceVoxels(start, view, 2.0, false, true, surface) && surface.distance > 0.0) {
            float3 onFace = start + view * surface.distance;
            float face = float(faceNumber(surface.normal));
            if (lightsIndirect) {
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
                float3 onSurface = onFace + surface.normal * SHADOW_RAY_OFFSET;
                float3 reflected = radianceAlong(onSurface, reflect(view, surface.normal), uniforms.lighting.traceDistance);
                reflection = float4(reflected, face);
            }
        }
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
        float lit = lightShadow(lights[brightest.lights[k]], gridPosition);
        result.x = (result.x & ~(NO_LIGHT << (k * 8))) | (brightest.lights[k] << (k * 8));
        result.y |= uint(round(saturate(lit) * 255.0)) << (k * 8);
    }
    lightShadowsStorage[threadId.xy] = result;
}

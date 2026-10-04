// Shadows from the frame's depth, traced at a fraction of the frame's resolution, one ray for
// each block of pixels and light, for shading to upscale, see `upscaledShadow` in
// `mesh.slang`.
//
// `cullLights` first lists the lights reaching each 16x16 pixel tile, then `traceShadows`
// traces the sun's shadows and those of each block's brightest few lights.

#include "common.slang"

// Each texel's sun visibility, tinted by see-through voxels, and its surface's distance from
// the camera.
[[vk::binding(13, 0)]] [[vk::image_format("rgba32f")]] RWTexture2D<float4> shadowImage;
[[vk::binding(14, 0)]] Sampler2D<float> depthBuffer;
// For each tile, how many lights reach it, then which, `TILE_LIGHTS + 1` words a tile.
[[vk::binding(18, 0)]] RWStructuredBuffer<uint> tileLights;
// Each texel's shadowed lights, 8 bits each in x, `0xff` for none, and how much of each gets
// through, 8 bits each in y.
[[vk::binding(19, 0)]] [[vk::image_format("rg32ui")]] RWTexture2D<uint2> lightShadowImage;
// Each texel's sharp reflection when its voxel is glossy, and its face's number, 0 for none.
[[vk::binding(21, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> reflectionImage;
// Each texel's light from everywhere but the sun, with its ambient occlusion, and its face's
// number, 0 for none.
[[vk::binding(23, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> indirectImage;

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
    uint2 frameSize = ubo.shadow.frameSize.xy;
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
            float4 world = mul(ubo.shadow.inverseViewProj, float4(ndc, 1.0));
            boxMin = min(boxMin, world.xyz / world.w);
            boxMax = max(boxMax, world.xyz / world.w);
        }
        for (uint i = index; i < ubo.shadow.lights.x; i += LIGHT_TILE * LIGHT_TILE) {
            Light light = lights[i];
            float3 nearest = clamp(light.positionRange.xyz, boxMin, boxMax);
            float distance = length(nearest - light.positionRange.xyz);
            if (distance < light.positionRange.w) {
                uint slot;
                InterlockedAdd(tileCount, 1, slot);
                tileList[slot] = i;
                float x = distance / light.positionRange.w;
                float window = saturate(1.0 - x * x * x * x);
                tileStrengths[slot] = dot(light.colorSpot.rgb, float3(0.3, 0.6, 0.1)) * window * window / (distance * distance + 1.0);
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // With more than fit, the brightest are kept, ranked by how many are brighter.
    uint tile = group.y * ubo.shadow.lights.y + group.x;
    uint total = tileCount;
    if (index == 0) {
        tileLights[tile * (TILE_LIGHTS + 1)] = min(total, TILE_LIGHTS);
    }
    for (uint i = index; i < total; i += LIGHT_TILE * LIGHT_TILE) {
        uint rank = i;
        if (total > TILE_LIGHTS) {
            rank = 0;
            float strength = tileStrengths[i];
            for (uint j = 0; j < total; j++) {
                float other = tileStrengths[j];
                rank += (other > strength || (other == strength && j < i)) ? 1 : 0;
            }
        }
        if (rank < TILE_LIGHTS) {
            tileLights[tile * (TILE_LIGHTS + 1) + 1 + rank] = tileList[i];
        }
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void traceShadows(uint3 threadId: SV_DispatchThreadID) {
    uint scale = ubo.shadow.params.x;
    uint2 size = ubo.shadow.params.yz;
    uint2 frameSize = ubo.shadow.frameSize.xy;
    if (any(threadId.xy >= size)) {
        return;
    }
    // The pixel at the block's center.
    uint2 pixel = min(threadId.xy * scale + scale / 2, frameSize - 1);
    float2 uv = (float2(pixel) + 0.5) / float2(frameSize);
    float depth = depthBuffer.SampleLevel(uv, 0.0);
    if (depth >= 1.0) {
        // Nothing drawn, the sky.
        shadowImage[threadId.xy] = float4(1.0, 1.0, 1.0, 1e9);
        lightShadowImage[threadId.xy] = uint2(0xffffffff, 0);
        reflectionImage[threadId.xy] = float4(0.0);
        indirectImage[threadId.xy] = float4(0.0);
        return;
    }
    float4 world = mul(ubo.shadow.inverseViewProj, float4(uv * 2.0 - 1.0, depth, 1.0));
    float3 position = world.xyz / world.w;
    float3 camera = ubo.gi.cameraPosition.xyz;
    float distance = length(position - camera);
    // Rays start just off the surface toward the camera, which is always on its open side.
    float3 toCamera = (camera - position) / max(distance, 1e-6);
    float3 gridPosition = position / max(ubo.lighting.shadowGrid.x, 1e-6) + toCamera * SHADOW_RAY_OFFSET;

    float3 visible = float3(1.0);
    if (ubo.lighting.shadowTable.w > 0) {
        visible = sunVisibility(gridPosition, ubo.lighting.sunDirection.xyz);
    }
    shadowImage[threadId.xy] = float4(visible, distance);

    // The voxel the pixel shows and the face it's seen through, found by a short ray from
    // just in front of it, then the light reaching that face from everywhere but the sun, and
    // what it reflects when it's glossy.
    float4 reflection = float4(0.0);
    float4 indirect = float4(0.0);
    bool lightsIndirect = ubo.shadow.frameSize.w != 0;
    if ((ubo.shadow.params.w != 0 || lightsIndirect) && ubo.lighting.shadowTable.w > 0) {
        float3 view = -toCamera;
        float3 start = position / ubo.lighting.shadowGrid.x - view;
        VoxelHit surface;
        if (traceVoxels(start, view, 2.0, false, surface) && surface.distance > 0.0) {
            float3 onFace = start + view * surface.distance;
            float face = float(faceNumber(surface.normal));
            if (lightsIndirect) {
                float3 light = indirectLight(onFace, surface.normal, toCamera, false);
                float occlusionStrength = ubo.lighting.options.x;
                if (occlusionStrength > 0.0) {
                    float3 inLevel = onFace / float(1 << surface.level);
                    light *= lerp(1.0, gridOpenness(surface.level, inLevel, int3(surface.normal)), occlusionStrength);
                }
                indirect = float4(light, face);
            }
            Material material = materialOf(surface.id);
            if (ubo.shadow.params.w != 0 && material.colorRoughness.w < GLOSSY_ROUGHNESS) {
                float3 onSurface = onFace + surface.normal * SHADOW_RAY_OFFSET;
                float3 reflected = radianceAlong(onSurface, reflect(view, surface.normal), ubo.lighting.shadowGrid.z);
                reflection = float4(reflected, face);
            }
        }
    }
    reflectionImage[threadId.xy] = reflection;
    indirectImage[threadId.xy] = indirect;

    // The block's tile's brightest lights here, as many as are shadowed, brightest first.
    uint shadowed = ubo.shadow.lights.w;
    uint brightest[4] = { 0xff, 0xff, 0xff, 0xff };
    float strengths[4] = { 0.0, 0.0, 0.0, 0.0 };
    uint tile = (pixel.y / LIGHT_TILE) * ubo.shadow.lights.y + pixel.x / LIGHT_TILE;
    uint count = tileLights[tile * (TILE_LIGHTS + 1)];
    for (uint i = 0; i < count; i++) {
        uint lightIndex = tileLights[tile * (TILE_LIGHTS + 1) + 1 + i];
        float strength = lightStrength(lights[lightIndex], position);
        for (uint k = 0; k < shadowed; k++) {
            if (strength > strengths[k]) {
                for (uint j = shadowed - 1; j > k; j--) {
                    strengths[j] = strengths[j - 1];
                    brightest[j] = brightest[j - 1];
                }
                strengths[k] = strength;
                brightest[k] = lightIndex;
                break;
            }
        }
    }
    uint2 result = uint2(0xffffffff, 0);
    for (uint k = 0; k < shadowed; k++) {
        if (brightest[k] == 0xff) {
            break;
        }
        float lit = lightShadow(lights[brightest[k]], gridPosition);
        result.x = (result.x & ~(0xffu << (k * 8))) | (brightest[k] << (k * 8));
        result.y |= uint(round(saturate(lit) * 255.0)) << (k * 8);
    }
    lightShadowImage[threadId.xy] = result;
}

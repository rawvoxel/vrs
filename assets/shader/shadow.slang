// Traces the sun's shadows at a fraction of the frame's resolution, one ray for each block of
// pixels, from the frame's depth. Shading upscales them, see `upscaledShadow` in `mesh.slang`.

#include "common.slang"

// Each texel's sun visibility, and its surface's distance from the camera.
[[vk::binding(13, 0)]] [[vk::image_format("rg32f")]] RWTexture2D<float2> shadowImage;
[[vk::binding(14, 0)]] Sampler2D<float> depthBuffer;

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
        shadowImage[threadId.xy] = float2(1.0, 1e9);
        return;
    }
    float4 world = mul(ubo.shadow.inverseViewProj, float4(uv * 2.0 - 1.0, depth, 1.0));
    float3 position = world.xyz / world.w;
    float3 camera = ubo.gi.cameraPosition.xyz;
    float distance = length(position - camera);
    float visible = 1.0;
    if (ubo.lighting.shadowTable.w > 0) {
        // Starts just off the surface toward the camera, which is always on its open side.
        float3 toCamera = (camera - position) / max(distance, 1e-6);
        float3 gridPosition = position / ubo.lighting.shadowGrid.x + toCamera * SHADOW_RAY_OFFSET;
        visible = sunVisibility(gridPosition, ubo.lighting.sunDirection.xyz);
    }
    shadowImage[threadId.xy] = float2(visible, distance);
}

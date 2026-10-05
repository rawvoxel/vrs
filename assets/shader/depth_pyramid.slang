// Builds one level of the depth pyramid used for occlusion culling.
// Each texel keeps the farthest depth of the 2x2 source texels it covers.

// Matches `renderer::culling::depth_pyramid::ReduceParams`.
struct ReduceParams {
    uint2 srcSize;
    uint2 dstSize;
};

[[vk::push_constant]] ConstantBuffer<ReduceParams> params;

[[vk::binding(0, 0)]] Texture2D<float> src;
[[vk::binding(1, 0)]] [[vk::image_format("r32f")]] RWTexture2D<float> dst;

[shader("compute")]
[numthreads(8, 8, 1)]
void reduceDepth(uint3 threadId: SV_DispatchThreadID) {
    uint2 texel = threadId.xy;
    if (any(texel >= params.dstSize)) {
        return;
    }

    // Levels are rounded up, so the last row or column may only cover one source texel.
    uint2 base = texel * 2;
    uint2 last = params.srcSize - 1;
    float depth = max(
        max(src.Load(int3(min(base, last), 0)), src.Load(int3(min(base + uint2(1, 0), last), 0))),
        max(src.Load(int3(min(base + uint2(0, 1), last), 0)), src.Load(int3(min(base + 1, last), 0))));
    dst[texel] = depth;
}

// Draws `renderer::Sky` where nothing else was drawn: a triangle covering the screen at the
// far plane, behind everything, as each pixel's view direction sees it.

#include "common.slang"

struct SkyOutput {
    float4 position : SV_Position;
    // In clip space, from -1 to 1 across the screen.
    float2 clip;
};

[shader("vertex")]
SkyOutput skyVert(uint vertexIndex: SV_VertexID) {
    SkyOutput output;
    output.clip = float2((vertexIndex << 1) & 2, vertexIndex & 2) * 2.0 - 1.0;
    output.position = float4(output.clip, 1.0, 1.0);
    return output;
}

[shader("fragment")]
float4 skyFrag(SkyOutput input) : SV_Target {
    // Two points along the pixel's ray, neither at the far plane, which an infinite
    // projection puts at infinity.
    float4 near = mul(ubo.shadow.inverseViewProj, float4(input.clip, 0.0, 1.0));
    float4 beyond = mul(ubo.shadow.inverseViewProj, float4(input.clip, 0.5, 1.0));
    float3 direction = normalize(beyond.xyz / beyond.w - near.xyz / near.w);
    return float4(toDisplay(skyRadiance(direction, true)), 1.0);
}

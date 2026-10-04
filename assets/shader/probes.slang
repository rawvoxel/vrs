// Draws the light probes as small cubes, for `renderer::DebugView::Probes`, each face colored
// with the light the probe holds from its direction. Probes not placed or not yet updated
// aren't drawn.

#include "common.slang"

// How wide a probe's cube is, in the shadow grid's finest voxels, for the finest cascade.
static const float PROBE_CUBE = 1.5;

struct ProbeOutput {
    float4 position : SV_Position;
    float3 normal;
    nointerpolation uint slot;
};

[shader("vertex")]
ProbeOutput probeVert(uint vertexIndex: SV_VertexID, uint slot: SV_InstanceID) {
    ProbeOutput output;
    output.slot = slot;
    Probe probe = probes[slot];
    // Off the screen, clipped away.
    output.position = float4(0.0, 0.0, 2.0, 1.0);
    output.normal = float3(0.0, 1.0, 0.0);
    if ((probe.state & PROBE_PLACED) == 0 || probe.updates == 0) {
        return output;
    }

    // 6 faces of 2 triangles, wound counter-clockwise seen from outside.
    uint face = vertexIndex / 6;
    uint corner = uint[6](0, 1, 2, 2, 1, 3)[vertexIndex % 6];
    int axis = int(face / 2);
    float side = (face & 1) != 0 ? 1.0 : -1.0;
    float3 normal = float3(0.0);
    normal[axis] = side;
    float3 u = float3(0.0);
    float3 v = float3(0.0);
    u[(axis + 1) % 3] = 1.0;
    v[(axis + 2) % 3] = 1.0;
    float2 at = float2(corner & 1, corner >> 1) * 2.0 - 1.0;
    // Facing out, the corners wind the other way on the negative sides.
    if (side < 0.0) {
        at.x = -at.x;
    }
    float3 local = (normal + u * at.x + v * at.y) * (PROBE_CUBE * 0.5 * float(1u << (slot / probesPerCascade())));
    float3 world = (probe.position + local) * ubo.lighting.shadowGrid.x;
    output.position = mul(ubo.proj, mul(ubo.view, float4(world, 1.0)));
    output.normal = normal;
    return output;
}

[shader("fragment")]
float4 probeFrag(ProbeOutput input) : SV_Target {
    // Stored to the power of 1 / 5.
    float3 stored = probeIrradiance.SampleLevel(atlasUV(input.slot, input.normal, IRRADIANCE_TEXELS), 0.0).rgb;
    return float4(pow(max(stored, 0.0), IRRADIANCE_GAMMA), 1.0);
}

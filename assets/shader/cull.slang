// Frustum culls a frame's candidate meshes, writing draw commands for the visible ones.

// Matches `mesh::MeshInfo`.
struct MeshInfo {
    float3 aabbMin;
    uint indexCount;
    float3 aabbMax;
    uint firstIndex;
    int vertexOffset;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

// Matches `vk::DrawIndexedIndirectCommand`.
struct DrawIndexedIndirectCommand {
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    int vertexOffset;
    uint firstInstance;
};

// Matches `cull::CullParams`.
struct CullParams {
    // Inward facing, a point is inside when `dot(plane.xyz, p) + plane.w >= 0`.
    float4 planes[6];
    uint candidateCount;
    // Non-zero appends visible draws and counts them, otherwise every candidate
    // keeps its own slot and culled draws get zero instances.
    uint compact;
};

[[vk::push_constant]] ConstantBuffer<CullParams> params;

[[vk::binding(0, 0)]] StructuredBuffer<MeshInfo> meshes;
[[vk::binding(1, 0)]] StructuredBuffer<uint> candidates;
[[vk::binding(2, 0)]] RWStructuredBuffer<DrawIndexedIndirectCommand> draws;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> drawCount;

bool isVisible(MeshInfo mesh) {
    for (uint i = 0; i < 6; i++) {
        float4 plane = params.planes[i];
        // The corner furthest along the plane's normal.
        float3 corner = select(plane.xyz >= 0.0, mesh.aabbMax, mesh.aabbMin);
        if (dot(plane.xyz, corner) + plane.w < 0.0) {
            return false;
        }
    }
    return true;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cullMain(uint3 threadId: SV_DispatchThreadID) {
    uint candidate = threadId.x;
    if (candidate >= params.candidateCount) {
        return;
    }

    MeshInfo mesh = meshes[candidates[candidate]];
    bool visible = mesh.indexCount > 0 && isVisible(mesh);

    DrawIndexedIndirectCommand draw;
    draw.indexCount = mesh.indexCount;
    draw.instanceCount = visible ? 1 : 0;
    draw.firstIndex = mesh.firstIndex;
    draw.vertexOffset = mesh.vertexOffset;
    draw.firstInstance = 0;

    if (params.compact == 0) {
        draws[candidate] = draw;
    } else if (visible) {
        uint slot;
        InterlockedAdd(drawCount[0], 1, slot);
        draws[slot] = draw;
    }
}

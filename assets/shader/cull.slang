// Culls a frame's candidate meshes, writing draw commands for the visible ones.
//
// Occlusion culling runs in two phases around a depth pyramid built in between:
// phase 1 draws what was visible last frame, phase 2 tests everything against the
// pyramid, draws what's newly visible and remembers what's visible for next frame.

// Matches `renderer::mesh_pool::MeshInfo`, and `MeshInfo` in `common/layout.slang`.
struct MeshInfo {
    float3 boundsMin;
    uint indexCount;
    float3 boundsMax;
    uint firstIndex;
    float3 origin;
    float scale;
    uint vertexBase;
    uint typesBase;
    uint kind;
    // Non-zero when phase 2 last found the mesh visible.
    uint visible;
    uint voxelsPerEdge;
    float positionSteps;
    uint occupancyBase;
    uint probesBase;
};

// Matches `vk::DrawIndexedIndirectCommand`.
struct DrawIndexedIndirectCommand {
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    int vertexOffset;
    uint firstInstance;
};

// Matches `renderer::culling::Phase`.
static const uint PHASE_FRUSTUM = 0;
static const uint PHASE_LAST_VISIBLE = 1;
static const uint PHASE_OCCLUSION = 2;

// Matches `renderer::culling::CullParams`.
struct CullParams {
    // Rows of the view projection matrix.
    float4 viewProj[4];
    // Size of the depth buffer the pyramid was built from.
    uint2 depthSize;
    uint candidateCount;
    uint phase;
    uint pyramidLevels;
};

[[vk::push_constant]] ConstantBuffer<CullParams> params;

[[vk::binding(0, 0)]] RWStructuredBuffer<MeshInfo> meshes;
[[vk::binding(1, 0)]] StructuredBuffer<uint> candidates;
// Visible draws are appended and counted. Phase 2 writes after phase 1's `candidateCount`
// draws, and counts in `drawCount[1]`.
[[vk::binding(2, 0)]] RWStructuredBuffer<DrawIndexedIndirectCommand> draws;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> drawCount;
// Each level's texels hold the farthest depth they cover, level 0 is half the depth size.
[[vk::binding(4, 0)]] Texture2D<float> depthPyramid;

float4 toClip(float3 position) {
    float4 p = float4(position, 1.0);
    return float4(
        dot(params.viewProj[0], p),
        dot(params.viewProj[1], p),
        dot(params.viewProj[2], p),
        dot(params.viewProj[3], p));
}

bool inFrustum(MeshInfo mesh) {
    float4 x = params.viewProj[0];
    float4 y = params.viewProj[1];
    float4 z = params.viewProj[2];
    float4 w = params.viewProj[3];
    // Inward facing for Vulkan's 0..1 depth, matches `renderer::culling::frustum_planes`.
    float4 planes[6] = { w + x, w - x, w + y, w - y, z, w - z };
    for (uint i = 0; i < 6; i++) {
        float4 plane = planes[i];
        // The corner furthest along the plane's normal.
        float3 corner = select(plane.xyz >= 0.0, mesh.boundsMax, mesh.boundsMin);
        if (dot(plane.xyz, corner) + plane.w < 0.0) {
            return false;
        }
    }
    return true;
}

bool isOccluded(MeshInfo mesh) {
    float2 ndcMin = float2(1.0, 1.0);
    float2 ndcMax = float2(-1.0, -1.0);
    float nearest = 1.0;
    for (uint corner = 0; corner < 8; corner++) {
        float3 position = select(
            bool3((corner & 1) != 0, (corner & 2) != 0, (corner & 4) != 0),
            mesh.boundsMax,
            mesh.boundsMin);
        float4 clip = toClip(position);
        // Boxes reaching behind the camera can't be projected, so count them as visible.
        if (clip.w <= 0.0) {
            return false;
        }
        float3 ndc = clip.xyz / clip.w;
        ndcMin = min(ndcMin, ndc.xy);
        ndcMax = max(ndcMax, ndc.xy);
        nearest = min(nearest, ndc.z);
    }
    if (nearest <= 0.0) {
        return false;
    }

    // Depth pixels the box covers, Vulkan's NDC y points down like the image's rows.
    uint2 last = params.depthSize - 1;
    uint2 pixelMin = min(uint2(saturate(ndcMin * 0.5 + 0.5) * float2(params.depthSize)), last);
    uint2 pixelMax = min(uint2(saturate(ndcMax * 0.5 + 0.5) * float2(params.depthSize)), last);

    // The first level where the box spans at most 2x2 texels, so 4 loads cover it.
    // Level `l` texels each cover `2^(l + 1)` depth pixels.
    uint span = max(pixelMax.x - pixelMin.x, pixelMax.y - pixelMin.y);
    uint level = span == 0 ? 0 : firstbithigh(span);
    level = min(level, params.pyramidLevels - 1);
    uint shift = level + 1;
    uint2 texelMin = pixelMin >> shift;
    uint2 texelMax = pixelMax >> shift;

    float farthest = max(
        max(depthPyramid.Load(int3(texelMin, level)), depthPyramid.Load(int3(texelMax.x, texelMin.y, level))),
        max(depthPyramid.Load(int3(texelMin.x, texelMax.y, level)), depthPyramid.Load(int3(texelMax, level))));
    return nearest > farthest;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cullMeshes(uint3 threadId: SV_DispatchThreadID) {
    uint candidate = threadId.x;
    if (candidate >= params.candidateCount) {
        return;
    }

    uint meshIndex = candidates[candidate];
    MeshInfo mesh = meshes[meshIndex];
    bool frustumVisible = mesh.indexCount > 0 && inFrustum(mesh);

    bool visible;
    if (params.phase == PHASE_FRUSTUM) {
        visible = frustumVisible;
    } else if (params.phase == PHASE_LAST_VISIBLE) {
        visible = frustumVisible && mesh.visible != 0;
    } else {
        bool visibleNow = frustumVisible && !isOccluded(mesh);
        // Phase 1 already drew what was visible last frame.
        visible = visibleNow && mesh.visible == 0;
        meshes[meshIndex].visible = visibleNow ? 1 : 0;
    }

    if (!visible) {
        return;
    }
    DrawIndexedIndirectCommand draw;
    draw.indexCount = mesh.indexCount;
    draw.instanceCount = 1;
    draw.firstIndex = mesh.firstIndex;
    // The mesh shaders pull vertices themselves, finding the mesh by its instance.
    draw.vertexOffset = 0;
    draw.firstInstance = meshIndex;

    uint firstDraw = params.phase == PHASE_OCCLUSION ? params.candidateCount : 0;
    uint slot;
    InterlockedAdd(drawCount[params.phase == PHASE_OCCLUSION ? 1 : 0], 1, slot);
    draws[firstDraw + slot] = draw;
}

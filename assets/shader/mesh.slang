// Draws the mesh pool's meshes, pulling each one's vertices in whichever format it uses.
//
// Colored meshes carry their own colors. Chunk meshes are colored by voxel: a fragment finds
// the voxel it belongs to in the chunk's packed types, then that type's palette color.

// Matches `mesh::MeshInfo`, and `MeshInfo` in `cull.slang`.
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
    uint visible;
    uint voxelsPerEdge;
    float positionSteps;
    uint _pad0;
    uint _pad1;
};

// Matches `uniforms::MvpMatrices`.
struct Uniforms {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
};

// Matches `mesh::MeshKind`.
static const uint KIND_COLORED = 0;
static const uint KIND_GREEDY = 1;
static const uint KIND_PACKED = 2;

// Matches `voxel::POSITION_BIAS`.
static const float POSITION_BIAS = 16.0;
// Matches `voxel::Face::normal`, indexed by the face in a greedy vertex's flags bits 1 to 3.
static const float3 FACE_NORMALS[6] = {
    float3(-1.0, 0.0, 0.0),
    float3(1.0, 0.0, 0.0),
    float3(0.0, -1.0, 0.0),
    float3(0.0, 1.0, 0.0),
    float3(0.0, 0.0, -1.0),
    float3(0.0, 0.0, 1.0),
};
// How far inside a face, in voxels, its fragments are moved to find their voxel: far above
// the float error of an interpolated position, far below a voxel.
static const float FACE_NUDGE = 1.0 / 256.0;

[[vk::binding(0, 0)]] ConstantBuffer<Uniforms> ubo;
[[vk::binding(1, 0)]] StructuredBuffer<uint> vertexWords;
[[vk::binding(2, 0)]] StructuredBuffer<MeshInfo> meshes;
// Every chunk's voxel types, laid out as `voxel::VoxelTypes` packs them.
[[vk::binding(3, 0)]] StructuredBuffer<uint> typeWords;
// Linear colors indexed by voxel type.
[[vk::binding(4, 0)]] StructuredBuffer<float4> palette;

struct VSOutput {
    float4 position : SV_Position;
    float3 color;
    // Chunk local, in the chunk's voxels.
    float3 voxelPosition;
    // Carried for lighting, which the renderer doesn't do yet.
    float3 normal;
    nointerpolation uint kind;
    nointerpolation uint typesBase;
    nointerpolation uint voxelsPerEdge;
};

// Matches `voxel::octahedral_decode`.
float3 octahedralDecode(uint word) {
    float2 folded = float2(word & 0xffff, word >> 16) / 65535.0 * 2.0 - 1.0;
    float3 normal = float3(folded.x, 1.0 - abs(folded.x) - abs(folded.y), folded.y);
    float below = max(-normal.y, 0.0);
    normal.x += normal.x >= 0.0 ? -below : below;
    normal.z += normal.z >= 0.0 ? -below : below;
    return normalize(normal);
}

// Matches `voxel::unpack_position`.
float3 unpackPosition(uint positionXy, uint positionZFlags, float steps) {
    float3 quantized = float3(positionXy & 0xffff, positionXy >> 16, positionZFlags & 0xffff);
    return quantized / steps - POSITION_BIAS;
}

[shader("vertex")]
VSOutput meshVert(uint vertexIndex: SV_VertexID, uint meshIndex: SV_StartInstanceLocation) {
    MeshInfo mesh = meshes[meshIndex];

    VSOutput output;
    output.kind = mesh.kind;
    output.typesBase = mesh.typesBase;
    output.voxelsPerEdge = mesh.voxelsPerEdge;
    output.color = float3(1.0, 0.0, 1.0);
    output.normal = float3(0.0, 1.0, 0.0);

    float3 local;
    uint word = mesh.vertexBase;
    if (mesh.kind == KIND_COLORED) {
        word += vertexIndex * 4;
        local = asfloat(uint3(vertexWords[word], vertexWords[word + 1], vertexWords[word + 2]));
        output.color = unpackUnorm4x8ToFloat(vertexWords[word + 3]).rgb;
    } else if (mesh.kind == KIND_GREEDY) {
        word += vertexIndex * 2;
        uint positionZFlags = vertexWords[word + 1];
        local = unpackPosition(vertexWords[word], positionZFlags, mesh.positionSteps);
        output.normal = FACE_NORMALS[min((positionZFlags >> 17) & 7, 5)];
    } else {
        word += vertexIndex * 3;
        local = unpackPosition(vertexWords[word], vertexWords[word + 1], mesh.positionSteps);
        output.normal = octahedralDecode(vertexWords[word + 2]);
    }
    output.voxelPosition = local;

    float4 world = float4(mesh.origin + local * mesh.scale, 1.0);
    output.position = mul(ubo.proj, mul(ubo.view, world));
    return output;
}

// The type id of voxel `cell` of the chunk whose types start at word `base`.
// Matches `voxel::VoxelTypes::type_at`.
uint typeAt(uint base, uint edge, uint3 cell) {
    uint header = typeWords[base];
    uint bits = header & 0xff;
    uint perWord = 32 / bits;
    uint index = cell.x + edge * (cell.y + edge * cell.z);
    uint word = typeWords[base + 1 + (header >> 8) + index / perWord];
    uint entry = (word >> ((index % perWord) * bits)) & ((1u << bits) - 1);
    uint paletteWord = typeWords[base + 1 + (entry >> 1)];
    return (paletteWord >> ((entry & 1) * 16)) & 0xffff;
}

// The chunk voxel containing voxel position `p`. Past the chunk's edges, its edge voxel.
uint3 cellAt(float3 p, uint edge) {
    return uint3(clamp(int3(floor(p)), int3(0), int3(int(edge) - 1)));
}

[shader("fragment")]
float4 meshFrag(VSOutput input) : SV_Target {
    // The triangle's outward normal, from screen derivatives. Taken before
    // any branch, so every pixel of the quad takes part.
    float3 faceNormal = normalize(cross(ddy(input.voxelPosition), ddx(input.voxelPosition)));
    if (input.kind == KIND_COLORED) {
        return float4(input.color, 1.0);
    }

    // A face lies between its solid voxel and the empty cell in front, so its fragments are
    // nudged inside first. Curved surfaces can still be in an empty cell, then the voxel
    // one cell behind is taken.
    uint edge = input.voxelsPerEdge;
    uint id = typeAt(input.typesBase, edge, cellAt(input.voxelPosition - faceNormal * FACE_NUDGE, edge));
    if (id == 0) {
        id = typeAt(input.typesBase, edge, cellAt(input.voxelPosition - faceNormal * (1.0 + FACE_NUDGE), edge));
    }

    uint paletteSize;
    uint stride;
    palette.GetDimensions(paletteSize, stride);
    // Types the palette doesn't cover show magenta.
    float3 color = id < paletteSize ? palette[id].rgb : float3(1.0, 0.0, 1.0);
    return float4(color, 1.0);
}

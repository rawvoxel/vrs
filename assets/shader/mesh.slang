// Draws the mesh pool's meshes, pulling each one's vertices in whichever format it uses.
//
// Colored meshes carry their own colors. Chunk meshes are colored by voxel: a fragment finds
// the voxel it belongs to in the chunk's packed types, then that type's palette color. Then
// they're lit by the sun, whose shadows are traced through the voxels, and the light probes,
// or the sky and the ground outside them.

#include "common.slang"

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

struct VSOutput {
    float4 position : SV_Position;
    float3 color;
    // Chunk local, in the chunk's voxels.
    float3 voxelPosition;
    float3 worldPosition;
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
    output.worldPosition = world.xyz;
    output.position = mul(ubo.proj, mul(ubo.view, world));
    return output;
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

    // Greedy normals are exact already, packed ones are interpolated between vertices.
    float3 normal = input.kind == KIND_GREEDY ? input.normal : normalize(input.normal);
    float3 toViewer = normalize(ubo.gi.cameraPosition.xyz - input.worldPosition);
    float3 light = lightAt(input.worldPosition, normal, faceNormal, toViewer, false);
    return float4(paletteColor(id) * light, 1.0);
}

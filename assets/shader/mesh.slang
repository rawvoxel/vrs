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
// How much nearer or further, relative to its distance, a shadow texel's surface can be than
// a pixel's and still shadow it.
static const float SHADOW_DEPTH_TOLERANCE = 0.08;

// The sun's visibility and the surface's distance for each block of pixels, when shadows are
// traced at a fraction of the frame's resolution, see `shadow.slang`.
[[vk::binding(15, 0)]] Sampler2D<float4> shadowTexels;

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

// The sun's visibility at the frame pixel centered on `pixel`, whose surface is `distance`
// from the camera, upscaled from the shadow texels: the 4 nearest, weighted by how near each
// is, of those whose surface is about as far. With none, the one whose surface is nearest in
// distance, so shadows don't bleed across edges.
float upscaledShadow(float2 pixel, float distance) {
    float scale = float(ubo.shadow.params.x);
    float2 size = float2(ubo.shadow.params.yz);
    // Each texel holds the pixel at its block's center.
    float2 texel = (pixel - 0.5 - floor(scale * 0.5)) / scale;
    int2 base = int2(floor(texel));
    float2 alpha = texel - float2(base);
    float sum = 0.0;
    float totalWeight = 0.0;
    float closest = 1e9;
    float closestVisible = 1.0;
    for (int i = 0; i < 4; i++) {
        int2 offset = int2(i & 1, i >> 1);
        float2 tap = clamp(float2(base + offset), 0.0, size - 1.0);
        float2 shadow = shadowTexels.SampleLevel((tap + 0.5) / size, 0.0).xy;
        float difference = abs(shadow.y - distance) / max(distance, 1e-3);
        if (difference < closest) {
            closest = difference;
            closestVisible = shadow.x;
        }
        float2 bilinear = lerp(1.0 - alpha, alpha, float2(offset));
        float weight = bilinear.x * bilinear.y * saturate(1.0 - difference / SHADOW_DEPTH_TOLERANCE);
        sum += weight * shadow.x;
        totalWeight += weight;
    }
    return totalWeight > 1e-4 ? sum / totalWeight : closestVisible;
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
    float3 fromCamera = input.worldPosition - ubo.gi.cameraPosition.xyz;
    float3 toViewer = -normalize(fromCamera);
    float sunVisible = ubo.shadow.params.x > 1 ? upscaledShadow(input.position.xy, length(fromCamera)) : -1.0;
    float3 light = lightAt(input.worldPosition, normal, faceNormal, toViewer, false, sunVisible);
    return float4(paletteColor(id) * light, 1.0);
}

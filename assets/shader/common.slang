// Shared by `mesh.slang` and `gi.slang`: the frame's bindings, decoding voxels, tracing rays
// through them, and lighting from the sun, the sky and the light probes.
//
// `gi.slang` defines `GI_WRITES` first, for the bindings only its compute shaders write.

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
    uint occupancyBase;
    uint probesBase;
};

// Matches `renderer::LightingUniforms`.
struct LightingUniforms {
    // Normalized, toward the sun.
    float4 sunDirection;
    float4 sunColor;
    float4 skyColor;
    float4 groundColor;
    // The chunk coordinate of the shadow table's corner, and chunks per side, 0 without shadows.
    int4 shadowTable;
    // The grid's voxel size, voxels per edge and the shadow distance in voxels.
    float4 shadowGrid;
};

// Matches `gi::GiUniforms`.
struct GiUniforms {
    // The probe cell at the volume's low corner, w is 1 when the volume is lit this frame.
    int4 volumeMin;
    // Probes along each axis, w is probes per row of the atlases.
    uint4 volumeCounts;
    // This frame's random rotation of the probes' rays, a quaternion.
    float4 rayRotation;
    float4 cameraPosition;
    // x is the frame's number.
    uint4 frame;
};

// Matches `uniforms::FrameUniforms`.
struct Uniforms {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
    LightingUniforms lighting;
    GiUniforms gi;
};

// A light probe, matches `gi::PROBE_SIZE`. Positions are in the shadow grid's voxels.
struct Probe {
    // The probe cell the probe's slot holds.
    int3 cell;
    // `PROBE_PLACED` when the cell has room for the probe.
    uint state;
    float3 position;
    // How many frames have updated it since it was placed.
    uint updates;
};

static const uint PROBE_PLACED = 1;

// Matches `lighting::NO_CHUNK`, a shadow table entry without a chunk.
static const uint NO_CHUNK = 0xffffffff;
// How far off a surface, in voxels, its shadow ray starts, so it starts in the cell in front.
static const float SHADOW_RAY_OFFSET = 1.0 / 64.0;
// Matches `voxel::OCCUPANCY_BLOCK`, voxels along each edge of a block of a chunk's occupancy.
static const int OCCUPANCY_BLOCK = 8;
// Moves a ray just past the boundary it's on, in voxels.
static const float RAY_EPSILON = 1.0 / 4096.0;
// Bounds on the work one ray does, so none can spin.
static const uint MAX_RAY_CHUNKS = 256;
static const uint MAX_RAY_VOXELS = 1024;

// Voxels along each edge of a probe's cell, and between neighboring probes. Matches
// `gi::PROBE_CELL`.
static const int PROBE_SPACING = 16;
// Matches `gi::IRRADIANCE_TEXELS` and `gi::DEPTH_TEXELS`, each with a one texel border.
static const int IRRADIANCE_TEXELS = 6;
static const int DEPTH_TEXELS = 14;
// Irradiance is stored to the power of 1 / 5, which blends a drop in light as fast as a
// rise looks [Majercik et al. 2021].
static const float IRRADIANCE_GAMMA = 5.0;
// How far probe visibility is tested away from the surface, in probe spacings
// [Majercik et al. 2021].
static const float SELF_SHADOW_BIAS = 0.3;

[[vk::binding(0, 0)]] ConstantBuffer<Uniforms> ubo;
[[vk::binding(1, 0)]] StructuredBuffer<uint> vertexWords;
[[vk::binding(2, 0)]] StructuredBuffer<MeshInfo> meshes;
// Every chunk's voxel types, laid out as `voxel::VoxelTypes` packs them, then its occupancy.
[[vk::binding(3, 0)]] StructuredBuffer<uint> typeWords;
// Linear colors indexed by voxel type.
[[vk::binding(4, 0)]] StructuredBuffer<float4> palette;
// The mesh index of the chunk at each chunk coordinate around the camera, or `NO_CHUNK`.
// `lighting.shadowTable.w` chunks per side, x-fastest.
[[vk::binding(5, 0)]] StructuredBuffer<uint> shadowTable;
// Each probe slot's probe, slots x-fastest by probe cell modulo the volume's size.
#ifdef GI_WRITES
[[vk::binding(6, 0)]] RWStructuredBuffer<Probe> probes;
#else
[[vk::binding(6, 0)]] StructuredBuffer<Probe> probes;
#endif
// Each probe's irradiance and distance maps, octahedral with a border, in rows of
// `gi.volumeCounts.w` probes. Distance maps hold the mean distance and its square.
[[vk::binding(7, 0)]] Sampler2D<float4> probeIrradiance;
[[vk::binding(8, 0)]] Sampler2D<float4> probeDepth;

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

// The color of voxel type `id`. Types the palette doesn't cover show magenta.
float3 paletteColor(uint id) {
    uint paletteSize;
    uint stride;
    palette.GetDimensions(paletteSize, stride);
    return id < paletteSize ? palette[id].rgb : float3(1.0, 0.0, 1.0);
}

// The mesh index of the chunk at chunk coordinate `chunk`, `NO_CHUNK` for none, or `outside`
// set when it's past the shadow table.
uint shadowChunk(int3 chunk, out bool outside) {
    int side = ubo.lighting.shadowTable.w;
    int3 relative = chunk - ubo.lighting.shadowTable.xyz;
    outside = any(relative < 0) || any(relative >= side);
    if (outside) {
        return NO_CHUNK;
    }
    return shadowTable[(relative.z * side + relative.y) * side + relative.x];
}

// Whether any voxel of block `block` of the chunk whose occupancy starts at word `base` isn't
// empty. Matches `voxel::VoxelTypes::occupancy`.
bool blockOccupied(uint base, int blocksPerEdge, int3 block) {
    uint index = uint(block.x + blocksPerEdge * (block.y + blocksPerEdge * block.z));
    return (typeWords[base + index / 32] & (1u << (index % 32))) != 0;
}

// Where a ray from `origin` along `direction`, with `inverse` its reciprocal, leaves the box
// from `boxMin` to `boxMax`, and across which `axis`. At least `t` plus a little, so the ray
// always moves on.
float exitDistance(float3 origin, float3 direction, float3 inverse, float3 boxMin, float3 boxMax, float t, out int axis) {
    float3 exits = (select(direction > 0.0, boxMax, boxMin) - origin) * inverse;
    axis = exits.x < exits.y ? (exits.x < exits.z ? 0 : 2) : (exits.y < exits.z ? 1 : 2);
    return max(exits[axis], t + RAY_EPSILON);
}

// The first solid voxel along a ray.
struct VoxelHit {
    // How far along the ray it starts.
    float distance;
    // The outward normal of the face the ray entered through.
    float3 normal;
    // Its type.
    uint id;
};

// Whether a ray from `origin` along unit `direction`, both in the shadow grid's voxels, enters
// a solid voxel within `maxDistance`, and which. Rays leaving the shadow table hit nothing.
//
// An Amanatides-Woo DDA, visiting every voxel the ray passes through in order. It runs a
// chunk at a time, crossing a chunk without voxels in one step, and within a chunk crosses
// each block its occupancy says is empty in one step. With `skipStartSolid`, solid voxels the
// ray starts in are the surface it starts on, such as on a curved surface, so they're skipped.
bool traceVoxels(float3 origin, float3 direction, float maxDistance, bool skipStartSolid, out VoxelHit hit) {
    hit.distance = 0.0;
    hit.normal = float3(0.0);
    hit.id = 0;
    int edge = int(ubo.lighting.shadowGrid.y);
    int blocksPerEdge = (edge + OCCUPANCY_BLOCK - 1) / OCCUPANCY_BLOCK;
    // A ray along an axis would divide by zero.
    direction = select(abs(direction) < 1e-6, select(direction < 0.0, -1e-6, 1e-6), direction);
    float3 inverse = 1.0 / direction;
    int3 stepDirection = select(direction > 0.0, int3(1), int3(-1));
    float3 delta = abs(inverse);

    float t = 0.0;
    // The axis the ray last crossed a boundary on, whose face it entered through.
    int axis = -1;
    bool leftSurface = !skipStartSolid;
    uint voxelSteps = 0;
    for (uint chunkStep = 0; chunkStep < MAX_RAY_CHUNKS; chunkStep++) {
        if (t > maxDistance) {
            return false;
        }
        int3 cell = int3(floor(origin + direction * (t + RAY_EPSILON)));
        int3 chunk = int3(floor(float3(cell) / float(edge)));
        bool outside;
        uint meshIndex = shadowChunk(chunk, outside);
        if (outside) {
            return false;
        }
        int3 chunkMin = chunk * edge;
        if (meshIndex == NO_CHUNK) {
            // Nothing to hit, so straight to where the ray leaves the chunk.
            t = exitDistance(origin, direction, inverse, float3(chunkMin), float3(chunkMin + edge), t, axis);
            leftSurface = true;
            continue;
        }

        MeshInfo mesh = meshes[meshIndex];
        // Where the ray crosses the next cell boundary on each axis.
        float3 crossings = (float3(cell + select(direction > 0.0, int3(1), int3(0))) - origin) * inverse;
        while (true) {
            int3 local = cell - chunkMin;
            if (any(local < 0) || any(local >= edge)) {
                break;
            }
            if (++voxelSteps > MAX_RAY_VOXELS) {
                return false;
            }
            int3 block = local / OCCUPANCY_BLOCK;
            if (!blockOccupied(mesh.occupancyBase, blocksPerEdge, block)) {
                // Nothing to hit, so straight to where the ray leaves the block.
                int3 blockMin = chunkMin + block * OCCUPANCY_BLOCK;
                int3 blockMax = min(blockMin + OCCUPANCY_BLOCK, chunkMin + edge);
                t = exitDistance(origin, direction, inverse, float3(blockMin), float3(blockMax), t, axis);
                if (t > maxDistance) {
                    return false;
                }
                leftSurface = true;
                cell = int3(floor(origin + direction * (t + RAY_EPSILON)));
                crossings = (float3(cell + select(direction > 0.0, int3(1), int3(0))) - origin) * inverse;
                continue;
            }

            uint id = typeAt(mesh.typesBase, uint(edge), uint3(local));
            if (id != 0) {
                if (leftSurface) {
                    hit.distance = t;
                    hit.id = id;
                    if (axis >= 0) {
                        hit.normal[axis] = -float(stepDirection[axis]);
                    }
                    return true;
                }
            } else {
                leftSurface = true;
            }

            if (crossings.x < crossings.y && crossings.x < crossings.z) {
                axis = 0;
                t = crossings.x;
                cell.x += stepDirection.x;
                crossings.x += delta.x;
            } else if (crossings.y < crossings.z) {
                axis = 1;
                t = crossings.y;
                cell.y += stepDirection.y;
                crossings.y += delta.y;
            } else {
                axis = 2;
                t = crossings.z;
                cell.z += stepDirection.z;
                crossings.z += delta.z;
            }
            if (t > maxDistance) {
                return false;
            }
        }
    }
    return false;
}

// 1 when a ray from `origin` along `direction`, both in the shadow grid's voxels, reaches the
// shadow distance without entering a solid voxel, otherwise 0.
float sunVisibility(float3 origin, float3 direction) {
    VoxelHit hit;
    return traceVoxels(origin, direction, ubo.lighting.shadowGrid.z, true, hit) ? 0.0 : 1.0;
}

// Folds a unit direction onto the square from -1 to 1, +z at its center [Cigolle et al. 2014].
float2 octahedralEncode(float3 direction) {
    float3 folded = direction / (abs(direction.x) + abs(direction.y) + abs(direction.z));
    float2 encoded = folded.xy;
    if (folded.z < 0.0) {
        encoded = (1.0 - abs(encoded.yx)) * select(encoded >= 0.0, 1.0, -1.0);
    }
    return encoded;
}

// Undoes `octahedralEncode`.
float3 octahedralDecodeSquare(float2 encoded) {
    float3 direction = float3(encoded, 1.0 - abs(encoded.x) - abs(encoded.y));
    if (direction.z < 0.0) {
        direction.xy = (1.0 - abs(direction.yx)) * select(direction.xy >= 0.0, 1.0, -1.0);
    }
    return normalize(direction);
}

// The slot of the probe of probe cell `cell`, false outside the volume.
bool probeSlot(int3 cell, out uint slot) {
    slot = 0;
    int3 counts = int3(ubo.gi.volumeCounts.xyz);
    int3 relative = cell - ubo.gi.volumeMin.xyz;
    if (any(relative < 0) || any(relative >= counts)) {
        return false;
    }
    int3 local = ((cell % counts) + counts) % counts;
    slot = uint(local.x + counts.x * (local.y + counts.y * local.z));
    return true;
}

// Where direction `direction` of slot `slot`'s map of `texels` interior texels is in its atlas.
float2 atlasUV(uint slot, float3 direction, int texels) {
    uint3 counts = ubo.gi.volumeCounts.xyz;
    uint columns = ubo.gi.volumeCounts.w;
    uint rows = (counts.x * counts.y * counts.z + columns - 1) / columns;
    int tile = texels + 2;
    float2 tileMin = float2(slot % columns, slot / columns) * tile;
    float2 texel = tileMin + 1.0 + (octahedralEncode(direction) * 0.5 + 0.5) * texels;
    return texel / float2(columns * tile, rows * tile);
}

// The irradiance the probes around a surface see, for a point `position` in the shadow grid's
// voxels with unit normal `normal`, seen from direction `toViewer`. Only probes updated at least
// `minUpdates` times count. `coverage` falls from 1 to 0 over the volume's last cell, and
// `found` is false when no probe counted.
//
// Blends the 8 probes around the point by trilinear weight, how far each faces the surface
// and a Chebyshev test of whether it can see the point [Majercik et al. 2019], tested a little
// off the surface so the surface doesn't shadow itself [Majercik et al. 2021].
float3 sampleIrradiance(float3 position, float3 normal, float3 toViewer, uint minUpdates, out float coverage, out bool found) {
    float spacing = float(PROBE_SPACING);
    float3 biased = position + (normal * 0.2 + toViewer * 0.8) * (0.75 * spacing * SELF_SHADOW_BIAS);
    // Probes sit at the centers of their cells unless voxels are in the way.
    float3 gridPosition = biased / spacing - 0.5;
    int3 base = int3(floor(gridPosition));
    float3 alpha = gridPosition - float3(base);

    float3 relative = gridPosition - float3(ubo.gi.volumeMin.xyz);
    float3 edgeDistance = min(relative, float3(ubo.gi.volumeCounts.xyz) - 1.0 - relative);
    coverage = saturate(min(edgeDistance.x, min(edgeDistance.y, edgeDistance.z)));

    float3 sum = float3(0.0);
    float totalWeight = 0.0;
    for (int i = 0; i < 8; i++) {
        int3 offset = int3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        int3 cell = base + offset;
        uint slot;
        if (!probeSlot(cell, slot)) {
            continue;
        }
        Probe probe = probes[slot];
        if ((probe.state & PROBE_PLACED) == 0 || probe.updates < minUpdates || any(probe.cell != cell)) {
            continue;
        }

        // Probes behind the surface see less of what it does.
        float3 toProbe = normalize(probe.position - position);
        float facing = (dot(toProbe, normal) + 1.0) * 0.5;
        float weight = facing * facing + 0.2;

        // Probes whose voxels are nearer than the point most likely can't see it.
        float3 probeToPoint = biased - probe.position;
        float distance = length(probeToPoint);
        float2 moments = probeDepth.SampleLevel(atlasUV(slot, probeToPoint / max(distance, 1e-4), DEPTH_TEXELS), 0.0).xy;
        if (distance > moments.x) {
            float variance = abs(moments.x * moments.x - moments.y);
            float beyond = distance - moments.x;
            float chebyshev = variance / (variance + beyond * beyond);
            weight *= max(chebyshev * chebyshev * chebyshev, 0.0);
        }
        // Barely visible probes fade out fast, the eye picks up faint light leaks easily.
        weight = max(weight, 1e-6);
        if (weight < 0.2) {
            weight *= weight * weight / (0.2 * 0.2);
        }
        float3 trilinear = lerp(1.0 - alpha, alpha, float3(offset));
        weight *= trilinear.x * trilinear.y * trilinear.z;

        // Decoded to a gamma 2 curve, which blends closer to how bright it looks.
        float3 irradiance = probeIrradiance.SampleLevel(atlasUV(slot, normal, IRRADIANCE_TEXELS), 0.0).rgb;
        sum += weight * pow(max(irradiance, 0.0), IRRADIANCE_GAMMA * 0.5);
        totalWeight += weight;
    }
    found = totalWeight > 0.0;
    if (!found) {
        return float3(0.0);
    }
    float3 blended = sum / totalWeight;
    return blended * blended;
}

// The light from everywhere but the sun reaching a surface facing `normal` where no probe
// covers it: the sky's and the ground's, blended by how far it faces up.
float3 ambientLight(float3 normal) {
    return lerp(ubo.lighting.groundColor.rgb, ubo.lighting.skyColor.rgb, normal.y * 0.5 + 0.5);
}

// The light reaching a surface at `worldPosition` with unit normal `normal`, seen from
// direction `toViewer`. `faceNormal` is the triangle's own, which rays leave the surface
// along.
//
// For the probes' own rays, `probeRay` is true: only probes that have been updated before
// count, and where none covers the surface inside the volume, there's no indirect light yet.
// The sky's would leak into closed rooms, and with every bounce adding to the next, take
// many frames to fade.
float3 lightAt(float3 worldPosition, float3 normal, float3 faceNormal, float3 toViewer, bool probeRay) {
    LightingUniforms lighting = ubo.lighting;
    float3 gridPosition = worldPosition / max(lighting.shadowGrid.x, 1e-6);
    float3 indirect = ambientLight(normal);
    if (ubo.gi.volumeMin.w != 0) {
        float coverage;
        bool found;
        float3 probesLight = sampleIrradiance(gridPosition, normal, toViewer, probeRay ? 2 : 1, coverage, found);
        if (!found) {
            probesLight = probeRay ? float3(0.0) : indirect;
        }
        indirect = lerp(indirect, probesLight, coverage);
    }
    float facing = dot(normal, lighting.sunDirection.xyz);
    if (facing <= 0.0) {
        return indirect;
    }
    float visible = 1.0;
    if (lighting.shadowTable.w > 0) {
        visible = sunVisibility(gridPosition + faceNormal * SHADOW_RAY_OFFSET, lighting.sunDirection.xyz);
    }
    return indirect + lighting.sunColor.rgb * facing * visible;
}

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

// The most levels of chunks the shadow grid has, each of voxels twice as big as the last.
// Matches `lighting::MAX_SHADOW_LEVELS`.
static const int MAX_SHADOW_LEVELS = 4;
// The most cascades of light probes, each spaced twice as far apart as the last. Matches
// `gi::MAX_CASCADES`.
static const int MAX_CASCADES = 4;

// Matches `renderer::LightingUniforms`.
struct LightingUniforms {
    // Normalized, toward the sun.
    float4 sunDirection;
    float4 sunColor;
    float4 skyColor;
    float4 groundColor;
    // The chunk coordinate of the finest level's shadow table's corner, and chunks per side,
    // 0 without shadows.
    int4 shadowTable;
    // The grid's voxel size, voxels per edge and the shadow distance in voxels, then how many
    // levels of chunks it has.
    float4 shadowGrid;
    // How strong voxel ambient occlusion is, then the debug view, see `renderer::DebugView`.
    float4 options;
    // The chunk coordinate of each level's shadow table's corner, in that level's chunks.
    int4 shadowLevels[MAX_SHADOW_LEVELS];
};

// Matches `gi::GiUniforms`.
struct GiUniforms {
    // Each cascade's probe cell at its low corner, in its own cells, w is 1 when the probes
    // light the frame.
    int4 volumeMins[MAX_CASCADES];
    // Probes along each axis of each cascade, w is probes per row of the atlases.
    uint4 volumeCounts;
    // This frame's random rotation of the probes' rays, a quaternion.
    float4 rayRotation;
    float4 cameraPosition;
    // x is the frame's number, y the rays traced each frame, z how many cascades there are, w
    // the frames between the finest cascade's probes' updates.
    uint4 frame;
    // `ProbeTuning`'s hysteresis, depth hysteresis, depth sharpness and self-shadow bias.
    float4 tuning;
    // `ProbeTuning`'s ray distance, then this frame's multipliers of the hysteresis for light
    // and for distances, below 1 while the probes converge faster after a change.
    float4 rays;
};

// Matches `shadows::ShadowUniforms`.
struct ShadowUniforms {
    // From clip space back to the world.
    float4x4 inverseViewProj;
    // Frame pixels per shadow texel along each axis, 1 when shading traces its own, then the
    // shadow image's width and height.
    uint4 params;
    // The frame's width and height, then 1 when the shadow pass ran this frame, then 1 when it
    // also lit faces with the light from everywhere but the sun.
    uint4 frameSize;
    // How many lights there are, the light tiles across and down the frame, and how many of
    // each pixel's lights are shadowed.
    uint4 lights;
};

// A point or spot light. Matches `lights::GpuLight`.
struct Light {
    // The position, then how far its light reaches, in world units.
    float4 positionRange;
    // Its brightness at its center, then 1 for a spot light.
    float4 colorSpot;
    // The spot's direction, then the cosine of its outer angle.
    float4 directionOuter;
    // The cosine of the spot's inner angle.
    float4 inner;
};

// Matches `uniforms::FrameUniforms`.
struct Uniforms {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
    LightingUniforms lighting;
    GiUniforms gi;
    ShadowUniforms shadow;
};

// A light probe, matches `gi::PROBE_SIZE`. Positions are in the shadow grid's finest voxels,
// cells in its cascade's.
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
// Where a probe's state counts down the frames it converges faster after nearby chunks
// changed, 8 bits [Majercik et al. 2021].
static const uint PROBE_FAST_SHIFT = 8;

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
static const int IRRADIANCE_TEXELS = 8;
static const int DEPTH_TEXELS = 14;
// Irradiance is stored to the power of 1 / 5, which blends a drop in light as fast as a
// rise looks [Majercik et al. 2021].
static const float IRRADIANCE_GAMMA = 5.0;

[[vk::binding(0, 0)]] ConstantBuffer<Uniforms> ubo;
[[vk::binding(1, 0)]] StructuredBuffer<uint> vertexWords;
[[vk::binding(2, 0)]] StructuredBuffer<MeshInfo> meshes;
// Every chunk's voxel types, laid out as `voxel::VoxelTypes` packs them, then its occupancy.
[[vk::binding(3, 0)]] StructuredBuffer<uint> typeWords;
// How voxels of a type look and light, in linear color. Matches `palette::GpuMaterial`.
struct Material {
    // The surface's color, then its roughness.
    float4 colorRoughness;
    // The light it gives off, then its metalness, plus 2 when it's see-through.
    float4 emissionMetalness;
};

// Materials indexed by voxel type.
[[vk::binding(4, 0)]] StructuredBuffer<Material> palette;
// The mesh index of the chunk at each chunk coordinate around the camera, or `NO_CHUNK`.
// `lighting.shadowTable.w` chunks per side, x-fastest.
[[vk::binding(5, 0)]] StructuredBuffer<uint> shadowTable;
// Each probe slot's probe, slots x-fastest by probe cell modulo the volume's size.
#ifdef GI_WRITES
[[vk::binding(6, 0)]] RWStructuredBuffer<Probe> probes;
#else
[[vk::binding(6, 0)]] StructuredBuffer<Probe> probes;
#endif
// The frame's lights, see `Renderer::lights`.
[[vk::binding(17, 0)]] StructuredBuffer<Light> lights;
// A bit for each shadow table entry, set when its chunk was added, removed or remeshed since
// last frame.
[[vk::binding(16, 0)]] StructuredBuffer<uint> changedChunks;
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

// The material of voxel type `id`. Types the palette doesn't cover show matte magenta.
Material materialOf(uint id) {
    uint paletteSize;
    uint stride;
    palette.GetDimensions(paletteSize, stride);
    if (id < paletteSize) {
        return palette[id];
    }
    Material magenta;
    magenta.colorRoughness = float4(1.0, 0.0, 1.0, 1.0);
    magenta.emissionMetalness = float4(0.0);
    return magenta;
}

// The mesh index of level `level`'s chunk at chunk coordinate `chunk`, in that level's
// chunks, `NO_CHUNK` for none, or `outside` set when it's past the level's shadow table.
uint levelChunk(int level, int3 chunk, out bool outside) {
    int side = ubo.lighting.shadowTable.w;
    int3 relative = chunk - ubo.lighting.shadowLevels[level].xyz;
    outside = any(relative < 0) || any(relative >= side);
    if (outside) {
        return NO_CHUNK;
    }
    return shadowTable[((level * side + relative.z) * side + relative.y) * side + relative.x];
}

// The finest level's chunk at `chunk`, see `levelChunk`.
uint shadowChunk(int3 chunk, out bool outside) {
    return levelChunk(0, chunk, outside);
}

// The mesh index of the chunk of the finest level with one holding point `position`, in the
// finest voxels, its level and its chunk coordinate, or `NO_CHUNK`. Either way, the finest
// level whose table holds the point and its chunk there, -1 past every table: no finer chunk
// overlaps it.
uint chunkAtPoint(float3 position, out int level, out int3 chunk, out int coverLevel, out int3 coverChunk) {
    int edge = int(ubo.lighting.shadowGrid.y);
    level = -1;
    chunk = int3(0);
    coverLevel = -1;
    coverChunk = int3(0);
    for (int l = 0; l < int(ubo.lighting.shadowGrid.w); l++) {
        int3 at = int3(floor(position / float(edge << l)));
        bool outside;
        uint meshIndex = levelChunk(l, at, outside);
        if (outside) {
            continue;
        }
        if (coverLevel < 0) {
            coverLevel = l;
            coverChunk = at;
        }
        if (meshIndex != NO_CHUNK) {
            level = l;
            chunk = at;
            return meshIndex;
        }
    }
    return NO_CHUNK;
}

// `value` divided by `divisor`, rounded down.
int3 floorDiv(int3 value, int divisor) {
    return int3(floor(float3(value) / float(divisor)));
}

// Whether any voxel of block `block` of the chunk whose occupancy starts at word `base` isn't
// empty. Matches `voxel::VoxelTypes::occupancy`.
bool blockOccupied(uint base, int blocksPerEdge, int3 block) {
    uint index = uint(block.x + blocksPerEdge * (block.y + blocksPerEdge * block.z));
    return (typeWords[base + index / 32] & (1u << (index % 32))) != 0;
}

// Whether voxel type `id` is see-through, see `Material`.
bool seeThrough(uint id) {
    return materialOf(id).emissionMetalness.w >= 2.0;
}

// How metallic `material` is, from 0 to 1.
float metalnessOf(Material material) {
    float packed = material.emissionMetalness.w;
    return packed >= 2.0 ? packed - 2.0 : packed;
}

// The type of the shadow grid's voxel `voxel`, 0 where there's no chunk.
uint gridVoxel(int3 voxel) {
    int edge = int(ubo.lighting.shadowGrid.y);
    int3 chunk = int3(floor(float3(voxel) / float(edge)));
    bool outside;
    uint meshIndex = shadowChunk(chunk, outside);
    if (meshIndex == NO_CHUNK) {
        return 0;
    }
    return typeAt(meshes[meshIndex].typesBase, uint(edge), uint3(voxel - chunk * edge));
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
    // How much light passes along the ray to it, or to its end without a hit, through the
    // see-through voxels on the way, tinted by them.
    float3 transmittance;
    // The level of the chunk it's in.
    int level;
};

// Whether a ray from `origin` along unit `direction`, both in the shadow grid's finest voxels,
// enters a solid voxel within `maxDistance`, and which. Rays leaving every level's shadow
// table hit nothing.
//
// An Amanatides-Woo DDA, visiting every voxel the ray passes through in order. It runs a
// chunk at a time, through the finest level's chunk wherever there is one, in that chunk's
// own voxels. It crosses a chunk without voxels in one step, and within a chunk crosses each
// block its occupancy says is empty in one step. With `skipStartSolid`, solid voxels the
// ray starts in are the surface it starts on, such as on a curved surface, so they're skipped.
bool traceVoxels(float3 origin, float3 direction, float maxDistance, bool skipStartSolid, out VoxelHit hit) {
    hit.distance = 0.0;
    hit.normal = float3(0.0);
    hit.id = 0;
    hit.transmittance = float3(1.0);
    hit.level = 0;
    // Whether the last voxel was see-through, so each pane tints the light once.
    bool inGlass = false;
    int edge = int(ubo.lighting.shadowGrid.y);
    int blocksPerEdge = (edge + OCCUPANCY_BLOCK - 1) / OCCUPANCY_BLOCK;
    // A ray along an axis would divide by zero.
    direction = select(abs(direction) < 1e-6, select(direction < 0.0, -1e-6, 1e-6), direction);
    float3 inverse = 1.0 / direction;
    int3 stepDirection = select(direction > 0.0, int3(1), int3(-1));

    float t = 0.0;
    // The axis the ray last crossed a boundary on, whose face it entered through.
    int axis = -1;
    bool leftSurface = !skipStartSolid;
    uint voxelSteps = 0;
    for (uint chunkStep = 0; chunkStep < MAX_RAY_CHUNKS; chunkStep++) {
        if (t > maxDistance) {
            return false;
        }
        float3 position = origin + direction * (t + RAY_EPSILON);
        int level;
        int3 chunk;
        int coverLevel;
        int3 coverChunk;
        uint meshIndex = chunkAtPoint(position, level, chunk, coverLevel, coverChunk);
        if (coverLevel < 0) {
            return false;
        }
        // The ray crosses the finest table's chunk here before looking again, no finer chunk
        // is in it.
        float coverSize = float(edge << coverLevel);
        float3 regionMin = float3(coverChunk) * coverSize;
        if (meshIndex == NO_CHUNK) {
            // Nothing to hit, so straight to where the ray leaves it.
            t = exitDistance(origin, direction, inverse, regionMin, regionMin + coverSize, t, axis);
            leftSurface = true;
            inGlass = false;
            continue;
        }

        // In the chunk's own voxels from here, `scale` finest voxels wide.
        float scale = float(1 << level);
        int3 chunkMin = chunk * edge;
        int3 cellsMin = int3(round(regionMin / scale));
        int3 cellsMax = cellsMin + (edge >> (level - coverLevel));
        MeshInfo mesh = meshes[meshIndex];
        int3 cell = int3(floor(position / scale));
        float3 delta = abs(inverse) * scale;
        // Where the ray crosses the next cell boundary on each axis.
        float3 crossings = (float3(cell + select(direction > 0.0, int3(1), int3(0))) * scale - origin) * inverse;
        while (true) {
            if (any(cell < cellsMin) || any(cell >= cellsMax)) {
                break;
            }
            int3 local = cell - chunkMin;
            if (++voxelSteps > MAX_RAY_VOXELS) {
                return false;
            }
            int3 block = local / OCCUPANCY_BLOCK;
            if (!blockOccupied(mesh.occupancyBase, blocksPerEdge, block)) {
                // Nothing to hit, so straight to where the ray leaves the block.
                int3 blockMin = chunkMin + block * OCCUPANCY_BLOCK;
                int3 blockMax = min(blockMin + OCCUPANCY_BLOCK, chunkMin + edge);
                t = exitDistance(origin, direction, inverse, float3(blockMin) * scale, float3(blockMax) * scale, t, axis);
                if (t > maxDistance) {
                    return false;
                }
                leftSurface = true;
                inGlass = false;
                cell = int3(floor((origin + direction * (t + RAY_EPSILON)) / scale));
                crossings = (float3(cell + select(direction > 0.0, int3(1), int3(0))) * scale - origin) * inverse;
                continue;
            }

            uint id = typeAt(mesh.typesBase, uint(edge), uint3(local));
            bool glass = id != 0 && seeThrough(id);
            if (id != 0 && !glass) {
                if (leftSurface) {
                    hit.distance = t;
                    hit.id = id;
                    hit.level = level;
                    if (axis >= 0) {
                        hit.normal[axis] = -float(stepDirection[axis]);
                    }
                    return true;
                }
            } else {
                leftSurface = true;
            }
            // Light passes through see-through voxels, tinted by each pane once.
            if (glass && !inGlass) {
                hit.transmittance *= materialOf(id).colorRoughness.rgb;
                if (all(hit.transmittance < 1.0 / 256.0)) {
                    // As good as opaque.
                    hit.distance = t;
                    hit.id = id;
                    hit.level = level;
                    return true;
                }
            }
            inGlass = glass;

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

// How much of the sun's light gets along a ray from `origin` along `direction`, both in the
// shadow grid's voxels, to the shadow distance: none when a solid voxel is in the way, and
// tinted by any see-through ones.
float3 sunVisibility(float3 origin, float3 direction) {
    VoxelHit hit;
    return traceVoxels(origin, direction, ubo.lighting.shadowGrid.z, true, hit) ? float3(0.0) : hit.transmittance;
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

// Probes in each cascade.
uint probesPerCascade() {
    return ubo.gi.volumeCounts.x * ubo.gi.volumeCounts.y * ubo.gi.volumeCounts.z;
}

// The slot of the probe of cascade `cascade`'s probe cell `cell`, false outside it.
bool probeSlot(int cascade, int3 cell, out uint slot) {
    slot = 0;
    int3 counts = int3(ubo.gi.volumeCounts.xyz);
    int3 relative = cell - ubo.gi.volumeMins[cascade].xyz;
    if (any(relative < 0) || any(relative >= counts)) {
        return false;
    }
    int3 local = ((cell % counts) + counts) % counts;
    slot = uint(cascade) * probesPerCascade() + uint(local.x + counts.x * (local.y + counts.y * local.z));
    return true;
}

// Where direction `direction` of slot `slot`'s map of `texels` interior texels is in its atlas.
float2 atlasUV(uint slot, float3 direction, int texels) {
    uint columns = ubo.gi.volumeCounts.w;
    uint rows = (probesPerCascade() * ubo.gi.frame.z + columns - 1) / columns;
    int tile = texels + 2;
    float2 tileMin = float2(slot % columns, slot / columns) * tile;
    float2 texel = tileMin + 1.0 + (octahedralEncode(direction) * 0.5 + 0.5) * texels;
    return texel / float2(columns * tile, rows * tile);
}

// The irradiance cascade `cascade`'s probes around a surface see, for a point `position` in
// the shadow grid's finest voxels with unit normal `normal`, seen from direction `toViewer`.
// Only probes updated at least `minUpdates` times count. `coverage` falls from 1 to 0 over the
// cascade's last cell, and `found` is false when no probe counted.
//
// Blends the 8 probes around the point by trilinear weight, how far each faces the surface
// and a Chebyshev test of whether it can see the point [Majercik et al. 2019], tested a little
// off the surface so the surface doesn't shadow itself [Majercik et al. 2021]. The trilinear
// weights follow where each probe actually is in its cell, not its cell's center, so probes
// moved off center blend smoothly [Roháček 2022].
float3 sampleIrradiance(int cascade, float3 position, float3 normal, float3 toViewer, uint minUpdates, out float coverage, out bool found) {
    float spacing = float(PROBE_SPACING << cascade);
    float3 biased = position + (normal * 0.2 + toViewer * 0.8) * (0.75 * spacing * ubo.gi.tuning.w);
    // Probes sit at the centers of their cells unless voxels are in the way.
    float3 gridPosition = biased / spacing - 0.5;
    int3 base = int3(floor(gridPosition));
    float3 alpha = gridPosition - float3(base);

    float3 relative = gridPosition - float3(ubo.gi.volumeMins[cascade].xyz);
    float3 edgeDistance = min(relative, float3(ubo.gi.volumeCounts.xyz) - 1.0 - relative);
    coverage = saturate(min(edgeDistance.x, min(edgeDistance.y, edgeDistance.z)));
    found = false;
    if (coverage <= 0.0) {
        return float3(0.0);
    }

    float3 sum = float3(0.0);
    float totalWeight = 0.0;
    for (int i = 0; i < 8; i++) {
        int3 offset = int3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        int3 cell = base + offset;
        uint slot;
        if (!probeSlot(cascade, cell, slot)) {
            continue;
        }
        Probe probe = probes[slot];
        if ((probe.state & PROBE_PLACED) == 0 || probe.updates < minUpdates || any(probe.cell != cell)) {
            continue;
        }

        // Probes behind the surface see less of what it does, and those straight behind it,
        // such as inside a building through its wall, almost nothing. Without the original's
        // floor of 0.2 [Majercik et al. 2019], which lets their darkness through thin walls
        // [Roháček 2022].
        float3 toProbe = normalize(probe.position - position);
        float facing = (dot(toProbe, normal) + 1.0) * 0.5;
        float weight = facing * facing;

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
        // A tent one cell wide around where the probe is, which is the usual trilinear weight
        // for a probe at its cell's center.
        float3 probeOffset = probe.position / spacing - 0.5 - float3(base);
        float3 trilinear = saturate(1.0 - abs(probeOffset - alpha));
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

// The light reaching a surface, straight from the sun and from everywhere else.
struct SurfaceLight {
    float3 direct;
    float3 indirect;
};

// The light from everywhere but the sun reaching a surface at `gridPosition`, in the shadow
// grid's voxels, with unit normal `normal`, seen from direction `toViewer`: the probes', or
// the sky and ground's outside them. See `lightParts` for `probeRay`.
//
// The finest cascade covering the surface lights it, blending into the next over its last
// cell. Where a cascade covers it but none of its probes count, the probes' own rays get no
// light, see `lightParts`.
float3 indirectLight(float3 gridPosition, float3 normal, float3 toViewer, bool probeRay) {
    float3 ambient = ambientLight(normal);
    if (ubo.gi.volumeMins[0].w == 0) {
        return ambient;
    }
    float3 probesLight = float3(0.0);
    // How much is left for coarser cascades, and how much of that none of the probes lit.
    float remaining = 1.0;
    float unlit = 0.0;
    for (int cascade = 0; cascade < int(ubo.gi.frame.z) && remaining > 0.0; cascade++) {
        float coverage;
        bool found;
        float3 light = sampleIrradiance(cascade, gridPosition, normal, toViewer, probeRay ? 2 : 1, coverage, found);
        float share = remaining * coverage;
        if (found) {
            probesLight += share * light;
        } else {
            unlit += share;
        }
        remaining -= share;
    }
    return probesLight + remaining * ambient + (probeRay ? 0.0 : unlit) * ambient;
}

// The light reaching a surface at `worldPosition` with unit normal `normal`, seen from
// direction `toViewer`. `faceNormal` is the triangle's own, which rays leave the surface
// along. `sunVisible` is how much of the sun reaches it, tinted by any see-through voxels in
// the way, when already known, or negative for a shadow ray to find out.
//
// For the probes' own rays, `probeRay` is true: only probes that have been updated before
// count, and where none covers the surface inside the volume, there's no indirect light yet.
// The sky's would leak into closed rooms, and with every bounce adding to the next, take
// many frames to fade.
SurfaceLight lightParts(float3 worldPosition, float3 normal, float3 faceNormal, float3 toViewer, bool probeRay, float3 sunVisible) {
    SurfaceLight light;
    float3 gridPosition = worldPosition / max(ubo.lighting.shadowGrid.x, 1e-6);
    light.indirect = indirectLight(gridPosition, normal, toViewer, probeRay);
    light.direct = sunLight(worldPosition, normal, faceNormal, sunVisible);
    return light;
}

// The sun's light reaching a surface, see `lightParts`.
float3 sunLight(float3 worldPosition, float3 normal, float3 faceNormal, float3 sunVisible) {
    LightingUniforms lighting = ubo.lighting;
    float facing = dot(normal, lighting.sunDirection.xyz);
    if (facing <= 0.0) {
        return float3(0.0);
    }
    float3 visible = sunVisible;
    if (visible.x < 0.0) {
        float3 gridPosition = worldPosition / max(lighting.shadowGrid.x, 1e-6);
        visible = lighting.shadowTable.w > 0
            ? sunVisibility(gridPosition + faceNormal * SHADOW_RAY_OFFSET, lighting.sunDirection.xyz)
            : float3(1.0);
    }
    return lighting.sunColor.rgb * facing * visible;
}

// Moves a point on a face into the voxel behind it, in voxels.
static const float FACE_NUDGE = 1.0 / 256.0;

// Whether level `level`'s voxel `voxel`, in that level's voxels, blocks light from reaching a
// corner: solid and not see-through.
bool gridOccludes(int level, int3 voxel) {
    int edge = int(ubo.lighting.shadowGrid.y);
    int3 chunk = floorDiv(voxel, edge);
    bool outside;
    uint meshIndex = levelChunk(level, chunk, outside);
    if (meshIndex == NO_CHUNK) {
        return false;
    }
    MeshInfo mesh = meshes[meshIndex];
    int3 local = voxel - chunk * edge;
    int blocksPerEdge = (edge + OCCUPANCY_BLOCK - 1) / OCCUPANCY_BLOCK;
    if (!blockOccupied(mesh.occupancyBase, blocksPerEdge, local / OCCUPANCY_BLOCK)) {
        return false;
    }
    uint id = typeAt(mesh.typesBase, uint(edge), uint3(local));
    return id != 0 && !seeThrough(id);
}

// How open the face of a level `level` chunk facing unit axis `normal` is at `position` on
// it, in that level's voxels, as `voxelOcclusion` in `mesh.slang` finds it.
float gridOpenness(int level, float3 position, int3 normal) {
    int3 front = int3(floor(position - float3(normal) * FACE_NUDGE)) + normal;
    int3 u = normal.x != 0 ? int3(0, 1, 0) : int3(1, 0, 0);
    int3 v = normal.z != 0 ? int3(0, 1, 0) : int3(0, 0, 1);
    float2 along = float2(frac(dot(position, float3(u))), frac(dot(position, float3(v))));
    bool around[9];
    for (int i = 0; i < 9; i++) {
        int2 at = int2(i % 3, i / 3) - 1;
        around[i] = i != 4 && gridOccludes(level, front + u * at.x + v * at.y);
    }
    return cornerOpenness(around, along);
}

// How open a face is at `along` across its voxel, from 0 to 1 on each axis, with `around`
// the 8 voxels around the one in front of it, 3 to a row, whether each blocks light: each
// corner darkens with those around it, two solid sides closing it entirely, and the point
// blends the four corners [classic voxel ambient occlusion].
float cornerOpenness(bool around[9], float2 along) {
    float corners[4];
    for (int i = 0; i < 4; i++) {
        int2 side = int2(i & 1, i >> 1) * 2 - 1;
        bool sideU = around[4 + side.x];
        bool sideV = around[4 + 3 * side.y];
        bool corner = around[4 + side.x + 3 * side.y];
        corners[i] = sideU && sideV ? 0.0 : (3.0 - float(sideU) - float(sideV) - float(corner)) / 3.0;
    }
    return lerp(lerp(corners[0], corners[1], along.x), lerp(corners[2], corners[3], along.x), along.y);
}

// Pixels along each edge of a tile lights are culled for. Matches `shadows::LIGHT_TILE`.
static const uint LIGHT_TILE = 16;
// The most lights a tile keeps, after its count, in `shadow.slang`'s tile lists, the
// brightest.
static const uint TILE_LIGHTS = 63;
// The most lights a frame has. Matches `lights::MAX_LIGHTS`.
static const uint MAX_LIGHTS = 255;

// The light `light` gives a surface at `worldPosition` facing `normal`, before shadows: its
// color, fading with the square of the distance to nothing at its range, within its cone.
// Also where it is, from the surface, in world units.
float3 lightReaching(Light light, float3 worldPosition, float3 normal, out float3 toLight, out float distance) {
    float3 offset = light.positionRange.xyz - worldPosition;
    distance = length(offset);
    toLight = offset / max(distance, 1e-6);
    float range = light.positionRange.w;
    float x = distance / range;
    float window = saturate(1.0 - x * x * x * x);
    float falloff = window * window / (distance * distance + 1.0);
    float cone = 1.0;
    if (light.colorSpot.w > 0.0) {
        float along = dot(-toLight, light.directionOuter.xyz);
        cone = smoothstep(light.directionOuter.w, light.inner.x, along);
    }
    return light.colorSpot.rgb * falloff * cone * max(dot(normal, toLight), 0.0);
}

// How bright `light` is at `worldPosition`, ignoring which way the surface faces, to choose
// which lights to shadow.
float lightStrength(Light light, float3 worldPosition) {
    float3 toLight;
    float distance;
    float3 reaching = lightReaching(light, worldPosition, -normalize(worldPosition - light.positionRange.xyz), toLight, distance);
    return dot(reaching, float3(0.3, 0.6, 0.1));
}

// How much of `light`'s light gets to `origin`, in the shadow grid's voxels: 0 behind solid
// voxels, dimmed by see-through ones. Traced from the light, past the solid voxels it starts
// in, so a lamp set in a glowing block or a fixture isn't shadowed by it, to half a voxel
// short of the surface, which a curved one could otherwise shadow itself in.
float lightShadow(Light light, float3 origin) {
    if (ubo.lighting.shadowTable.w == 0) {
        return 1.0;
    }
    float3 position = light.positionRange.xyz / ubo.lighting.shadowGrid.x;
    float3 offset = origin - position;
    float distance = length(offset);
    if (distance <= 0.5) {
        return 1.0;
    }
    VoxelHit hit;
    if (traceVoxels(position, offset / distance, distance - 0.5, true, hit)) {
        return 0.0;
    }
    return dot(hit.transmittance, float3(1.0 / 3.0));
}

// The light the frame's lights give a ray's hit at `worldPosition`, facing `normal`: each
// light's, the brightest `shadowed` shadowed, at most 4.
float3 lightsAtHit(float3 worldPosition, float3 normal, uint shadowed) {
    uint count = ubo.shadow.lights.x;
    float3 origin = worldPosition / ubo.lighting.shadowGrid.x + normal * SHADOW_RAY_OFFSET;
    // The brightest, by light index.
    uint brightest[4] = { 0xff, 0xff, 0xff, 0xff };
    float strengths[4] = { 0.0, 0.0, 0.0, 0.0 };
    float3 total = float3(0.0);
    for (uint i = 0; i < count; i++) {
        float3 toLight;
        float distance;
        float3 reaching = lightReaching(lights[i], worldPosition, normal, toLight, distance);
        float strength = dot(reaching, float3(0.3, 0.6, 0.1));
        if (strength <= 0.0) {
            continue;
        }
        total += reaching;
        // Kept in order, brightest first.
        for (uint k = 0; k < shadowed; k++) {
            if (strength > strengths[k]) {
                for (uint j = shadowed - 1; j > k; j--) {
                    strengths[j] = strengths[j - 1];
                    brightest[j] = brightest[j - 1];
                }
                strengths[k] = strength;
                brightest[k] = i;
                break;
            }
        }
    }
    // The brightest are shadowed, by taking away what doesn't get through.
    for (uint k = 0; k < shadowed; k++) {
        if (brightest[k] == 0xff) {
            break;
        }
        Light light = lights[brightest[k]];
        float3 toLight;
        float distance;
        float3 reaching = lightReaching(light, worldPosition, normal, toLight, distance);
        total -= reaching * (1.0 - lightShadow(light, origin));
    }
    return max(total, 0.0);
}

// Faces narrower than this reflect what's around them sharply, traced by the shadow pass,
// blending into the probes' blurry light toward it.
static const float GLOSSY_ROUGHNESS = 0.5;

// The number of the face whose outward normal is the unit axis `normal`, from 1 to 6.
uint faceNumber(float3 normal) {
    float3 size = abs(normal);
    uint axis = size.x > size.y && size.x > size.z ? 0 : (size.y > size.z ? 1 : 2);
    return 1 + axis * 2 + (normal[axis] > 0.0 ? 1 : 0);
}

// The light coming back along a ray from `origin` along unit `direction`, both in the shadow
// grid's voxels, within `maxDistance`: what it hits, lit as the camera would see it, or the
// sky, through any see-through voxels on the way.
float3 radianceAlong(float3 origin, float3 direction, float maxDistance) {
    VoxelHit hit;
    if (!traceVoxels(origin, direction, maxDistance, false, hit)) {
        return ambientLight(direction) * hit.transmittance;
    }
    if (hit.distance <= 0.0) {
        return float3(0.0);
    }
    float3 worldPosition = (origin + direction * hit.distance) * ubo.lighting.shadowGrid.x;
    float3 light = lightAt(worldPosition, hit.normal, hit.normal, -direction, false, float3(-1.0))
        + lightsAtHit(worldPosition, hit.normal, min(ubo.shadow.lights.w, 1));
    Material material = materialOf(hit.id);
    return (material.colorRoughness.rgb * light + material.emissionMetalness.rgb) * hit.transmittance;
}

// How much of a light from unit direction `toLight` a surface with unit normal `normal` and
// `roughness` reflects toward `toViewer`, as a highlight, before Fresnel: a normalized
// Blinn-Phong lobe, sharper the smoother it is.
float highlight(float3 normal, float3 toViewer, float3 toLight, float roughness) {
    float alpha = max(roughness * roughness, 0.002);
    float exponent = 2.0 / (alpha * alpha) - 2.0;
    float3 halfway = normalize(toLight + toViewer);
    return (exponent + 8.0) / (8.0 * 3.14159265) * pow(saturate(dot(normal, halfway)), exponent);
}

// The color a surface of `material` with unit normal `normal` shows toward `toViewer`, lit by
// `lit` from everywhere, and reflecting `reflection`, the light from the way the viewer's
// reflected. With roughness 1 and no metalness, it's `lit` times its color. Smoother, more of
// the light is reflected, more at grazing angles [Schlick 1994]. Metals reflect it all, in
// their own color.
float3 shadeSurface(Material material, float3 normal, float3 toViewer, float3 lit, float3 reflection) {
    float3 base = material.colorRoughness.rgb;
    float roughness = material.colorRoughness.w;
    float metalness = metalnessOf(material);
    float3 reflectance = lerp(float3(0.04), base, metalness);
    float grazing = pow(1.0 - saturate(dot(normal, toViewer)), 5.0);
    float3 fresnel = reflectance + (max(float3(1.0 - roughness), reflectance) - reflectance) * grazing;
    float3 reflected = fresnel * (1.0 - roughness);
    float3 dielectric = base * (1.0 - reflected) * lit + reflected * reflection;
    float3 metal = fresnel * reflection;
    return lerp(dielectric, metal, metalness) + material.emissionMetalness.rgb;
}

// All the light `lightParts` finds reaching a surface.
float3 lightAt(float3 worldPosition, float3 normal, float3 faceNormal, float3 toViewer, bool probeRay, float3 sunVisible) {
    SurfaceLight light = lightParts(worldPosition, normal, faceNormal, toViewer, probeRay, sunVisible);
    return light.direct + light.indirect;
}

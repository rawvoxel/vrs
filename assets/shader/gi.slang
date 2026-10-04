// Updates the light probes each frame, after Majercik et al. 2019 and 2021.
//
// 1. `classifyProbes` moves each probe to where its cell's chunk says it goes, baked when the
//    chunk was added, and lists those near voxels, the only ones shading anything. Every
//    cascade's probes are classified together, and share the frame's rays.
// 2. `planRays` shares the frame's rays between the listed probes, so the cost stays steady.
// 3. `traceProbeRays` traces the probes' rays through the voxels and shades their hits, with
//    the probes' own light from last frame, which adds a bounce of light every frame.
// 4. `updateIrradiance` and `updateDepth` blend the rays into each probe's maps.

#define GI_WRITES
#include "common.slang"

// Matches `gi::ACTIVE_LIST`: tracing's group counts, the updates' group counts, how many
// probes are listed and how many rays each traces come first.
static const uint ACTIVE_LIST = 8;
static const uint ACTIVE_COUNT = 6;
static const uint ACTIVE_RAYS = 7;
// The fewest and most rays a probe traces in a frame. Match `gi::MIN_RAYS_PER_PROBE`.
static const uint MIN_RAYS_PER_PROBE = 32;
static const uint MAX_RAYS_PER_PROBE = 256;
// Rays traced by each group of `traceProbeRays`, the fewest a probe traces.
static const uint TRACE_GROUP = MIN_RAYS_PER_PROBE;
// Matches `gi::CELL_PLACED`.
static const uint CELL_PLACED = 1u << 31;
// The probe's cage's diagonal, plus as far as a probe can be off its cell's center, for the
// finest cascade. Rays going further are left out of the distance maps, so a texel's rays
// passing the edge of a wall don't average its distance out past the wall and leak light
// through it [Roháček 2022].
static const float MAX_PROBE_DEPTH = float(PROBE_SPACING) * 1.7320508 * 1.5;

[[vk::binding(9, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> irradianceAtlas;
[[vk::binding(10, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> depthAtlas;
// Each listed probe's rays: their radiance, then how far they went.
[[vk::binding(11, 0)]] RWStructuredBuffer<float4> rayResults;
// The dispatches' group counts and the frame's ray share, then the listed probes' slots.
[[vk::binding(12, 0)]] RWStructuredBuffer<uint> active;

// The cascade of the probe in slot `slot`.
int cascadeOf(uint slot) {
    return int(slot / probesPerCascade());
}

// Frames between the updates of the probe in slot `slot`, which converges faster for `fast`
// more frames: `ProbeTuning::update_interval` in the finest cascade, twice as many in each
// coarser one, further off, or every frame while it converges faster.
uint updatePeriod(uint slot, uint fast) {
    return fast > 0 ? 1 : ubo.gi.frame.w << uint(cascadeOf(slot));
}

// Where cascade `cascade`'s probe cell `cell` puts its probe, in the finest voxels, false when
// it has no room for one. From the chunk of the finest level holding the cell's center, see
// `chunkAtPoint`: one of the cascade's level has its cells baked when it was added, see
// `gi::bake_probe_cells`, a finer one has the baked cell holding the center. A coarser one's
// voxels are bigger than the cell, its probe is at its center when that's empty. A cell
// without a chunk is empty, with its probe at its center.
bool probePlacement(int cascade, int3 cell, out float3 position) {
    int spacing = PROBE_SPACING << cascade;
    float3 center = (float3(cell) + 0.5) * float(spacing);
    position = center;
    int level;
    int3 chunk;
    int coverLevel;
    int3 coverChunk;
    uint meshIndex = chunkAtPoint(center, level, chunk, coverLevel, coverChunk);
    if (meshIndex == NO_CHUNK) {
        return true;
    }
    int edge = int(ubo.lighting.shadowGrid.y);
    float scale = float(1 << level);
    if (level > cascade) {
        // The center is a corner of 8 of its voxels. The probe stays there when they're all
        // empty, else goes to the center of an empty one, so its rays don't start in voxels.
        int3 low = int3(round(center / scale)) - chunk * edge - 1;
        int empty = -1;
        bool allEmpty = true;
        for (int i = 0; i < 8; i++) {
            int3 voxel = low + int3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
            bool inChunk = all(voxel >= 0) && all(voxel < edge);
            if (inChunk && typeAt(meshes[meshIndex].typesBase, uint(edge), uint3(voxel)) != 0) {
                allEmpty = false;
            } else if (empty < 0) {
                empty = i;
            }
        }
        if (!allEmpty && empty >= 0) {
            int3 voxel = low + int3(empty & 1, (empty >> 1) & 1, (empty >> 2) & 1);
            position = (float3(chunk * edge + voxel) + 0.5) * scale;
        }
        return empty >= 0;
    }
    int cells = edge / PROBE_SPACING;
    int3 local = int3(floor(center / scale)) - chunk * edge;
    int3 bakedCell = local / PROBE_SPACING;
    uint word = typeWords[meshes[meshIndex].probesBase + uint(bakedCell.x + cells * (bakedCell.y + cells * bakedCell.z))];
    if ((word & CELL_PLACED) == 0) {
        return false;
    }
    float3 inCell = float3(word & 63, (word >> 6) & 63, (word >> 12) & 63) * 0.5;
    position = (float3(chunk * edge + bakedCell * PROBE_SPACING) + inCell) * scale;
    return true;
}

// Whether any of cascade `cascade`'s probe cells from `low` to `high` has voxels. From the
// masks baked after the probe words of the chunks they're in, each chunk looked up once, at
// the cascade's level and coarser. Chunks of finer levels only have chunks with voxels.
bool cellsHaveVoxels(int cascade, int3 low, int3 high) {
    int edge = int(ubo.lighting.shadowGrid.y);
    int cells = edge / PROBE_SPACING;
    int spacing = PROBE_SPACING << cascade;
    // In the finest voxels.
    int3 regionMin = low * spacing;
    int3 regionMax = (high + 1) * spacing - 1;
    for (int level = 0; level < int(ubo.lighting.shadowGrid.w); level++) {
        int cellSize = PROBE_SPACING << level;
        int3 cellLow = floorDiv(regionMin, cellSize);
        int3 cellHigh = floorDiv(regionMax, cellSize);
        int3 chunkLow = floorDiv(cellLow, cells);
        int3 chunkHigh = floorDiv(cellHigh, cells);
        for (int z = chunkLow.z; z <= chunkHigh.z; z++) {
            for (int y = chunkLow.y; y <= chunkHigh.y; y++) {
                for (int x = chunkLow.x; x <= chunkHigh.x; x++) {
                    int3 chunk = int3(x, y, z);
                    bool outside;
                    uint meshIndex = levelChunk(level, chunk, outside);
                    if (meshIndex == NO_CHUNK) {
                        continue;
                    }
                    if (level < cascade) {
                        return true;
                    }
                    uint mask = meshes[meshIndex].probesBase + uint(cells * cells * cells);
                    int3 from = max(cellLow - chunk * cells, 0);
                    int3 to = min(cellHigh - chunk * cells, cells - 1);
                    for (int cz = from.z; cz <= to.z; cz++) {
                        for (int cy = from.y; cy <= to.y; cy++) {
                            for (int cx = from.x; cx <= to.x; cx++) {
                                uint index = uint(cx + cells * (cy + cells * cz));
                                if ((typeWords[mask + index / 32] & (1u << (index % 32))) != 0) {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return false;
}

// Frames a probe converges faster after nearby chunks changed, for its light, and of those,
// for its distances [Majercik et al. 2021].
static const uint FAST_FRAMES = 10;
static const uint FAST_DEPTH_FRAMES = 7;

// Whether a chunk of any level holding any of cascade `cascade`'s probe cells from `low` to
// `high` changed since last frame.
bool chunksChanged(int cascade, int3 low, int3 high) {
    int edge = int(ubo.lighting.shadowGrid.y);
    int side = ubo.lighting.shadowTable.w;
    int spacing = PROBE_SPACING << cascade;
    int3 regionMin = low * spacing;
    int3 regionMax = (high + 1) * spacing - 1;
    for (int level = 0; level < int(ubo.lighting.shadowGrid.w); level++) {
        int3 corner = ubo.lighting.shadowLevels[level].xyz;
        int3 chunkLow = floorDiv(regionMin, edge << level) - corner;
        int3 chunkHigh = floorDiv(regionMax, edge << level) - corner;
        for (int z = max(chunkLow.z, 0); z <= min(chunkHigh.z, side - 1); z++) {
            for (int y = max(chunkLow.y, 0); y <= min(chunkHigh.y, side - 1); y++) {
                for (int x = max(chunkLow.x, 0); x <= min(chunkHigh.x, side - 1); x++) {
                    uint index = uint(((level * side + z) * side + y) * side + x);
                    if ((changedChunks[index / 32] & (1u << (index % 32))) != 0) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void classifyProbes(uint3 threadId: SV_DispatchThreadID) {
    int3 counts = int3(ubo.gi.volumeCounts.xyz);
    uint slot = threadId.x;
    if (slot >= probesPerCascade() * ubo.gi.frame.z) {
        return;
    }
    // The cell in its cascade whose probe this slot holds.
    int cascade = cascadeOf(slot);
    uint inCascade = slot % probesPerCascade();
    int3 local = int3(inCascade % counts.x, (inCascade / counts.x) % counts.y, inCascade / (counts.x * counts.y));
    int3 volumeMin = ubo.gi.volumeMins[cascade].xyz;
    int3 cell = volumeMin + (((local - volumeMin) % counts) + counts) % counts;

    float3 position;
    bool placed = probePlacement(cascade, cell, position);
    Probe probe = probes[slot];
    bool wasPlaced = (probe.state & PROBE_PLACED) != 0;
    if (any(probe.cell != cell) || placed != wasPlaced || (placed && any(position != probe.position))) {
        // New to the volume, or its chunk changed: what it saw before was from somewhere else.
        probe.cell = cell;
        probe.state = placed ? PROBE_PLACED : 0;
        probe.position = position;
        probe.updates = 0;
        probes[slot] = probe;
    }
    if (!placed) {
        return;
    }

    // Probes next to a chunk added, removed or remeshed converge faster for a while.
    uint fast = (probe.state >> PROBE_FAST_SHIFT) & 0xff;
    fast = chunksChanged(cascade, cell - 1, cell + 1) ? FAST_FRAMES : (fast > 0 ? fast - 1 : 0);
    uint state = (probe.state & 0xff) | (fast << PROBE_FAST_SHIFT);
    if (state != probe.state) {
        probe.state = state;
        probes[slot].state = state;
    }

    // Each probe updates every few frames, in turn, unless nearby chunks just changed.
    uint period = updatePeriod(slot, fast);
    if (((ubo.gi.frame.x + slot) & (period - 1)) != 0) {
        return;
    }

    // Only probes with voxels in or next to their cell shade any surface.
    if (!cellsHaveVoxels(cascade, cell - 1, cell + 1)) {
        return;
    }
    probes[slot].updates = probe.updates + 1;
    uint index;
    InterlockedAdd(active[ACTIVE_COUNT], 1, index);
    active[ACTIVE_LIST + index] = slot;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void planRays() {
    // The most rays, a power of two, each listed probe can trace within the frame's budget.
    uint count = active[ACTIVE_COUNT];
    uint budget = ubo.gi.frame.y;
    uint rays = MIN_RAYS_PER_PROBE;
    while (rays < MAX_RAYS_PER_PROBE && rays * 2 * count <= budget) {
        rays *= 2;
    }
    active[ACTIVE_RAYS] = rays;
    active[0] = count * (rays / TRACE_GROUP);
    active[3] = count;
}

// Ray `index` of `count` of every probe this frame: evenly spread over the sphere
// [Marques et al. 2013], rotated randomly each frame so probes see every direction over time.
float3 probeRayDirection(uint index, uint count) {
    float goldenRatio = (sqrt(5.0) + 1.0) * 0.5;
    float phi = 2.0 * 3.14159265 * frac(float(index) * (goldenRatio - 1.0));
    float cosTheta = 1.0 - (2.0 * float(index) + 1.0) / float(count);
    float sinTheta = sqrt(saturate(1.0 - cosTheta * cosTheta));
    float3 direction = float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
    float4 q = ubo.gi.rayRotation;
    return direction + 2.0 * cross(q.xyz, cross(q.xyz, direction) + q.w * direction);
}

[shader("compute")]
[numthreads(TRACE_GROUP, 1, 1)]
void traceProbeRays(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID) {
    uint rays = active[ACTIVE_RAYS];
    uint groupsPerProbe = rays / TRACE_GROUP;
    uint listed = group.x / groupsPerProbe;
    uint ray = (group.x % groupsPerProbe) * TRACE_GROUP + thread.x;
    uint slot = active[ACTIVE_LIST + listed];
    Probe probe = probes[slot];
    float3 direction = probeRayDirection(ray, rays);
    float4 result;
    VoxelHit hit;
    // Coarser cascades' rays reach further, in proportion.
    float reach = ubo.gi.rays.x * float(1 << cascadeOf(slot));
    if (!traceVoxels(probe.position, direction, reach, false, hit)) {
        // The sky, or the ground past the world, through any see-through voxels.
        result = float4(ambientLight(direction) * hit.transmittance, 1e9);
    } else if (hit.distance <= 0.0) {
        // Voxels fill the probe's spot, though its chunk says otherwise: dark and close, so
        // it isn't trusted [Majercik et al. 2021].
        result = float4(0.0);
    } else {
        // Lit as the camera would see it, with last frame's light from the probes.
        float3 worldPosition = (probe.position + direction * hit.distance) * ubo.lighting.shadowGrid.x;
        float3 light = lightAt(worldPosition, hit.normal, hit.normal, -direction, true, float3(-1.0))
            + lightsAtHit(worldPosition, hit.normal, ubo.shadow.lights.w);
        Material material = materialOf(hit.id);
        // Glowing voxels light what's around them through the probes, seen through any
        // see-through voxels on the way.
        float3 radiance = material.colorRoughness.rgb * light + material.emissionMetalness.rgb;
        result = float4(radiance * hit.transmittance, hit.distance);
    }
    rayResults[listed * rays + ray] = result;
}

groupshared float4 sharedResults[MAX_RAYS_PER_PROBE];
groupshared float3 sharedDirections[MAX_RAYS_PER_PROBE];

// Loads listed probe `listed`'s rays for the whole group of `groupSize` threads, and returns
// how many there are.
uint loadRays(uint listed, uint threadIndex, uint groupSize) {
    uint rays = active[ACTIVE_RAYS];
    for (uint i = threadIndex; i < rays; i += groupSize) {
        sharedResults[i] = rayResults[listed * rays + i];
        sharedDirections[i] = probeRayDirection(i, rays);
    }
    GroupMemoryBarrierWithGroupSync();
    return rays;
}

// The direction of texel `texel` of a map of `texels` interior texels with a one texel border.
// Border texels take the direction of the interior texel they continue across the octahedron's
// fold, so sampling between them blends seamlessly.
float3 texelDirection(int2 texel, int texels) {
    int last = texels + 1;
    bool left = texel.x == 0;
    bool right = texel.x == last;
    bool top = texel.y == 0;
    bool bottom = texel.y == last;
    int2 interior = texel - 1;
    if ((left || right) && (top || bottom)) {
        interior = int2(left ? texels - 1 : 0, top ? texels - 1 : 0);
    } else if (left || right) {
        interior = int2(left ? 0 : texels - 1, texels - texel.y);
    } else if (top || bottom) {
        interior = int2(texels - texel.x, top ? 0 : texels - 1);
    }
    float2 encoded = (float2(interior) + 0.5) / float(texels) * 2.0 - 1.0;
    return octahedralDecodeSquare(encoded);
}

// Where texel `texel` of listed probe `listed`'s map of `texels` interior texels is in its atlas.
int2 atlasTexel(uint listed, int2 texel, int texels) {
    uint slot = active[ACTIVE_LIST + listed];
    uint columns = ubo.gi.volumeCounts.w;
    return int2(slot % columns, slot / columns) * (texels + 2) + texel;
}

[shader("compute")]
[numthreads(IRRADIANCE_TEXELS + 2, IRRADIANCE_TEXELS + 2, 1)]
void updateIrradiance(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID, uint threadIndex: SV_GroupIndex) {
    uint rays = loadRays(group.x, threadIndex, (IRRADIANCE_TEXELS + 2) * (IRRADIANCE_TEXELS + 2));
    uint slot = active[ACTIVE_LIST + group.x];
    Probe probe = probes[slot];

    // The average light of the rays, weighted by how far each faces the texel's direction.
    float3 direction = texelDirection(int2(thread.xy), IRRADIANCE_TEXELS);
    float3 sum = float3(0.0);
    float totalWeight = 0.0;
    for (uint i = 0; i < rays; i++) {
        float weight = max(dot(direction, sharedDirections[i]), 0.0);
        sum += weight * sharedResults[i].rgb;
        totalWeight += weight;
    }
    float3 irradiance = totalWeight > 1e-4 ? sum / totalWeight : float3(0.0);
    irradiance = pow(max(irradiance, 0.0), 1.0 / IRRADIANCE_GAMMA);

    int2 texel = atlasTexel(group.x, int2(thread.xy), IRRADIANCE_TEXELS);
    float3 old = irradianceAtlas[texel].rgb;
    // A probe's first update replaces what was there, and big changes are let through
    // faster [Majercik et al. 2021].
    // Probes updated every few frames keep as much less each time, so they follow changes as
    // fast.
    uint fast = (probe.state >> PROBE_FAST_SHIFT) & 0xff;
    float hysteresis = probe.updates <= 1
        ? 0.0
        : pow(ubo.gi.tuning.x * ubo.gi.rays.y * (fast > 0 ? 0.5 : 1.0), float(updatePeriod(slot, fast)));
    float3 change = abs(irradiance - old);
    float largestChange = max(change.x, max(change.y, change.z));
    if (largestChange > 0.25) {
        hysteresis = max(hysteresis - 0.15, 0.0);
    }
    if (largestChange > 0.8) {
        hysteresis = 0.0;
    }
    irradianceAtlas[texel] = float4(lerp(irradiance, old, hysteresis), 1.0);
}

[shader("compute")]
[numthreads(DEPTH_TEXELS + 2, DEPTH_TEXELS + 2, 1)]
void updateDepth(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID, uint threadIndex: SV_GroupIndex) {
    uint rays = loadRays(group.x, threadIndex, (DEPTH_TEXELS + 2) * (DEPTH_TEXELS + 2));
    uint slot = active[ACTIVE_LIST + group.x];
    Probe probe = probes[slot];
    float maxDepth = MAX_PROBE_DEPTH * float(1 << cascadeOf(slot));

    // The mean distance and squared distance of the rays near the texel's direction that hit
    // voxels within the cage, or far when none did.
    float3 direction = texelDirection(int2(thread.xy), DEPTH_TEXELS);
    float2 sum = float2(0.0);
    float nearWeight = 0.0;
    float totalWeight = 0.0;
    for (uint i = 0; i < rays; i++) {
        // Sharpened, so the distances' edges follow the voxels [Majercik et al. 2019].
        float weight = pow(max(dot(direction, sharedDirections[i]), 0.0), ubo.gi.tuning.z);
        float distance = sharedResults[i].w;
        totalWeight += weight;
        if (distance <= maxDepth) {
            sum += weight * float2(distance, distance * distance);
            nearWeight += weight;
        }
    }
    if (totalWeight <= 1e-4) {
        return;
    }
    float2 moments = nearWeight > 1e-4
        ? sum / nearWeight
        : float2(maxDepth, maxDepth * maxDepth);

    int2 texel = atlasTexel(group.x, int2(thread.xy), DEPTH_TEXELS);
    float2 old = depthAtlas[texel].xy;
    uint fast = (probe.state >> PROBE_FAST_SHIFT) & 0xff;
    float hysteresis = probe.updates <= 1
        ? 0.0
        : pow(ubo.gi.tuning.y * ubo.gi.rays.z * (fast > FAST_FRAMES - FAST_DEPTH_FRAMES ? 0.5 : 1.0), float(updatePeriod(slot, fast)));
    depthAtlas[texel] = float4(lerp(moments, old, hysteresis), 0.0, 0.0);
}

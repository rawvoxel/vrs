// Updates the light probes each frame, after Majercik et al. 2019 and 2021.
//
// 1. `classifyProbes` moves each probe to where its cell's chunk says it goes, baked when the
//    chunk was added, and lists those near voxels, the only ones shading anything. Every
//    cascade's probes are classified together, and share the frame's rays.
// 2. `planRays` shares the frame's rays between the listed probes, so the cost stays steady.
// 3. `traceProbeRays` traces the probes' rays through the voxels and shades their hits, with
//    the probes' own light from last frame, which adds a bounce of light every frame.
// 4. `updateIrradiance` and `updateDistances` blend the rays into each probe's maps.

#define PROBE_WRITES
#include "common.slang"

// Matches `renderer::lighting::probes::volume::ACTIVE_LIST`: tracing's group counts, the updates'
// group counts, how many probes are listed and how many rays each traces come first.
static const uint ACTIVE_LIST = 8;
static const uint ACTIVE_TRACE_GROUPS = 0;
static const uint ACTIVE_UPDATE_GROUPS = 3;
static const uint ACTIVE_COUNT = 6;
static const uint ACTIVE_RAYS = 7;
// The fewest rays a probe traces in a frame, which matches
// `renderer::lighting::probes::volume::MIN_RAYS_PER_PROBE`, and the most.
static const uint MIN_RAYS_PER_PROBE = 32;
static const uint MAX_RAYS_PER_PROBE = 256;
// Rays traced by each group of `traceProbeRays`, the fewest a probe traces.
static const uint TRACE_GROUP = MIN_RAYS_PER_PROBE;
// Matches `renderer::lighting::probes::cells::CELL_PLACED`.
static const uint CELL_PLACED = 1u << 31;
// The probe's cage's diagonal, plus as far as a probe can be off its cell's center, for the
// finest cascade. Rays going further are left out of the distance maps, so a texel's rays
// passing the edge of a wall don't average its distance out past the wall and leak light
// through it [Roháček 2022].
static const float MAX_PROBE_DISTANCE = float(PROBE_CELL) * 1.7320508 * 1.5;

[[vk::binding(9, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> probeIrradianceStorage;
[[vk::binding(10, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> probeDistancesStorage;
// Each listed probe's rays: their radiance, then how far they went.
[[vk::binding(11, 0)]] RWStructuredBuffer<float4> probeRays;
// The dispatches' group counts and the frame's ray share, then the listed probes' slots.
[[vk::binding(12, 0)]] RWStructuredBuffer<uint> activeProbes;

// The frames a probe whose state is `state` has left converging faster.
uint fastFrames(uint state) {
    return (state >> PROBE_FAST_SHIFT) & 0xff;
}

// Frames between the updates of the probe in slot `slot`, which converges faster for `fast`
// more frames: `ProbeTuning::update_interval` in the finest cascade, twice as many in each
// coarser one, further off, or every frame while it converges faster.
uint updatePeriod(uint slot, uint fast) {
    return fast > 0 ? 1 : uniforms.probeVolume.updateInterval << uint(cascadeOf(slot));
}

// Whether the `size` wide cube of voxels from `low` of the chunk whose types start at
// `typesBase`, `edge` voxels wide, is empty.
bool voxelsEmpty(uint typesBase, int edge, int3 low, int size) {
    for (int i = 0; i < size * size * size; i++) {
        int3 voxel = low + int3(i % size, (i / size) % size, i / (size * size));
        if (typeAt(typesBase, uint(edge), uint3(voxel)) != 0) {
            return false;
        }
    }
    return true;
}

// Where a probe cell `size` voxels wide from voxel `low` of mesh `mesh`, a chunk `edge` voxels
// wide, puts its probe, in that chunk's half voxels from `low`, false when it's solid: as
// `bake_probe_cells` does, the center of the largest empty part of it nearest its center,
// breadth first. `size` is at most `OCCUPANCY_BLOCK`, so the cell is inside one block, and
// most cells, with none of its voxels, are settled by it.
bool placeInVoxels(MeshInfo mesh, int edge, int3 low, int size, out int3 halfVoxel) {
    halfVoxel = int3(size);
    if (!blockOccupied(mesh.occupancyBase, occupancyBlocks(edge), low / OCCUPANCY_BLOCK)) {
        return true;
    }
    for (int node = size; node >= 1; node /= 2) {
        int nodes = size / node;
        int bestDistance = 0x7fffffff;
        for (int i = 0; i < nodes * nodes * nodes; i++) {
            int3 at = int3(i % nodes, (i / nodes) % nodes, i / (nodes * nodes));
            int3 nodeCenter = at * node * 2 + node;
            int3 offset = nodeCenter - size;
            // Not `dot`, which on integers needs the integer dot product feature.
            int distance = offset.x * offset.x + offset.y * offset.y + offset.z * offset.z;
            if (distance < bestDistance && voxelsEmpty(mesh.typesBase, edge, low + at * node, node)) {
                bestDistance = distance;
                halfVoxel = nodeCenter;
            }
        }
        if (bestDistance != 0x7fffffff) {
            return true;
        }
    }
    return false;
}

// Where a probe cell from voxel `low` of mesh `mesh`, a chunk `edge` voxels wide, puts its probe,
// in that chunk's half voxels from its corner, false when it has no room for one, from the
// cells baked when the chunk was added, see `bake_probe_cells`. The probe cell is `2^finer`
// baked cells wide, of which those in the chunk count: as the bake does, its center when
// they're all empty, else the center of the empty one nearest it, else the baked probe
// nearest it.
bool placeInBakedCells(MeshInfo mesh, int edge, int3 low, int finer, out float3 halfVoxel) {
    int cells = edge / PROBE_CELL;
    int width = PROBE_CELL << finer;
    float3 center = float3(low * 2 + width);
    halfVoxel = center;
    int3 from = max(floorDiv(low, PROBE_CELL), 0);
    int3 to = min(floorDiv(low + width - 1, PROBE_CELL), cells - 1);
    uint mask = mesh.probesBase + uint(cells * cells * cells);
    float bestEmpty = 1e30;
    float bestPlaced = 1e30;
    float3 placed = center;
    for (int z = from.z; z <= to.z; z++) {
        for (int y = from.y; y <= to.y; y++) {
            for (int x = from.x; x <= to.x; x++) {
                int3 baked = int3(x, y, z);
                uint index = uint(x + cells * (y + cells * z));
                bool hasVoxels = (chunkWords[mask + index / 32] & (1u << (index % 32))) != 0;
                if (!hasVoxels) {
                    float3 cellCenter = float3(baked * PROBE_CELL * 2 + PROBE_CELL);
                    float distance = dot(cellCenter - center, cellCenter - center);
                    if (distance < bestEmpty) {
                        bestEmpty = distance;
                        halfVoxel = cellCenter;
                    }
                    continue;
                }
                uint word = chunkWords[mesh.probesBase + index];
                if ((word & CELL_PLACED) != 0) {
                    float3 at = float3(baked * PROBE_CELL * 2) + float3(word & 63, (word >> 6) & 63, (word >> 12) & 63);
                    float distance = dot(at - center, at - center);
                    if (distance < bestPlaced) {
                        bestPlaced = distance;
                        placed = at;
                    }
                }
            }
        }
    }
    // None with voxels: the whole cell is empty, its probe at its center.
    if (bestEmpty < 1e30 && bestPlaced == 1e30) {
        halfVoxel = center;
        return true;
    }
    if (bestEmpty < 1e30) {
        return true;
    }
    halfVoxel = placed;
    return bestPlaced < 1e30;
}

// Where cascade `cascade`'s probe cell `cell` puts its probe, in the finest voxels, false when it
// has no room for one: as `renderer::lighting::probes::bake_probe_cells` places them, in the
// largest empty part of the cell nearest its center, from the chunk of the finest level
// holding the cell's center, see `chunkAtPoint`. One of the cascade's level, see `probeLevel`,
// or finer, has the cells it baked when it was added, see `placeInBakedCells`. A coarser one's
// voxels are searched, see `placeInVoxels`, at most 8 to a side of the cell, or when the cell
// is smaller than one of them, the probe is at its center when that's empty. A cell without a
// chunk is empty, with its probe at its center. Probes placed the same whatever the level of
// the chunks around them light those chunks the same.
bool probePlacement(int cascade, int3 cell, out float3 position) {
    int spacing = probeSpacing(cascade);
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
    MeshInfo mesh = meshes[meshIndex];
    int edge = int(uniforms.lighting.gridVoxelsPerEdge);
    float scale = float(1 << level);
    int cascadeLevel = probeLevel(cascade);
    if (level > cascadeLevel) {
        int size = PROBE_CELL >> (level - cascadeLevel);
        if (size == 0) {
            int3 voxel = int3(floor(center / scale)) - chunk * edge;
            return typeAt(mesh.typesBase, uint(edge), uint3(voxel)) == 0;
        }
        int3 low = cell * size - chunk * edge;
        int3 halfVoxel;
        bool placed = placeInVoxels(mesh, edge, low, size, halfVoxel);
        position = (float3(chunk * edge + low) + float3(halfVoxel) * 0.5) * scale;
        return placed;
    }
    int finer = cascadeLevel - level;
    int3 low = cell * (PROBE_CELL << finer) - chunk * edge;
    float3 halfVoxel;
    bool placed = placeInBakedCells(mesh, edge, low, finer, halfVoxel);
    position = (float3(chunk * edge) + halfVoxel * 0.5) * scale;
    return placed;
}

// Whether any of cascade `cascade`'s probe cells from `low` to `high` has voxels. From the
// masks baked after the probe words of the chunks they're in, each chunk looked up once, at
// the cascade's level and coarser. Chunks of finer levels only have chunks with voxels.
bool cellsHaveVoxels(int cascade, int3 low, int3 high) {
    int edge = int(uniforms.lighting.gridVoxelsPerEdge);
    int cells = edge / PROBE_CELL;
    int spacing = probeSpacing(cascade);
    // In the finest voxels.
    int3 regionMin = low * spacing;
    int3 regionMax = (high + 1) * spacing - 1;
    for (int level = 0; level < int(uniforms.lighting.gridLevels); level++) {
        int cellSize = PROBE_CELL << level;
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
                    if (level < probeLevel(cascade)) {
                        return true;
                    }
                    uint mask = meshes[meshIndex].probesBase + uint(cells * cells * cells);
                    int3 from = max(cellLow - chunk * cells, 0);
                    int3 to = min(cellHigh - chunk * cells, cells - 1);
                    for (int cz = from.z; cz <= to.z; cz++) {
                        for (int cy = from.y; cy <= to.y; cy++) {
                            for (int cx = from.x; cx <= to.x; cx++) {
                                uint index = uint(cx + cells * (cy + cells * cz));
                                if ((chunkWords[mask + index / 32] & (1u << (index % 32))) != 0) {
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

// A ray's distance when it hits nothing.
static const float SKY_DISTANCE = 1e9;

// Frames a probe converges faster after nearby chunks changed, for its light, and of those,
// for its distances [Majercik et al. 2021].
static const uint FAST_FRAMES = 10;
static const uint FAST_DISTANCE_FRAMES = 7;

// Whether a chunk of any level over slot `slot`'s probe's cell or a neighboring one changed
// since last frame, see `Renderer::mark_changed_probes`.
bool probeChanged(uint slot) {
    return (changedProbes[slot / 32] & (1u << (slot % 32))) != 0;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void classifyProbes(uint3 threadId: SV_DispatchThreadID) {
    int3 counts = int3(uniforms.probeVolume.volumeCounts);
    uint slot = threadId.x;
    if (slot >= probesPerCascade() * uniforms.probeVolume.cascades) {
        return;
    }
    // The cell in its cascade whose probe this slot holds.
    int cascade = cascadeOf(slot);
    uint inCascade = slot % probesPerCascade();
    int3 local = int3(inCascade % counts.x, (inCascade / counts.x) % counts.y, inCascade / (counts.x * counts.y));
    int3 volumeMin = uniforms.probeVolume.volumeMins[cascade].xyz;
    int3 cell = volumeMin + floorMod(local - volumeMin, counts);

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

    // Probes next to a chunk added, removed or remeshed converge faster for a while, and those
    // off are tried again.
    bool changed = probeChanged(slot);
    uint fast = fastFrames(probe.state);
    fast = changed ? FAST_FRAMES : (fast > 0 ? fast - 1 : 0);
    uint state = (probe.state & ((1u << PROBE_FAST_SHIFT) - 1)) | (fast << PROBE_FAST_SHIFT);
    if (changed) {
        state &= ~PROBE_OFF;
    }
    if (state != probe.state) {
        probe.state = state;
        probes[slot].state = state;
    }
    // Inside voxels, see `updateIrradiance`.
    if ((state & PROBE_OFF) != 0) {
        return;
    }

    // Each probe updates every few frames, in turn, unless nearby chunks just changed.
    uint period = updatePeriod(slot, fast);
    if (((uniforms.probeVolume.frameNumber + slot) & (period - 1)) != 0) {
        return;
    }

    // Only probes with voxels in or next to their cell shade any surface.
    if (!cellsHaveVoxels(cascade, cell - 1, cell + 1)) {
        return;
    }
    probes[slot].updates = probe.updates + 1;
    uint index;
    InterlockedAdd(activeProbes[ACTIVE_COUNT], 1, index);
    activeProbes[ACTIVE_LIST + index] = slot;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void planRays() {
    // The most rays, a power of two, each listed probe can trace within the frame's budget.
    uint count = activeProbes[ACTIVE_COUNT];
    uint budget = uniforms.probeVolume.raysPerFrame;
    uint rays = MIN_RAYS_PER_PROBE;
    while (rays < MAX_RAYS_PER_PROBE && rays * 2 * count <= budget) {
        rays *= 2;
    }
    activeProbes[ACTIVE_RAYS] = rays;
    activeProbes[ACTIVE_TRACE_GROUPS] = count * (rays / TRACE_GROUP);
    activeProbes[ACTIVE_UPDATE_GROUPS] = count;
}

// Ray `index` of `count` of every probe this frame: evenly spread over the sphere
// [Marques et al. 2013], rotated randomly each frame so probes see every direction over time.
float3 probeRayDirection(uint index, uint count) {
    float goldenRatio = (sqrt(5.0) + 1.0) * 0.5;
    float phi = 2.0 * PI * frac(float(index) * (goldenRatio - 1.0));
    float cosTheta = 1.0 - (2.0 * float(index) + 1.0) / float(count);
    float sinTheta = sqrt(saturate(1.0 - cosTheta * cosTheta));
    float3 direction = float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
    float4 q = uniforms.probeVolume.rayRotation;
    return direction + 2.0 * cross(q.xyz, cross(q.xyz, direction) + q.w * direction);
}

[shader("compute")]
[numthreads(TRACE_GROUP, 1, 1)]
void traceProbeRays(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID) {
    uint rays = activeProbes[ACTIVE_RAYS];
    uint groupsPerProbe = rays / TRACE_GROUP;
    uint listed = group.x / groupsPerProbe;
    uint ray = (group.x % groupsPerProbe) * TRACE_GROUP + thread.x;
    uint slot = activeProbes[ACTIVE_LIST + listed];
    Probe probe = probes[slot];
    float3 direction = probeRayDirection(ray, rays);
    float4 result;
    VoxelHit hit;
    // Probes further apart, as in coarser cascades, reach further, in proportion.
    float reach = uniforms.probeVolume.rayDistance * float(1 << probeLevel(cascadeOf(slot)));
    if (!traceVoxels(probe.position, direction, reach, false, false, hit)) {
        // The sky, or the ground past the world, through any see-through voxels.
        result = float4(ambientLight(direction) * hit.transmittance, SKY_DISTANCE);
    } else if (hit.distance <= 0.0 || hit.backface) {
        // In voxels, or seeing a voxel's back: dark, and nearer than it is, so the probe sees
        // it as shadowed and doesn't light through it [Majercik et al. 2021]. Negative, to
        // count toward the probe being inside voxels [Rohacek 2022].
        result = float4(0.0, 0.0, 0.0, -max(hit.distance * 0.2, 1e-3));
    } else {
        // Lit as the camera would see it, with last frame's light from the probes.
        float3 worldPosition = (probe.position + direction * hit.distance) * uniforms.lighting.gridVoxelSize;
        float3 light = sunAndIndirectLightAt(worldPosition, hit.normal, hit.normal, -direction, true, float3(-1.0))
            + lightsAtHit(worldPosition, hit.normal, uniforms.trace.shadowedLights);
        Material material = materialOf(hit.id);
        // Glowing voxels light what's around them through the probes, seen through any
        // see-through voxels on the way.
        float3 radiance = material.colorRoughness.rgb * light + material.emissionMetalness.rgb;
        result = float4(radiance * hit.transmittance, hit.distance);
    }
    probeRays[listed * rays + ray] = result;
}

// How much of its old value a probe keeps on its `updates`th update, of `rays` rays, `period`
// frames after its last: `hysteresis` for each frame, so probes updated every few frames follow
// changes as fast, but as much less for each fraction of the most rays it traced fewer. Fewer
// rays are noisier, so each counts for less, and the probes' noise stays the same however many
// share the frame's rays. A new probe's first updates are averaged, its first replacing what
// was there, so it settles quickly before it keeps as much.
float kept(float hysteresis, uint period, uint rays, uint updates) {
    float steady = pow(hysteresis, float(period) * float(rays) / float(MAX_RAYS_PER_PROBE));
    float averaged = float(max(updates, 1) - 1) / float(max(updates, 1));
    return min(steady, averaged);
}

groupshared float4 sharedResults[MAX_RAYS_PER_PROBE];
groupshared float3 sharedDirections[MAX_RAYS_PER_PROBE];

// Loads listed probe `listed`'s rays for the whole group of `groupSize` threads, and returns
// how many there are.
uint loadRays(uint listed, uint threadIndex, uint groupSize) {
    uint rays = activeProbes[ACTIVE_RAYS];
    for (uint i = threadIndex; i < rays; i += groupSize) {
        sharedResults[i] = probeRays[listed * rays + i];
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
    uint slot = activeProbes[ACTIVE_LIST + listed];
    uint columns = uniforms.probeVolume.atlasColumns;
    return int2(slot % columns, slot / columns) * (texels + 2) + texel;
}

[shader("compute")]
[numthreads(IRRADIANCE_TEXELS + 2, IRRADIANCE_TEXELS + 2, 1)]
void updateIrradiance(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID, uint threadIndex: SV_GroupIndex) {
    uint rays = loadRays(group.x, threadIndex, (IRRADIANCE_TEXELS + 2) * (IRRADIANCE_TEXELS + 2));
    uint slot = activeProbes[ACTIVE_LIST + group.x];
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
    float3 old = probeIrradianceStorage[texel].rgb;
    // How much of the old light the probe keeps, see `kept`, less when it changed a lot, so
    // big changes are let through faster [Majercik et al. 2021].
    uint fast = fastFrames(probe.state);
    float hysteresis = kept(uniforms.probeVolume.hysteresis * uniforms.probeVolume.hysteresisScale * (fast > 0 ? 0.5 : 1.0), updatePeriod(slot, fast), rays, probe.updates);
    float3 change = abs(irradiance - old);
    float largestChange = max(change.x, max(change.y, change.z));
    if (largestChange > 0.25) {
        hysteresis = max(hysteresis - 0.15, 0.0);
    }
    if (largestChange > 0.8) {
        hysteresis = 0.0;
    }
    probeIrradianceStorage[texel] = float4(lerp(irradiance, old, hysteresis), 1.0);

    // More than a quarter of the voxels its rays hit seen from the back, the probe is inside
    // voxels: it's turned off, until chunks near it change [Majercik et al. 2021]. Of those
    // hit, not of every ray: rays from inside rock left out for being solid throughout cross
    // it as open space, and many escape to the sky, though from open space no voxel's back can
    // be seen at all [Rohacek 2022].
    if (threadIndex == 0) {
        uint backs = 0;
        uint hits = 0;
        for (uint i = 0; i < rays; i++) {
            float distance = sharedResults[i].w;
            backs += distance < 0.0 ? 1 : 0;
            hits += distance < SKY_DISTANCE ? 1 : 0;
        }
        if (backs * 4 > hits) {
            probes[slot].state = probe.state | PROBE_OFF;
        }
    }
}

[shader("compute")]
[numthreads(DISTANCE_TEXELS + 2, DISTANCE_TEXELS + 2, 1)]
void updateDistances(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID, uint threadIndex: SV_GroupIndex) {
    uint rays = loadRays(group.x, threadIndex, (DISTANCE_TEXELS + 2) * (DISTANCE_TEXELS + 2));
    uint slot = activeProbes[ACTIVE_LIST + group.x];
    Probe probe = probes[slot];
    float maxDistance = MAX_PROBE_DISTANCE * float(1 << probeLevel(cascadeOf(slot)));

    // The mean distance and squared distance of the rays near the texel's direction that hit
    // voxels within the cage, or far when none did.
    float3 direction = texelDirection(int2(thread.xy), DISTANCE_TEXELS);
    float2 sum = float2(0.0);
    float nearWeight = 0.0;
    float totalWeight = 0.0;
    for (uint i = 0; i < rays; i++) {
        // Sharpened, so the distances' edges follow the voxels [Majercik et al. 2019].
        float weight = pow(max(dot(direction, sharedDirections[i]), 0.0), uniforms.probeVolume.distanceSharpness);
        // Backs' distances are negative, see `traceProbeRays`.
        float distance = abs(sharedResults[i].w);
        totalWeight += weight;
        if (distance <= maxDistance) {
            sum += weight * float2(distance, distance * distance);
            nearWeight += weight;
        }
    }
    if (totalWeight <= 1e-4) {
        return;
    }
    float2 moments = nearWeight > 1e-4
        ? sum / nearWeight
        : float2(maxDistance, maxDistance * maxDistance);

    int2 texel = atlasTexel(group.x, int2(thread.xy), DISTANCE_TEXELS);
    float2 old = probeDistancesStorage[texel].xy;
    uint fast = fastFrames(probe.state);
    float hysteresis = kept(uniforms.probeVolume.distanceHysteresis * uniforms.probeVolume.distanceHysteresisScale * (fast > FAST_FRAMES - FAST_DISTANCE_FRAMES ? 0.5 : 1.0), updatePeriod(slot, fast), rays, probe.updates);
    probeDistancesStorage[texel] = float4(lerp(moments, old, hysteresis), 0.0, 0.0);
}

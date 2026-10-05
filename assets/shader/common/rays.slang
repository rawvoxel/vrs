// Tracing rays through the voxel grid's voxels.

// How far off a surface, in voxels, its shadow ray starts, so it starts in the cell in front.
static const float SHADOW_RAY_OFFSET = 1.0 / 64.0;

// Moves a ray just past the boundary it's on, in voxels.
static const float RAY_EPSILON = 1.0 / 4096.0;
// Bounds on the work one ray does, so none can spin.
static const uint MAX_RAY_CHUNKS = 256;
static const uint MAX_RAY_VOXELS = 1024;

// How the last `traceVoxels` ended, for `DebugView::Trace`.
static const uint TRACE_HIT = 0;
static const uint TRACE_LEFT_TABLES = 1;
static const uint TRACE_PAST_DISTANCE = 2;
static const uint TRACE_OUT_OF_VOXELS = 3;
static const uint TRACE_OUT_OF_CHUNKS = 4;
static uint traceEnd;

// Where a ray from `origin` along `direction`, with `inverse` its reciprocal, leaves the box
// from `boxMin` to `boxMax`, and across which `axis`. At least `t` plus a little, so the ray
// always moves on.
float exitDistance(float3 origin, float3 direction, float3 inverse, float3 boxMin, float3 boxMax, float t, out int axis) {
    float3 exits = (select(direction > 0.0, boxMax, boxMin) - origin) * inverse;
    axis = exits.x < exits.y ? (exits.x < exits.z ? 0 : 2) : (exits.y < exits.z ? 1 : 2);
    return max(exits[axis], t + RAY_EPSILON);
}

// The float just below `x`, for a box's exclusive upper bound.
float3 below(float3 x) {
    return select(x > 0.0, asfloat(asuint(x) - 1), select(x < 0.0, asfloat(asuint(x) + 1), float3(-1e-30)));
}

// A box no position is outside, for `pastFace` and `inCell` to narrow.
struct RayBounds {
    float3 lowest;
    float3 highest;
};

RayBounds unbounded() {
    RayBounds bounds = { float3(-1e30), float3(1e30) };
    return bounds;
}

// Past the face on `axis` of the box from `boxMin` to `boxMax` that a ray going `direction`
// just left it through.
RayBounds pastFace(float3 direction, int axis, float3 boxMin, float3 boxMax) {
    RayBounds bounds = unbounded();
    if (direction[axis] > 0.0) {
        bounds.lowest[axis] = boxMax[axis];
    } else {
        bounds.highest[axis] = below(boxMin)[axis];
    }
    return bounds;
}

// Past the face on `axis` of the box from `boxMin` to `boxMax` that a ray going `direction`
// just entered it through.
RayBounds pastEntry(float3 direction, int axis, float3 boxMin, float3 boxMax) {
    RayBounds bounds = unbounded();
    if (direction[axis] > 0.0) {
        bounds.lowest[axis] = boxMin[axis];
    } else {
        bounds.highest[axis] = below(boxMax)[axis];
    }
    return bounds;
}

// In cell `cell` of cells `scale` wide.
RayBounds inCell(int3 cell, float scale) {
    RayBounds bounds = { float3(cell) * scale, below(float3(cell + 1) * scale) };
    return bounds;
}

// Where a ray from `origin` along `direction` is `t` along it, a little further, in `bounds`.
// Far from the origin, where a voxel's coordinates are only exact to a few thousandths, a
// position just past a boundary can round back onto it or short of it, so the bounds of the
// cell or region the ray is known to have entered keep it there, and it always moves on.
float3 rayPosition(float3 origin, float3 direction, float t, RayBounds bounds) {
    return clamp(origin + direction * (t + RAY_EPSILON), bounds.lowest, bounds.highest);
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
    // The level of the chunk it's in, and the chunk's mesh index.
    int level;
    uint mesh;
    // Whether the ray reached it through a side of its chunk where it has no face, from solid
    // voxels beyond, such as a chunk left out for being solid throughout: its back.
    bool backface;
};

// Whether a ray from `origin` along unit `direction`, both in the voxel grid's finest voxels,
// enters a solid voxel within `maxDistance`, and which. Rays leaving the box around every
// level's for good hit nothing, as do rays out of their budget of `MAX_RAY_CHUNKS` chunks and
// `MAX_RAY_VOXELS` voxels, which only very long rays through busy chunks run out of.
//
// An Amanatides-Woo DDA, visiting every voxel the ray passes through in order. It runs a
// chunk at a time, through the finest level's chunk wherever there is one, in that chunk's
// own voxels. It crosses a chunk without voxels in one step, and within a chunk crosses each
// block its occupancy says is empty in one step. With `skipStartSolid`, solid voxels the
// ray starts in are the surface it starts on, such as on a curved surface, so they're
// skipped. Light passes through see-through voxels, unless `stopAtSeeThrough`.
bool traceVoxels(float3 origin, float3 direction, float maxDistance, bool skipStartSolid, bool stopAtSeeThrough, out VoxelHit hit) {
    hit.distance = 0.0;
    hit.normal = float3(0.0);
    hit.id = 0;
    hit.transmittance = float3(1.0);
    hit.level = 0;
    hit.mesh = NO_CHUNK;
    hit.backface = false;
    traceEnd = TRACE_HIT;
    // Whether the last voxel was see-through, so each pane tints the light once.
    bool inGlass = false;
    int edge = int(uniforms.lighting.gridVoxelsPerEdge);
    int blocksPerEdge = occupancyBlocks(edge);
    // A ray along an axis would divide by zero.
    direction = select(abs(direction) < 1e-6, select(direction < 0.0, -1e-6, 1e-6), direction);
    float3 inverse = 1.0 / direction;
    int3 stepDirection = select(direction > 0.0, int3(1), int3(-1));

    float t = 0.0;
    // The axis the ray last crossed a boundary on, whose face it entered through.
    int axis = -1;
    bool leftSurface = !skipStartSolid;
    uint voxelSteps = 0;
    // Where the ray is known to be, past the last boundary it crossed.
    RayBounds bounds = unbounded();
    for (uint chunkStep = 0; chunkStep < MAX_RAY_CHUNKS; chunkStep++) {
        if (t > maxDistance) {
            traceEnd = TRACE_PAST_DISTANCE;
            return false;
        }
        float3 position = rayPosition(origin, direction, t, bounds);
        int level;
        int3 chunk;
        int coverLevel;
        int3 coverChunk;
        uint meshIndex = chunkAtPoint(position, level, chunk, coverLevel, coverChunk);
        if (coverLevel < 0) {
            // In no level's box. In the box around them all, the ray crosses the coarsest
            // level's chunk here, which every box holds all or none of, outside it, it goes on
            // to where it enters it, if it ever does.
            float3 gridMin = float3(uniforms.lighting.gridBoxMin.xyz);
            float3 gridMax = float3(uniforms.lighting.gridBoxMax.xyz);
            if (all(position >= gridMin) && all(position < gridMax)) {
                float coarseSize = float(edge << (int(uniforms.lighting.gridLevels) - 1));
                float3 coarseMin = floor(position / coarseSize) * coarseSize;
                t = exitDistance(origin, direction, inverse, coarseMin, coarseMin + coarseSize, t, axis);
                bounds = pastFace(direction, axis, coarseMin, coarseMin + coarseSize);
            } else {
                float3 near = (select(direction > 0.0, gridMin, gridMax) - origin) * inverse;
                float3 far = (select(direction > 0.0, gridMax, gridMin) - origin) * inverse;
                float enter = max(max(near.x, near.y), near.z);
                float leave = min(min(far.x, far.y), far.z);
                if (enter >= leave || leave <= t) {
                    traceEnd = TRACE_LEFT_TABLES;
                    return false;
                }
                axis = near.x > near.y ? (near.x > near.z ? 0 : 2) : (near.y > near.z ? 1 : 2);
                t = max(enter, t + RAY_EPSILON);
                bounds = pastEntry(direction, axis, gridMin, gridMax);
            }
            leftSurface = true;
            inGlass = false;
            continue;
        }
        // The ray crosses the finest table's chunk here before looking again, no finer chunk
        // is in it.
        float coverSize = float(edge << coverLevel);
        float3 regionMin = float3(coverChunk) * coverSize;
        if (meshIndex == NO_CHUNK) {
            // Nothing to hit, so straight to where the ray leaves it.
            t = exitDistance(origin, direction, inverse, regionMin, regionMin + coverSize, t, axis);
            bounds = pastFace(direction, axis, regionMin, regionMin + coverSize);
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
                // On to the region of the cell it stepped into.
                bounds = inCell(cell, scale);
                break;
            }
            int3 local = cell - chunkMin;
            if (++voxelSteps > MAX_RAY_VOXELS) {
                traceEnd = TRACE_OUT_OF_VOXELS;
                return false;
            }
            int3 block = local / OCCUPANCY_BLOCK;
            if (!blockOccupied(mesh.occupancyBase, blocksPerEdge, block)) {
                // Nothing to hit, so straight to where the ray leaves the block.
                int3 blockMin = chunkMin + block * OCCUPANCY_BLOCK;
                int3 blockMax = min(blockMin + OCCUPANCY_BLOCK, chunkMin + edge);
                t = exitDistance(origin, direction, inverse, float3(blockMin) * scale, float3(blockMax) * scale, t, axis);
                if (t > maxDistance) {
                    traceEnd = TRACE_PAST_DISTANCE;
                    return false;
                }
                leftSurface = true;
                inGlass = false;
                bounds = pastFace(direction, axis, float3(blockMin) * scale, float3(blockMax) * scale);
                cell = int3(floor(rayPosition(origin, direction, t, bounds) / scale));
                crossings = (float3(cell + select(direction > 0.0, int3(1), int3(0))) * scale - origin) * inverse;
                continue;
            }

            uint id = typeAt(mesh.typesBase, uint(edge), uint3(local));
            bool glass = false;
            bool solid = false;
            if (id != 0) {
                Material material = materialOf(id);
                glass = isSeeThrough(material) && !stopAtSeeThrough;
                solid = !glass;
                // Light passes through see-through voxels, tinted by each pane once, and ones
                // tinting it to next to nothing are as good as solid.
                if (glass && !inGlass) {
                    hit.transmittance *= material.colorRoughness.rgb;
                    solid = all(hit.transmittance < 1.0 / 256.0);
                }
            }
            if (solid && leftSurface) {
                hit.distance = t;
                hit.id = id;
                hit.level = level;
                hit.mesh = meshIndex;
                if (axis >= 0) {
                    hit.normal[axis] = -float(stepDirection[axis]);
                    // Only chunks' sides tell faces from backs, where the voxel beyond is in
                    // another chunk, or none; within one, the voxel the ray came from is empty.
                    bool throughSide = local[axis] == (stepDirection[axis] > 0 ? 0 : edge - 1);
                    if (throughSide && mesh.kind == KIND_AXIS_ALIGNED) {
                        int face = axis * 2 + (stepDirection[axis] > 0 ? 0 : 1);
                        hit.backface = !boundaryFace(mesh, edge, face, local);
                    }
                }
                return true;
            }
            if (!solid) {
                leftSurface = true;
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
                traceEnd = TRACE_PAST_DISTANCE;
                return false;
            }
        }
    }
    traceEnd = TRACE_OUT_OF_CHUNKS;
    return false;
}

// How much of the sun's light gets along a ray from `origin` along `direction`, both in the
// voxel grid's voxels, to the trace distance: none when a solid voxel is in the way, and
// tinted by any see-through ones.
float3 sunVisibility(float3 origin, float3 direction) {
    VoxelHit hit;
    return traceVoxels(origin, direction, uniforms.lighting.traceDistance, true, false, hit) ? float3(0.0) : hit.transmittance;
}

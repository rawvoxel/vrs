// Finding voxels: chunks' packed types and occupancy, and the chunk table's chunks around the
// camera.

// Matches `renderer::lighting::voxel_grid::NO_CHUNK`, a chunk table entry without a chunk.
static const uint NO_CHUNK = 0xffffffff;

// Moves a point on a face into the voxel behind it, in voxels.
static const float FACE_NUDGE = 1.0 / 256.0;
// Matches `mesh::chunk::OCCUPANCY_BLOCK`, voxels along each edge of a block of a chunk's occupancy.
static const int OCCUPANCY_BLOCK = 8;

// World position `world` in the voxel grid's finest voxels.
float3 toGrid(float3 world) {
    return world / max(uniforms.lighting.gridVoxelSize, 1e-6);
}

// The type id of voxel `cell` of the chunk whose types start at word `base`.
// Matches `mesh::chunk::VoxelTypes::type_at`.
uint typeAt(uint base, uint edge, uint3 cell) {
    uint header = chunkWords[base];
    uint bits = header & 0xff;
    uint perWord = 32 / bits;
    uint index = cell.x + edge * (cell.y + edge * cell.z);
    uint word = chunkWords[base + 1 + (header >> 8) + index / perWord];
    uint entry = (word >> ((index % perWord) * bits)) & ((1u << bits) - 1);
    uint paletteWord = chunkWords[base + 1 + (entry >> 1)];
    return (paletteWord >> ((entry & 1) * 16)) & 0xffff;
}

// Blocks of a chunk's occupancy along each edge, for chunks `edge` voxels wide.
int occupancyBlocks(int edge) {
    return (edge + OCCUPANCY_BLOCK - 1) / OCCUPANCY_BLOCK;
}

// Whether any voxel of block `block` of the chunk whose occupancy starts at word `base` isn't
// empty. Matches `mesh::chunk::VoxelTypes::occupancy`.
bool blockOccupied(uint base, int blocksPerEdge, int3 block) {
    uint index = uint(block.x + blocksPerEdge * (block.y + blocksPerEdge * block.z));
    return (chunkWords[base + index / 32] & (1u << (index % 32))) != 0;
}

// Whether voxel `local` of the chunk whose types start at `typesBase` and occupancy at
// `occupancyBase`, `edge` voxels wide, blocks light: solid and not see-through.
bool voxelOccludes(uint typesBase, uint occupancyBase, int edge, int3 local) {
    // Most voxels around a face are in the open, in blocks with none.
    if (!blockOccupied(occupancyBase, occupancyBlocks(edge), local / OCCUPANCY_BLOCK)) {
        return false;
    }
    uint id = typeAt(typesBase, uint(edge), uint3(local));
    return id != 0 && !typeIsSeeThrough(id);
}

// The chunk table key of level `level`'s chunk at `relative` in its box: the level, then its
// place, 10 bits an axis. Matches `renderer::lighting::voxel_grid::TableLayout::key`.
uint chunkKey(int level, int3 relative) {
    return (uint(level) << 30) | (uint(relative.z) << 20) | (uint(relative.y) << 10) | uint(relative.x);
}

// A chunk key's bits mixed for its first slot, lowbias32 [Wellons 2018]. Matches
// `renderer::lighting::voxel_grid::chunk_hash`.
uint chunkHash(uint key) {
    uint x = key;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// The mesh index of level `level`'s chunk at chunk coordinate `chunk`, in that level's
// chunks, `NO_CHUNK` for none, or `outside` set when it's past the level's box.
uint levelChunk(int level, int3 chunk, out bool outside) {
    int3 relative = chunk - uniforms.lighting.chunkTableCorners[level].xyz;
    outside = any(relative < 0) || any(relative >= uniforms.lighting.chunkTableSizes[level].xyz);
    if (outside) {
        return NO_CHUNK;
    }
    // From the slot its key hashes to on, up to its key or a slot without a chunk, of which
    // the table, at most a quarter full, has plenty.
    uint key = chunkKey(level, relative);
    uint mask = uniforms.lighting.chunkTableMask;
    uint slot = chunkHash(key) & mask;
    for (uint looked = 0; looked <= mask; looked++) {
        uint2 entry = chunkTable[slot];
        if (entry.y == NO_CHUNK || entry.x == key) {
            return entry.y;
        }
        slot = (slot + 1) & mask;
    }
    return NO_CHUNK;
}

// The mesh index of the chunk of the finest level with one holding point `position`, in the
// finest voxels, its level and its chunk coordinate, or `NO_CHUNK`. Either way, the finest
// level whose box holds the point and its chunk there, -1 past every box: no finer level's
// box overlaps that chunk.
uint chunkAtPoint(float3 position, out int level, out int3 chunk, out int coverLevel, out int3 coverChunk) {
    int edge = int(uniforms.lighting.gridVoxelsPerEdge);
    level = -1;
    chunk = int3(0);
    coverLevel = -1;
    coverChunk = int3(0);
    for (int l = 0; l < int(uniforms.lighting.gridLevels); l++) {
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

// Whether level `level`'s voxel `voxel`, in that level's voxels, blocks light. Open where
// there's no chunk.
bool gridOccludes(int level, int3 voxel) {
    int edge = int(uniforms.lighting.gridVoxelsPerEdge);
    int3 chunk = floorDiv(voxel, edge);
    bool outside;
    uint meshIndex = levelChunk(level, chunk, outside);
    if (meshIndex == NO_CHUNK) {
        return false;
    }
    MeshInfo mesh = meshes[meshIndex];
    return voxelOccludes(mesh.typesBase, mesh.occupancyBase, edge, voxel - chunk * edge);
}

// Voxels along each edge of a probe's cell, and between neighboring probes, in the finest
// cascade, and of the cells each chunk bakes its probes' places in. Matches
// `renderer::lighting::probes::PROBE_CELL`.
static const int PROBE_CELL = 16;

// Where chunk `mesh`'s boundary faces start in `chunkWords`, after its probe cells: for chunks a
// multiple of `PROBE_CELL` voxels wide, a word for each cell and a bit, 32 to a word, see
// `renderer::lighting::probes::bake_probe_cells`.
uint boundaryFacesBase(MeshInfo mesh) {
    int edge = int(mesh.voxelsPerEdge);
    if (edge % PROBE_CELL != 0) {
        return mesh.probesBase;
    }
    uint cells = uint(edge / PROBE_CELL);
    uint count = cells * cells * cells;
    return mesh.probesBase + count + (count + 31) / 32;
}

// Whether voxel `local` of axis aligned chunk `mesh`, `edge` voxels wide, along its side `face`,
// numbered as `mesh::Face`, has a face out of that side, see `mesh::chunk::boundary_faces`.
bool boundaryFace(MeshInfo mesh, int edge, int face, int3 local) {
    int axis = face / 2;
    int2 along = axis == 0 ? local.yz : (axis == 1 ? local.xz : local.xy);
    uint bit = uint(face * edge * edge + along.x + edge * along.y);
    return (chunkWords[boundaryFacesBase(mesh) + bit / 32] & (1u << (bit % 32))) != 0;
}

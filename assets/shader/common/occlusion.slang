// Voxel ambient occlusion: how open a face is, from the voxels around it.

// The two axes along a face facing unit axis `normal`.
void faceAxes(int3 normal, out int3 u, out int3 v) {
    u = normal.x != 0 ? int3(0, 1, 0) : int3(1, 0, 0);
    v = normal.z != 0 ? int3(0, 1, 0) : int3(0, 0, 1);
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

// How open the face of a level `level` chunk facing unit axis `normal` is at `position` on
// it, in that level's voxels, see `cornerOpenness`.
float gridOpenness(int level, float3 position, int3 normal) {
    int3 front = int3(floor(position - float3(normal) * FACE_NUDGE)) + normal;
    int3 u;
    int3 v;
    faceAxes(normal, u, v);
    float2 along = float2(frac(dot(position, float3(u))), frac(dot(position, float3(v))));
    bool around[9];
    for (int i = 0; i < 9; i++) {
        int2 at = int2(i % 3, i / 3) - 1;
        around[i] = i != 4 && gridOccludes(level, front + u * at.x + v * at.y);
    }
    return cornerOpenness(around, along);
}

// Constants and encodings with nothing to do with voxels in particular.

// How bright each channel of linear color looks.
static const float3 LUMINANCE = float3(0.3, 0.6, 0.1);
static const float PI = 3.14159265;

// `value` divided by `divisor`, rounded down.
int3 floorDiv(int3 value, int divisor) {
    return int3(floor(float3(value) / float(divisor)));
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

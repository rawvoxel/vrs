// What voxels of each type are made of.

// Added to a material's metalness when it's see-through. Matches `renderer::material::GpuMaterial`.
static const float SEE_THROUGH = 2.0;

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

// Whether light passes through `material`, see `Material`.
bool isSeeThrough(Material material) {
    return material.emissionMetalness.w >= SEE_THROUGH;
}

// Whether voxel type `id` is see-through.
bool typeIsSeeThrough(uint id) {
    return isSeeThrough(materialOf(id));
}

// How metallic `material` is, from 0 to 1.
float metalnessOf(Material material) {
    float packed = material.emissionMetalness.w;
    return isSeeThrough(material) ? packed - SEE_THROUGH : packed;
}

// The color a surface shows: its material lit, reflecting what's around it.

// Faces with a roughness below this reflect what's around them sharply, traced by the shadow
// pass, blending into the probes' blurry light toward it.
static const float GLOSSY_ROUGHNESS = 0.5;
// How many of the lights a reflection ray's hit shadows: one, the brightest, as reflections
// are traced for many pixels.
static const uint REFLECTION_SHADOWED_LIGHTS = 1;

// The number of the side of a voxel `normal` faces most, from 1 to 6, for exact and
// interpolated normals alike.
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
    if (!traceVoxels(origin, direction, maxDistance, false, false, hit)) {
        return ambientLight(direction) * hit.transmittance;
    }
    if (hit.distance <= 0.0) {
        return float3(0.0);
    }
    float3 worldPosition = (origin + direction * hit.distance) * uniforms.lighting.gridVoxelSize;
    uint shadowed = min(uniforms.trace.shadowedLights, REFLECTION_SHADOWED_LIGHTS);
    float3 light = sunAndIndirectLightAt(worldPosition, hit.normal, hit.normal, -direction, false, float3(-1.0))
        + lightsAtHit(worldPosition, hit.normal, shadowed);
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
    return (exponent + 8.0) / (8.0 * PI) * pow(saturate(dot(normal, halfway)), exponent);
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

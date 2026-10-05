// The light reaching a surface: from the sun, and from everywhere else, the probes' or the
// sky's and the ground's.

// The light from everywhere but the sun reaching a surface facing `normal` where no probe
// covers it: the sky's and the ground's, blended by how far it faces up.
float3 ambientLight(float3 normal) {
    return lerp(ubo.lighting.groundColor.rgb, ubo.lighting.skyColor.rgb, normal.y * 0.5 + 0.5);
}

// The light from everywhere but the sun reaching a surface at `gridPosition`, in the shadow
// grid's voxels, with unit normal `normal`, seen from direction `toViewer`: the probes', or
// the sky and ground's outside them.
//
// The finest cascade covering the surface lights it, blending into the next over its last
// cell. For the probes' own rays, `probeRay` is true: only probes updated before count, and
// where a cascade covers the surface but none of its probes count, there's no light yet. The
// sky's would leak into closed rooms, and with every bounce adding to the next, take many
// frames to fade.
float3 indirectLight(float3 gridPosition, float3 normal, float3 toViewer, bool probeRay) {
    float3 ambient = ambientLight(normal);
    if (ubo.probeVolume.volumeMins[0].w == 0) {
        return ambient;
    }
    // Probes the probes' rays sample must have been updated once before, so the rays don't
    // see this frame's own update.
    uint minUpdates = probeRay ? 2 : 1;
    float3 probesLight = float3(0.0);
    // How much is left for coarser cascades, and how much of that none of the probes lit.
    float remaining = 1.0;
    float unlit = 0.0;
    for (int cascade = 0; cascade < int(ubo.probeVolume.frame.z) && remaining > 0.0; cascade++) {
        float coverage;
        bool found;
        float3 light = sampleIrradiance(cascade, gridPosition, normal, toViewer, minUpdates, coverage, found);
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

// The sun's light reaching a surface at `worldPosition` with unit normal `normal`.
// `faceNormal` is the triangle's own, which rays leave the surface along. `sunVisible` is how
// much of the sun reaches it, tinted by any see-through voxels in the way, when already known,
// or negative for a shadow ray to find out.
float3 sunLight(float3 worldPosition, float3 normal, float3 faceNormal, float3 sunVisible) {
    LightingUniforms lighting = ubo.lighting;
    float facing = dot(normal, lighting.sunDirection.xyz);
    if (facing <= 0.0) {
        return float3(0.0);
    }
    float3 visible = sunVisible;
    if (visible.x < 0.0) {
        visible = lighting.shadowTable.w > 0
            ? sunVisibility(toGrid(worldPosition) + faceNormal * SHADOW_RAY_OFFSET, lighting.sunDirection.xyz)
            : float3(1.0);
    }
    return lighting.sunColor.rgb * facing * visible;
}

// The light reaching a surface, straight from the sun and the lights, and from everywhere else.
struct SurfaceLight {
    float3 direct;
    float3 indirect;
};

// All the light from the sun and everywhere else reaching a surface, see `sunLight` and
// `indirectLight`.
float3 lightAt(float3 worldPosition, float3 normal, float3 faceNormal, float3 toViewer, bool probeRay, float3 sunVisible) {
    return sunLight(worldPosition, normal, faceNormal, sunVisible)
        + indirectLight(toGrid(worldPosition), normal, toViewer, probeRay);
}

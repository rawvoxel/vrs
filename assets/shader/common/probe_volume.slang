// Sampling the light probes' cascades.

// Voxels along each edge of a probe's cell, and between neighboring probes, in the finest
// cascade. Matches `renderer::lighting::probes::PROBE_CELL`.
static const int PROBE_CELL = 16;
// Matches `renderer::lighting::probes::volume::IRRADIANCE_TEXELS` and
// `renderer::lighting::probes::volume::DEPTH_TEXELS`, each with a one texel border.
static const int IRRADIANCE_TEXELS = 8;
static const int DEPTH_TEXELS = 14;
// Irradiance is stored to the power of 1 / 5, which blends a drop in light as fast as a
// rise looks [Majercik et al. 2021].
static const float IRRADIANCE_GAMMA = 5.0;

// Probes in each cascade.
uint probesPerCascade() {
    return ubo.gi.volumeCounts.x * ubo.gi.volumeCounts.y * ubo.gi.volumeCounts.z;
}

// The cascade of the probe in slot `slot`.
int cascadeOf(uint slot) {
    return int(slot / probesPerCascade());
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
    float spacing = float(PROBE_CELL << cascade);
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

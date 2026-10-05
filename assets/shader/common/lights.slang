// The point and spot lights: their falloff, culling them into tiles, choosing the brightest,
// and their shadows.

// Pixels along each edge of a tile lights are culled for. Matches
// `renderer::lighting::shadows::pass::LIGHT_TILE`.
static const uint LIGHT_TILE = 16;
// The most lights a tile keeps, its brightest. Matches
// `renderer::lighting::shadows::pass::TILE_LIGHTS`.
static const uint TILE_LIGHTS = 63;
// Words a tile's light list takes: its count, then its lights.
static const uint TILE_WORDS = TILE_LIGHTS + 1;
// The most lights a frame has. Matches `renderer::lighting::lights::MAX_LIGHTS`.
static const uint MAX_LIGHTS = 255;
// The most of a pixel's lights that are shadowed. Matches `renderer::MAX_SHADOWED_LIGHTS`.
static const uint MAX_SHADOWED_LIGHTS = 4;
// No light, in 8 bits. Lights are numbered below `MAX_LIGHTS`, so it's never one.
static const uint NO_LIGHT = 0xff;
// A light shadow texel without shadowed lights: every 8 bits `NO_LIGHT`.
static const uint2 NO_SHADOWED_LIGHTS = uint2(0xffffffff, 0);

// The tile of lights pixel `pixel` is in.
uint tileOf(uint2 pixel) {
    return (pixel.y / LIGHT_TILE) * ubo.shadow.lights.y + pixel.x / LIGHT_TILE;
}

// How much of a light's brightness gets `distance` from it, with `range` its reach: with the
// square of the distance, windowed to nothing at its range.
float lightFalloff(float distance, float range) {
    float x = distance / range;
    float window = saturate(1.0 - x * x * x * x);
    return window * window / (distance * distance + 1.0);
}

// The light `light` gives a point at `worldPosition`, before shadows and whichever way it
// faces: its color, fading with distance, within its cone. Also where it is, from the point,
// in world units.
float3 lightColorAt(Light light, float3 worldPosition, out float3 toLight, out float distance) {
    float3 offset = light.positionRange.xyz - worldPosition;
    distance = length(offset);
    toLight = offset / max(distance, 1e-6);
    float cone = 1.0;
    if (light.colorSpot.w > 0.0) {
        float along = dot(-toLight, light.directionOuter.xyz);
        cone = smoothstep(light.directionOuter.w, light.inner.x, along);
    }
    return light.colorSpot.rgb * lightFalloff(distance, light.positionRange.w) * cone;
}

// The light `light` gives a surface at `worldPosition` facing `normal`, before shadows, see
// `lightColorAt`.
float3 lightReaching(Light light, float3 worldPosition, float3 normal, out float3 toLight, out float distance) {
    return lightColorAt(light, worldPosition, toLight, distance) * max(dot(normal, toLight), 0.0);
}

// How bright `light` is at `worldPosition`, whichever way a surface there faces, to choose
// which lights to shadow.
float lightStrength(Light light, float3 worldPosition) {
    float3 toLight;
    float distance;
    return dot(lightColorAt(light, worldPosition, toLight, distance), LUMINANCE);
}

// A few of the brightest lights, brightest first, by light number, `NO_LIGHT` past the last.
struct BrightestLights {
    uint lights[MAX_SHADOWED_LIGHTS];
    float strengths[MAX_SHADOWED_LIGHTS];
    // The light each gives, for those that keep it.
    float3 colors[MAX_SHADOWED_LIGHTS];
};

BrightestLights noBrightestLights() {
    BrightestLights brightest;
    for (uint k = 0; k < MAX_SHADOWED_LIGHTS; k++) {
        brightest.lights[k] = NO_LIGHT;
        brightest.strengths[k] = 0.0;
        brightest.colors[k] = float3(0.0);
    }
    return brightest;
}

// Keeps light `light` of `strength`, giving `color`, among the brightest `kept` of `brightest`.
void keepBrightest(inout BrightestLights brightest, uint kept, uint light, float strength, float3 color) {
    for (uint k = 0; k < kept; k++) {
        if (strength > brightest.strengths[k]) {
            for (uint j = kept - 1; j > k; j--) {
                brightest.lights[j] = brightest.lights[j - 1];
                brightest.strengths[j] = brightest.strengths[j - 1];
                brightest.colors[j] = brightest.colors[j - 1];
            }
            brightest.lights[k] = light;
            brightest.strengths[k] = strength;
            brightest.colors[k] = color;
            return;
        }
    }
}

// How much of `light`'s light gets to `origin`, in the shadow grid's voxels: 0 behind solid
// voxels, dimmed by see-through ones. Traced from the light, past the solid voxels it starts
// in, so a lamp set in a glowing block or a fixture isn't shadowed by it, to half a voxel
// short of the surface, which a curved one could otherwise shadow itself in.
float lightShadow(Light light, float3 origin) {
    if (ubo.lighting.shadowTable.w == 0) {
        return 1.0;
    }
    float3 position = toGrid(light.positionRange.xyz);
    float3 offset = origin - position;
    float distance = length(offset);
    if (distance <= 0.5) {
        return 1.0;
    }
    VoxelHit hit;
    if (traceVoxels(position, offset / distance, distance - 0.5, true, false, hit)) {
        return 0.0;
    }
    return dot(hit.transmittance, float3(1.0 / 3.0));
}

// The light the frame's lights give a ray's hit at `worldPosition`, facing `normal`: each
// light's, the brightest `shadowed` shadowed.
float3 lightsAtHit(float3 worldPosition, float3 normal, uint shadowed) {
    shadowed = min(shadowed, MAX_SHADOWED_LIGHTS);
    float3 origin = toGrid(worldPosition) + normal * SHADOW_RAY_OFFSET;
    BrightestLights brightest = noBrightestLights();
    float3 total = float3(0.0);
    for (uint i = 0; i < ubo.shadow.lights.x; i++) {
        float3 toLight;
        float distance;
        float3 reaching = lightReaching(lights[i], worldPosition, normal, toLight, distance);
        float strength = dot(reaching, LUMINANCE);
        if (strength > 0.0) {
            total += reaching;
            keepBrightest(brightest, shadowed, i, strength, reaching);
        }
    }
    // The brightest are shadowed, by taking away what doesn't get through.
    for (uint k = 0; k < shadowed && brightest.lights[k] != NO_LIGHT; k++) {
        total -= brightest.colors[k] * (1.0 - lightShadow(lights[brightest.lights[k]], origin));
    }
    return max(total, 0.0);
}

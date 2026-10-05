// The data the CPU shares with the shaders: mesh records, uniforms, lights, materials and probes,
// each matching its Rust struct, and the limits they're sized by.

// Matches `mesh::MeshInfo`, and `MeshInfo` in `cull.slang`.
struct MeshInfo {
    float3 boundsMin;
    uint indexCount;
    float3 boundsMax;
    uint firstIndex;
    float3 origin;
    float scale;
    uint vertexBase;
    uint typesBase;
    uint kind;
    uint visible;
    uint voxelsPerEdge;
    float positionSteps;
    uint occupancyBase;
    uint probesBase;
};

// Matches `mesh::MeshKind`.
static const uint KIND_COLORED = 0;
static const uint KIND_GREEDY = 1;
static const uint KIND_PACKED = 2;
static const uint KIND_LIT = 3;

// The most levels of chunks the shadow grid has, each of voxels twice as big as the last.
// Matches `renderer::MAX_SHADOW_LEVELS`.
static const int MAX_SHADOW_LEVELS = 4;
// The most cascades of light probes, each spaced twice as far apart as the last. Matches
// `lighting::gi::MAX_CASCADES`.
static const int MAX_CASCADES = 4;

// Matches `renderer::LightingUniforms`.
struct LightingUniforms {
    // Normalized, toward the sun.
    float4 sunDirection;
    float4 sunColor;
    float4 skyColor;
    float4 groundColor;
    // Chunks per side of each level's shadow table in w, 0 without shadows. The corners are
    // in `shadowLevels`.
    int4 shadowTable;
    // The grid's voxel size, voxels per edge and the shadow distance in voxels, then how many
    // levels of chunks it has.
    float4 shadowGrid;
    // How strong voxel ambient occlusion is, then the debug view, see `renderer::DebugView`,
    // the exposure and the tonemapping, see `renderer::Tonemapping`.
    float4 options;
    // The chunk coordinate of each level's shadow table's corner, in that level's chunks.
    int4 shadowLevels[MAX_SHADOW_LEVELS];
    // The sky's zenith and horizon colors, then the cosine of the sun's radius, its
    // brightness, the haze distance, and 1 when there's a sky, see `renderer::Sky`.
    float4 sky[3];
};

// Matches `lighting::gi::GiUniforms`.
struct GiUniforms {
    // Each cascade's probe cell at its low corner, in its own cells, w is 1 when the probes
    // light the frame.
    int4 volumeMins[MAX_CASCADES];
    // Probes along each axis of each cascade, w is probes per row of the atlases.
    uint4 volumeCounts;
    // This frame's random rotation of the probes' rays, a quaternion.
    float4 rayRotation;
    float4 cameraPosition;
    // x is the frame's number, y the rays traced each frame, z how many cascades there are, w
    // the frames between the finest cascade's probes' updates.
    uint4 frame;
    // `ProbeTuning`'s hysteresis, depth hysteresis, depth sharpness and self-shadow bias.
    float4 tuning;
    // `ProbeTuning`'s ray distance, then this frame's multipliers of the hysteresis for light
    // and for distances, below 1 while the probes converge faster after a change.
    float4 rays;
};

// Matches `lighting::shadows::ShadowUniforms`.
struct ShadowUniforms {
    // From clip space back to the world.
    float4x4 inverseViewProj;
    // Frame pixels per shadow texel along each axis, then the shadow images' width and
    // height, then 1 when any material is glossy, for reflections to be traced.
    uint4 params;
    // The frame's width and height, then 1 when the shadow pass ran this frame, then 1 when it
    // also lit faces with the light from everywhere but the sun.
    uint4 frameSize;
    // How many lights there are, the light tiles across and down the frame, and how many of
    // each pixel's lights are shadowed.
    uint4 lights;
};

// A point or spot light. Matches `lighting::lights::GpuLight`.
struct Light {
    // The position, then how far its light reaches, in world units.
    float4 positionRange;
    // Its brightness at its center, then 1 for a spot light.
    float4 colorSpot;
    // The spot's direction, then the cosine of its outer angle.
    float4 directionOuter;
    // The cosine of the spot's inner angle.
    float4 inner;
};

// How voxels of a type look and light, in linear color. Matches `renderer::material::GpuMaterial`.
struct Material {
    // The surface's color, then its roughness.
    float4 colorRoughness;
    // The light it gives off, then its metalness, plus `SEE_THROUGH` when it's see-through.
    float4 emissionMetalness;
};

// Matches `shader::uniforms::FrameUniforms`.
struct Uniforms {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
    LightingUniforms lighting;
    GiUniforms gi;
    ShadowUniforms shadow;
};

// A light probe, matches `lighting::gi::volume::PROBE_SIZE`. Positions are in the shadow grid's
// finest voxels, cells in its cascade's.
struct Probe {
    // The probe cell the probe's slot holds.
    int3 cell;
    // `PROBE_PLACED` when the cell has room for the probe.
    uint state;
    float3 position;
    // How many frames have updated it since it was placed.
    uint updates;
};

static const uint PROBE_PLACED = 1;
// Where a probe's state counts down the frames it converges faster after nearby chunks
// changed, 8 bits [Majercik et al. 2021].
static const uint PROBE_FAST_SHIFT = 8;

// The data the CPU shares with the shaders: mesh records, uniforms, lights, materials and probes,
// each matching its Rust struct, and the limits they're sized by.

// Matches `renderer::mesh_pool::MeshInfo`, and `MeshInfo` in `cull.slang`.
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

// Matches `renderer::mesh_pool::MeshKind`.
static const uint KIND_COLORED = 0;
static const uint KIND_AXIS_ALIGNED = 1;
static const uint KIND_PACKED = 2;

// The most levels of chunks the voxel grid has, each of voxels twice as big as the last.
// Matches `renderer::MAX_GRID_LEVELS`.
static const int MAX_GRID_LEVELS = 4;
// The most cascades of light probes, each spaced twice as far apart as the last. Matches
// `renderer::lighting::probes::MAX_CASCADES`.
static const int MAX_CASCADES = 4;

// Chunks along each side of each level's chunk table, centered on the camera. Matches
// `renderer::lighting::voxel_grid::CHUNK_TABLE_SIDE`.
static const int CHUNK_TABLE_SIDE = 32;

// Matches `renderer::LightingUniforms`.
struct LightingUniforms {
    // Normalized, toward the sun.
    float3 sunDirection;
    // How strongly voxel ambient occlusion darkens corners, from 0 to 1.
    float ambientOcclusion;
    float3 sunColor;
    // See `renderer::DebugView`.
    uint debugView;
    float3 skyColor;
    float exposure;
    float3 groundColor;
    // See `renderer::Tonemapping`.
    uint tonemapping;
    // The voxel grid: its finest voxels' size, voxels along each chunk's edge, its levels, 0
    // without a grid, and how far sun shadow and reflection rays reach, in the finest voxels.
    float gridVoxelSize;
    uint gridVoxelsPerEdge;
    uint gridLevels;
    float traceDistance;
    // The chunk coordinate of each level's chunk table's low corner, in that level's chunks.
    int4 chunkTableCorners[MAX_GRID_LEVELS];
    // The sky, see `renderer::Sky`: its zenith and horizon colors, the cosine of the sun disc's
    // radius, its brightness, the haze distance, and 1 when there's a sky.
    float3 skyZenith;
    float sunDiscCosine;
    float3 skyHorizon;
    float sunBrightness;
    float hazeDistance;
    uint hasSky;
    float2 padding;
};

// Matches `renderer::lighting::probes::ProbeUniforms`.
struct ProbeUniforms {
    // Each cascade's probe cell at its low corner, in its own cells.
    int4 volumeMins[MAX_CASCADES];
    // Probes along each axis of each cascade, and how many cascades there are.
    uint3 volumeCounts;
    uint cascades;
    // Probes per row of the atlases, the frame's number, the rays traced each frame, and the
    // frames between the finest cascade's probes' updates.
    uint atlasColumns;
    uint frameNumber;
    uint raysPerFrame;
    uint updateInterval;
    // This frame's random rotation of the probes' rays, a quaternion.
    float4 rayRotation;
    // `renderer::ProbeTuning`'s.
    float hysteresis;
    float distanceHysteresis;
    float distanceSharpness;
    float selfShadowBias;
    float rayDistance;
    // This frame's multipliers of the hysteresis for light and for distances, below 1 while
    // the probes converge faster after a change.
    float hysteresisScale;
    float distanceHysteresisScale;
    // 1 when the probes light the frame.
    uint enabled;
};

// Matches `renderer::lighting::trace::TraceUniforms`.
struct TraceUniforms {
    // Frame pixels per trace texel along each axis, then 1 when any material is glossy, for
    // reflections to be traced, then the trace images' width and height.
    uint pixelsPerTexel;
    uint traceReflections;
    uint2 imageSize;
    // How many lights there are, the light tiles across and down the frame, and how many of
    // each pixel's lights are shadowed.
    uint lightCount;
    uint tilesAcross;
    uint tilesDown;
    uint shadowedLights;
};

// A point or spot light. Matches `renderer::lighting::lights::GpuLight`.
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

// Matches `renderer::shader::uniforms::FrameUniforms`.
struct Uniforms {
    float4x4 view;
    float4x4 proj;
    // From clip space back to the world, to find a pixel's position from its depth.
    float4x4 inverseViewProj;
    float3 cameraPosition;
    float padding;
    // The frame's width and height in pixels, then 1 when the trace pass ran, then 1 when it
    // also found the light from everywhere but the sun, for shading to upscale.
    uint2 frameSize;
    uint usesTracePass;
    uint tracedIndirect;
    LightingUniforms lighting;
    ProbeUniforms probeVolume;
    TraceUniforms trace;
};

// A light probe, matches `renderer::lighting::probes::volume::PROBE_SIZE`. Positions are in the
// voxel grid's finest voxels, cells in its cascade's.
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

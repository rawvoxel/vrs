// Draws the mesh pool's meshes, pulling each one's vertices in whichever format it uses.
//
// Colored meshes carry their own colors and normals, and are lit as rough surfaces. Chunk meshes
// are colored by voxel: a fragment finds the voxel it belongs to in the chunk's packed types, then
// that type's material. Then they're lit by the sun and the lights, with the shadows the trace pass
// traced, the light probes or the sky and the ground, darkened by ambient occlusion, and when
// glossy, reflect what's around them. Below full resolution, the trace pass lit them with the light
// from everywhere but the sun too, and they upscale it.

#include "common.slang"

// Matches `mesh::POSITION_BIAS`.
static const float POSITION_BIAS = 16.0;
// Matches `mesh::Face::normal`, indexed by the face in an axis aligned vertex's flags bits 1 to 3.
static const float3 FACE_NORMALS[6] = {
    float3(-1.0, 0.0, 0.0),
    float3(1.0, 0.0, 0.0),
    float3(0.0, -1.0, 0.0),
    float3(0.0, 1.0, 0.0),
    float3(0.0, 0.0, -1.0),
    float3(0.0, 0.0, 1.0),
};
// How far axis aligned quads are pushed out past their edges, in pixels, so where one quad's corner
// meets the middle of another's edge, rounding can't open a crack between them.
static const float QUAD_OVERLAP_PIXELS = 0.25;
// Marks an axis aligned vertex that isn't one of a quad's 4 corners in a row, in its flags.
// Matches `mesh::LOOSE_FLAG`.
static const uint LOOSE_FLAG = 16;
// How much nearer or further, relative to its distance, a trace texel's surface can be than
// a pixel's and still shadow it.
static const float SHADOW_DEPTH_TOLERANCE = 0.08;

// The sun's visibility and the surface's distance for each block of pixels, from the shadow
// pass, see `trace_pass.slang`.
[[vk::binding(15, 0)]] Sampler2D<float4> shadows;
// The lights reaching each tile of pixels, see `cullLights` in `trace_pass.slang`.
[[vk::binding(18, 0)]] StructuredBuffer<uint> lightTiles;
// Each block's shadowed lights and how much of each gets through, see `traceSurfaces`.
[[vk::binding(20, 0)]] Texture2D<uint2> lightShadows;
// Each block's sharp reflection when its voxel is glossy, see `traceSurfaces`.
[[vk::binding(22, 0)]] Texture2D<float4> reflections;
// Each block's light from everywhere but the sun, with its ambient occlusion, see
// `traceSurfaces`.
[[vk::binding(24, 0)]] Texture2D<float4> indirect;

struct VSOutput {
    float4 position : SV_Position;
    float3 color;
    // Chunk local, in the chunk's voxels.
    float3 voxelPosition;
    float3 worldPosition;
    float3 normal;
    // A `KIND_` constant.
    nointerpolation uint kind;
    nointerpolation uint typesBase;
    nointerpolation uint occupancyBase;
    nointerpolation uint voxelsPerEdge;
    // World units per voxel.
    nointerpolation float scale;
};

// Whether pixels the trace pass didn't light with the light from everywhere but the sun find
// it themselves, from the probes, with ambient occlusion, else take the sky's and the
// ground's. Off for `PipelinePass::ColorOverUpscaledIndirect`, which leaves that code out,
// whose registers would slow every pixel.
[vk::constant_id(0)] const bool PER_PIXEL_INDIRECT = true;

// Match `renderer::DebugView`.
static const uint DEBUG_LIGHT = 1;
static const uint DEBUG_DIRECT = 2;
static const uint DEBUG_INDIRECT = 3;
static const uint DEBUG_AMBIENT_OCCLUSION = 4;
static const uint DEBUG_TRACE = 6;
static const uint DEBUG_LEVELS = 7;
static const uint DEBUG_CASCADES = 8;

// Matches `mesh::octahedral::decode`.
float3 octahedralDecode(uint word) {
    float2 folded = float2(word & 0xffff, word >> 16) / 65535.0 * 2.0 - 1.0;
    float3 normal = float3(folded.x, 1.0 - abs(folded.x) - abs(folded.y), folded.y);
    float below = max(-normal.y, 0.0);
    normal.x += normal.x >= 0.0 ? -below : below;
    normal.z += normal.z >= 0.0 ? -below : below;
    return normalize(normal);
}

// Matches `mesh::unpack_position`.
float3 unpackPosition(uint positionXy, uint positionZFlags, float steps) {
    float3 quantized = float3(positionXy & 0xffff, positionXy >> 16, positionZFlags & 0xffff);
    return quantized / steps - POSITION_BIAS;
}

[shader("vertex")]
VSOutput meshVert(uint vertexIndex: SV_VertexID, uint meshIndex: SV_StartInstanceLocation) {
    MeshInfo mesh = meshes[meshIndex];

    VSOutput output;
    output.kind = mesh.kind;
    output.typesBase = mesh.typesBase;
    output.occupancyBase = mesh.occupancyBase;
    output.voxelsPerEdge = mesh.voxelsPerEdge;
    output.scale = mesh.scale;
    output.color = float3(1.0, 0.0, 1.0);
    output.normal = float3(0.0, 1.0, 0.0);

    float3 local;
    // Away from the middle of the vertex's quad, for axis aligned quads, which overlap a little.
    float3 outward = float3(0.0);
    uint word = mesh.vertexBase;
    if (mesh.kind == KIND_COLORED) {
        word += vertexIndex * 5;
        local = asfloat(uint3(vertexWords[word], vertexWords[word + 1], vertexWords[word + 2]));
        output.color = unpackUnorm4x8ToFloat(vertexWords[word + 3]).rgb;
        output.normal = octahedralDecode(vertexWords[word + 4]);
    } else if (mesh.kind == KIND_AXIS_ALIGNED) {
        word += vertexIndex * 2;
        uint positionZFlags = vertexWords[word + 1];
        local = unpackPosition(vertexWords[word], positionZFlags, mesh.positionSteps);
        output.normal = FACE_NORMALS[min((positionZFlags >> 17) & 7, 5)];
        // Axis aligned quads are 4 vertices in a row, loose vertices stay put.
        if (((positionZFlags >> 16) & LOOSE_FLAG) == 0) {
            uint quad = mesh.vertexBase + (vertexIndex & ~3u) * 2;
            float3 center = float3(0.0);
            for (uint corner = 0; corner < 4; corner++) {
                center += unpackPosition(vertexWords[quad + corner * 2], vertexWords[quad + corner * 2 + 1], mesh.positionSteps);
            }
            outward = sign(local - center * 0.25);
        }
    } else {
        word += vertexIndex * 3;
        local = unpackPosition(vertexWords[word], vertexWords[word + 1], mesh.positionSteps);
        output.normal = octahedralDecode(vertexWords[word + 2]);
    }
    output.voxelPosition = local;

    float4 world = float4(mesh.origin + local * mesh.scale, 1.0);
    output.position = mul(uniforms.proj, mul(uniforms.view, world));
    if (any(outward != 0.0)) {
        // A pixel's width in the world at the vertex's distance, for any projection.
        float pixel = 2.0 * output.position.w / (abs(uniforms.proj[1][1]) * float(uniforms.frameSize.y));
        world.xyz += outward * pixel * QUAD_OVERLAP_PIXELS;
        output.position = mul(uniforms.proj, mul(uniforms.view, world));
    }
    output.worldPosition = world.xyz;
    return output;
}

// The chunk voxel containing voxel position `p`. Past the chunk's edges, its edge voxel.
uint3 voxelAt(float3 p, uint edge) {
    return uint3(clamp(int3(floor(p)), int3(0), int3(int(edge) - 1)));
}

// The 4 trace texels nearest a frame pixel, weighted by how near each is, of those whose
// surface is about as far from the camera as the pixel's, so shadows don't bleed across
// edges. With none, the one whose surface is nearest in distance counts alone. All zero
// without the trace pass.
struct ShadowTaps {
    int2 texels[4];
    float weights[4];
    float4 sun[4];
    // The one whose surface is nearest in distance.
    int closest;
};

// The taps for the frame pixel centered on `pixel`, whose surface is `distance` from the
// camera.
ShadowTaps shadowTaps(float2 pixel, float distance) {
    ShadowTaps taps;
    float scale = float(uniforms.trace.pixelsPerTexel);
    float2 size = float2(uniforms.trace.imageSize);
    // Each texel holds the pixel at its block's center.
    float2 texel = (pixel - 0.5 - floor(scale * 0.5)) / scale;
    int2 base = int2(floor(texel));
    float2 alpha = texel - float2(base);
    float totalWeight = 0.0;
    float closest = 1e9;
    int closestTap = 0;
    for (int i = 0; i < 4; i++) {
        int2 offset = int2(i & 1, i >> 1);
        float2 tap = clamp(float2(base + offset), 0.0, size - 1.0);
        taps.texels[i] = int2(tap);
        taps.sun[i] = shadows.SampleLevel((tap + 0.5) / size, 0.0);
        float difference = abs(taps.sun[i].w - distance) / max(distance, 1e-3);
        if (difference < closest) {
            closest = difference;
            closestTap = i;
        }
        float2 bilinear = lerp(1.0 - alpha, alpha, float2(offset));
        taps.weights[i] = bilinear.x * bilinear.y * saturate(1.0 - difference / SHADOW_DEPTH_TOLERANCE);
        totalWeight += taps.weights[i];
    }
    for (int i = 0; i < 4; i++) {
        taps.weights[i] = totalWeight > 1e-4 ? taps.weights[i] / totalWeight : float(i == closestTap);
    }
    taps.closest = closestTap;
    return taps;
}

// The sun's visibility, tinted by see-through voxels, upscaled from `taps`.
float3 upscaledShadow(ShadowTaps taps) {
    float3 visible = float3(0.0);
    for (int i = 0; i < 4; i++) {
        visible += taps.weights[i] * taps.sun[i].rgb;
    }
    return visible;
}

// Face number `face`'s value from `texels`, which hold each block's and its face's number,
// upscaled from `taps`, from those of the same face. `found` is false when none were.
float3 upscaledByFace(Texture2D<float4> texels, ShadowTaps taps, uint face, out bool found) {
    float3 sum = float3(0.0);
    float totalWeight = 0.0;
    for (int i = 0; i < 4; i++) {
        if (taps.weights[i] <= 0.0) {
            continue;
        }
        float4 texel = texels.Load(int3(taps.texels[i], 0));
        if (uint(texel.w + 0.5) == face) {
            sum += taps.weights[i] * texel.rgb;
            totalWeight += taps.weights[i];
        }
    }
    found = totalWeight > 0.0;
    return found ? sum / totalWeight : float3(0.0);
}

// The value from `texels`, which hold each block's and its face's number, upscaled from
// `taps`, from those that found a face, whichever it is. Weighted as the taps are, or equally
// when none of those are. `found` is false when none found a face.
float3 upscaledAnyFace(Texture2D<float4> texels, ShadowTaps taps, out bool found) {
    float3 weighted = float3(0.0);
    float totalWeight = 0.0;
    float3 any = float3(0.0);
    float count = 0.0;
    for (int i = 0; i < 4; i++) {
        float4 texel = texels.Load(int3(taps.texels[i], 0));
        if (texel.w > 0.0) {
            weighted += taps.weights[i] * texel.rgb;
            totalWeight += taps.weights[i];
            any += texel.rgb;
            count += 1.0;
        }
    }
    found = count > 0.0;
    if (totalWeight > 0.0) {
        return weighted / totalWeight;
    }
    return found ? any / count : float3(0.0);
}

// How much of light `lightIndex`'s light gets through, upscaled from `taps`, from those that
// shadowed it. Lights none of them shadowed aren't shadowed.
float upscaledLightShadow(ShadowTaps taps, uint lightIndex) {
    float visible = 0.0;
    float totalWeight = 0.0;
    for (int i = 0; i < 4; i++) {
        if (taps.weights[i] <= 0.0) {
            continue;
        }
        uint2 shadowed = lightShadows.Load(int3(taps.texels[i], 0));
        for (uint k = 0; k < MAX_SHADOWED_LIGHTS; k++) {
            if (((shadowed.x >> (k * 8)) & NO_LIGHT) == lightIndex) {
                visible += taps.weights[i] * float((shadowed.y >> (k * 8)) & 0xff) / 255.0;
                totalWeight += taps.weights[i];
                break;
            }
        }
    }
    return totalWeight > 0.0 ? visible / totalWeight : 1.0;
}

// Whether chunk voxel `cell` blocks light from reaching a corner: solid and not see-through.
// Past the chunk's edges, the voxel grid's voxel at `gridCell`, when the chunk is the grid's
// finest size, otherwise open.
bool occludes(VSOutput input, int3 cell, int3 gridCell) {
    int edge = int(input.voxelsPerEdge);
    if (all(cell >= 0) && all(cell < edge)) {
        return voxelOccludes(input.typesBase, input.occupancyBase, edge, cell);
    }
    float gridSize = uniforms.lighting.gridVoxelSize;
    bool inGrid = uniforms.lighting.gridLevels > 0
        && abs(input.scale - gridSize) <= gridSize * 1e-5
        && int(uniforms.lighting.gridVoxelsPerEdge) == edge;
    return inGrid && gridOccludes(0, gridCell);
}

// How open an axis aligned face is where `input` lies on it, from 0, enclosed, to 1: each corner of
// its voxel darkens with the voxels around it in front of the face, two solid sides closing
// it entirely, and the pixel blends its voxel's four corners [classic voxel ambient
// occlusion].
float voxelOcclusion(VSOutput input, int3 normal) {
    // The face's voxel, the empty one in front, and the same in the voxel grid.
    int3 solid = int3(floor(input.voxelPosition - float3(normal) * FACE_NUDGE));
    int3 front = solid + normal;
    int3 gridFront = int3(floor(toGrid(input.worldPosition) - float3(normal) * FACE_NUDGE)) + normal;
    int3 u;
    int3 v;
    faceAxes(normal, u, v);
    float2 along = float2(frac(dot(input.voxelPosition, float3(u))), frac(dot(input.voxelPosition, float3(v))));

    // The 8 voxels around the one in front, in its plane, each looked up once.
    bool around[9];
    for (int i = 0; i < 9; i++) {
        int2 at = int2(i % 3, i / 3) - 1;
        int3 offset = u * at.x + v * at.y;
        around[i] = i != 4 && occludes(input, front + offset, gridFront + offset);
    }
    return cornerOpenness(around, along);
}

[shader("fragment")]
float4 meshFrag(VSOutput input) : SV_Target {
    // The triangle's outward normal, from screen derivatives. Taken before any branch, so
    // every pixel of the quad takes part.
    float3 faceNormal = normalize(cross(ddy(input.voxelPosition), ddx(input.voxelPosition)));

    // A chunk's face lies between its solid voxel and the empty cell in front, so its
    // fragments are nudged inside first. Curved surfaces can still be in an empty cell, then
    // the voxel one cell behind is taken.
    bool colored = input.kind == KIND_COLORED;
    uint edge = input.voxelsPerEdge;
    uint id = 0;
    if (!colored) {
        id = typeAt(input.typesBase, edge, voxelAt(input.voxelPosition - faceNormal * FACE_NUDGE, edge));
        if (id == 0) {
            id = typeAt(input.typesBase, edge, voxelAt(input.voxelPosition - faceNormal * (1.0 + FACE_NUDGE), edge));
        }
    }

    // Axis aligned normals are exact already, packed and colored ones are interpolated between
    // vertices.
    float3 normal = input.kind == KIND_AXIS_ALIGNED ? input.normal : normalize(input.normal);
    float3 fromCamera = input.worldPosition - uniforms.cameraPosition;
    float3 toViewer = -normalize(fromCamera);
    // With the trace pass, the sun's and the lights' shadows are upscaled from it.
    bool shadowPass = uniforms.usesTracePass != 0;
    ShadowTaps taps = {};
    float3 sunVisible = float3(-1.0);
    if (shadowPass) {
        taps = shadowTaps(input.position.xy, length(fromCamera));
        sunVisible = upscaledShadow(taps);
    }
    // The light from everywhere but the sun, darkened in corners and crevices, upscaled from
    // the trace pass when it lit faces, else for each pixel.
    SurfaceLight light;
    light.direct = sunLight(input.worldPosition, normal, faceNormal, sunVisible);
    uint debugView = uniforms.lighting.debugView;
    bool upscaled = false;
    // Packed faces take that of the side of their voxel they face most.
    if (shadowPass && uniforms.tracedIndirect != 0 && debugView != DEBUG_AMBIENT_OCCLUSION) {
        light.indirect = upscaledByFace(indirect, taps, faceNumber(normal), upscaled);
        // Along edges, where no texel near is of the same face, those of any face, which look
        // the same, rather than some pixels finding their own and slowing those around them.
        if (!upscaled) {
            light.indirect = upscaledAnyFace(indirect, taps, upscaled);
        }
    }
    float openness = 1.0;
    if (!upscaled && !PER_PIXEL_INDIRECT) {
        light.indirect = ambientLight(normal);
    } else if (!upscaled) {
        light.indirect = indirectLight(toGrid(input.worldPosition), normal, toViewer, false);
        float occlusionStrength = uniforms.lighting.ambientOcclusion;
        if (input.kind == KIND_AXIS_ALIGNED && occlusionStrength > 0.0) {
            openness = lerp(1.0, voxelOcclusion(input, int3(round(normal))), occlusionStrength);
        }
        light.indirect *= openness;
    }
    Material material = materialOf(id);
    if (colored) {
        material.colorRoughness = float4(input.color, 1.0);
        material.emissionMetalness = float4(0.0);
    }
    float roughness = material.colorRoughness.w;
    // The sun's and the lights' highlights, which smooth faces reflect.
    float3 highlights = float3(0.0);
    bool glossy = roughness < GLOSSY_ROUGHNESS;
    if (glossy) {
        highlights += light.direct * highlight(normal, toViewer, uniforms.lighting.sunDirection, roughness);
    }

    // The lights reaching the pixel's tile.
    if (shadowPass && uniforms.trace.lightCount > 0) {
        uint tile = tileOf(uint2(input.position.xy)) * TILE_WORDS;
        uint count = lightTiles[tile];
        for (uint i = 0; i < count; i++) {
            uint lightIndex = lightTiles[tile + 1 + i];
            float3 toLight;
            float distance;
            float3 reaching = lightReaching(lights[lightIndex], input.worldPosition, normal, toLight, distance);
            if (any(reaching > 0.0)) {
                float3 reached = reaching * upscaledLightShadow(taps, lightIndex);
                light.direct += reached;
                if (glossy) {
                    highlights += reached * highlight(normal, toViewer, toLight, roughness);
                }
            }
        }
    }

    // What the face reflects: the light reaching it from everywhere when it's rough, and when
    // it's glossy, blending into a sharp reflection, traced by the trace pass where it could,
    // or the probes' light from the way the viewer's reflected, the sky's and ground's without
    // `PER_PIXEL_INDIRECT`.
    float3 lit = light.direct + light.indirect;
    float3 reflection = lit;
    if (glossy) {
        float3 reflected = reflect(-toViewer, normal);
        bool found = false;
        float3 sharp = float3(0.0);
        if (shadowPass) {
            sharp = upscaledByFace(reflections, taps, faceNumber(normal), found);
        }
        if (!found) {
            sharp = PER_PIXEL_INDIRECT
                ? indirectLight(toGrid(input.worldPosition), reflected, toViewer, false)
                : ambientLight(reflected);
        }
        float sharpness = saturate(1.0 - roughness / GLOSSY_ROUGHNESS);
        reflection = lerp(lit, sharp + highlights, sharpness);
    }

    switch (debugView) {
    case DEBUG_LIGHT:
        return float4(lit, 1.0);
    case DEBUG_DIRECT:
        return float4(light.direct, 1.0);
    case DEBUG_INDIRECT:
        return float4(light.indirect, 1.0);
    case DEBUG_AMBIENT_OCCLUSION:
        return float4(float3(openness), 1.0);
    case DEBUG_CASCADES:
        return float4(cascadeShares(toGrid(input.worldPosition), normal, toViewer), 1.0);
    case DEBUG_TRACE:
    case DEBUG_LEVELS: {
        // The trace pass's texel for this pixel, see `traceSurfaces`.
        float4 trace = shadowPass ? reflections.Load(int3(int2(input.position.xy) / int(uniforms.trace.pixelsPerTexel), 0)) : float4(0.0);
        if (debugView == DEBUG_LEVELS) {
            static const float3 LEVEL_COLORS[4] = {
                float3(0.0, 1.0, 0.0), float3(0.0, 0.3, 1.0), float3(1.0, 1.0, 0.0), float3(1.0, 0.0, 1.0)
            };
            bool traced = shadowPass && trace.x > 0.5;
            // Not found: white where the search started in a voxel, red where it met none.
            float3 notFound = trace.z > 0.5 ? float3(1.0) : float3(1.0, 0.0, 0.0);
            return float4(traced ? LEVEL_COLORS[min(uint(round(trace.z)), 3)] : (shadowPass ? notFound : float3(0.0)), 1.0);
        }
        float brightness = 0.2 + 0.8 * trace.y;
        // Rays that gave up rather than reaching the sky or a voxel stand out.
        uint end = uint(round(trace.w));
        float3 color = trace.x > 0.5 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
        color = end == TRACE_PAST_DISTANCE ? float3(1.0, 1.0, 0.0)
            : end == TRACE_OUT_OF_VOXELS ? float3(1.0, 0.0, 1.0)
            : end == TRACE_OUT_OF_CHUNKS ? float3(0.0, 1.0, 1.0)
            : color;
        return float4(color * brightness, 1.0);
    }
    default: {
        float3 shaded = shadeSurface(material, normal, toViewer, lit, reflection);
        return float4(toDisplay(hazed(shaded, fromCamera)), 1.0);
    }
    }
}

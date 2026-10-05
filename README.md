# vrs

A Vulkan 1.3 renderer for voxel games, which culls meshes on the GPU.

Every mesh lives in one shared vertex buffer and one index buffer. Each frame a compute shader
frustum culls them, then occlusion culls them against a depth pyramid in two phases, and draws
what's visible with indirect draws the GPU counts itself. Meshes can be created and destroyed
every frame, as chunks are remeshed, without stalling.

Chunks use compact vertex formats, 8 bytes a vertex for axis aligned faces and 12 for any
other, and are colored per voxel: each pixel finds the voxel it belongs to in the chunk's
palette packed voxel types, then that type's material in the renderer's palette.

They're lit by a sun whose shadows are traced through the voxels, point and spot lights, and
dynamic diffuse global illumination from light probes around the camera, with glossy
reflections, glowing and see-through voxels, and levels of detail, under a sky.

## Usage

```rust,no_run
use ash::vk;
use vrs::{
    App,
    mesh::Vertex,
    renderer::{DrawList, Renderer},
};

// A windowed app takes the window's handles, see `App::new`.
let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
let app = App::headless(app_info).unwrap();
let mut renderer = Renderer::new(&app, 800, 600).unwrap();

// Uploaded by the next frame, then drawn every frame it's listed.
let (normal, col) = ([0.0, 0.0, -1.0], [1.0, 0.0, 0.0]);
let triangle = renderer
    .create_mesh(
        // Counter-clockwise on screen, so it faces the camera.
        &[0, 2, 1],
        &[
            Vertex { pos: [0.0, -0.5, 0.5], normal, col },
            Vertex { pos: [0.5, 0.5, 0.5], normal, col },
            Vertex { pos: [-0.5, 0.5, 0.5], normal, col },
        ],
    )
    .unwrap();

// Every frame, with the camera's `view` and `proj` for culling.
let draws = DrawList {
    meshes: &[&triangle],
    ..Default::default()
};
renderer.submit(800, 600, &draws).unwrap();

renderer.destroy_mesh(triangle);
// The renderer must be dropped before the app.
drop(renderer);
```

- **Meshes** are created, staging their data straight away, and drawn by listing their
  `MeshId`s in a `DrawList`. Ids aren't `Copy`, so `destroy_mesh` taking one means a destroyed
  mesh can't be drawn again. Its space is reused once no frame in flight can still draw it.
  Indices are relative to each mesh's first vertex.
- **Chunk meshes** come from `create_chunk_mesh` and a `mesh::chunk::ChunkMesh`:
  `AxisAlignedVertex`es for axis aligned faces, which keep a face index instead of a normal, or
  `PackedVertex`es for any surface. Positions are chunk local, placed by the chunk's `origin`
  and `voxel_size`, which includes its LOD. Its `VoxelTypes` are palette compressed, and
  already packed words can be handed over as they are with `VoxelTypes::from_words`. Type ids index the
  palette from `set_materials`, or `set_palette` for plain colors, 0 is empty space. Cells
  past a chunk's edge take the chunk's own edge voxel, there's no lookup into neighbors, and
  see-through faces are drawn like any other.
- **Colored meshes** come from `create_mesh`, `mesh::Vertex`es in world space with colors and
  normals of their own, for what sits among the voxels without being voxels, such as leaves
  and grass. They're lit like chunks, as rough surfaces, but cast no shadows and the light
  probes don't see them, and a card seen from both sides needs both windings.
- **Materials**: each palette entry is a `Material`: a color, light it gives off, roughness,
  metalness, and whether it's see-through. Glowing voxels light what's around them through
  the light probes. See-through voxels, such as glass, are drawn as solid faces, but light
  passes through them tinted by their color, once per pane. Smooth materials reflect what's
  around them, more at grazing angles, and metals reflect in their own color.
- **Lighting**: `Renderer::lighting` lights chunks with sky and ground colors, blended by
  which way a face points, and a sun. With a `ShadowGrid`, shadow rays are traced toward the
  sun through the voxels with a DDA, across chunks, so hills shadow their neighbors. Each
  chunk also keeps one bit per 8³ block of voxels saying whether any is solid, and rays cross
  empty blocks in one step. The grid is the chunks of one voxel size and width, placed on a
  grid from the world's origin, and with `ShadowGrid::levels`, coarser levels of detail of
  them, each of voxels twice as big. Rays go through the finest level's chunk wherever there
  is one. Chunks of other sizes are lit but cast no shadows, and chunks with no faces, such as
  under the ground, should still be created for their voxels to cast shadows.
  With shadows or lights, frames draw their depth first, trace the shadows from it, then draw
  their color over it, which saves tracing for pixels drawn over. `Lighting::shadow_resolution`
  traces one shadow ray per 2x2 or 4x4 pixels instead of every pixel, upscaling the shadows by
  distance so they don't bleed across edges. Below full resolution, the shadow pass also
  finds the light from everywhere but the sun, the probes' and the ambient occlusion, for
  each block of pixels, which shading upscales by face. That leaves sampling probes out of
  shading, which is most of its cost, so chunks the shadow grid doesn't hold are lit by the
  sky and ground colors alone then.
  Axis aligned faces' corners and crevices are darkened by voxel ambient occlusion, from the voxels
  around each face, `Lighting::ambient_occlusion`.
- **Lights**: `Renderer::lights` holds up to 255 point and spot lights. Each frame lists the
  lights reaching each 16x16 pixel tile, the brightest when more than fit, and each pixel's
  brightest `Lighting::shadowed_lights`, up to 4, are shadowed at the shadow resolution.
  Shadow rays start from the light, past any voxels it's inside, so a light can sit inside a
  glowing block as its lamp.
  The light probes see the lights too.
- **Reflections**: smooth voxels, with a roughness under 0.5, have a reflection ray traced
  for each shadow texel and upscaled like the shadows, which blends into the probes' light
  from the reflected direction as they get rougher. The sun's and the lights' highlights are
  added on top.
- **Global illumination**: with `RendererConfig::global_illumination`, a volume of light
  probes 16 voxels apart follows the camera, and lights chunks inside it with light bounced
  off the voxels around them, in place of the sky and ground colors. With
  `ProbeConfig::cascades`, further volumes follow it too, each with probes twice as far apart,
  reaching twice as far, for coarser levels of detail. It's dynamic diffuse global
  illumination (Majercik et al. 2019 and 2021):
  - When a chunk is added, each 16³ cell of it bakes where its probe goes: the largest empty
    part of the cell nearest its center, found breadth first, or none when it's solid.
  - The probes with voxels in or next to their cell share a fixed budget of rays,
    `ProbeConfig::rays_per_frame`, so the cost stays steady. Each updates every
    `ProbeTuning::update_interval` frames, 4 by default, taking turns, and twice as rarely
    in each coarser cascade, but blends in as much more each time, so the light follows
    changes as fast. Each traces its share through the voxels with the same DDA, shades
    what they hit with the sun, the lights and last frame's probes, which adds a bounce every
    frame, and blends the result into small octahedral maps of the light reaching it and how
    far away voxels are.
  - Probes converge faster for a while after the sun or sky change, or chunks near them are
    added, removed or remeshed. `Renderer::converge_lighting` does it for everything, such as
    after a cut to another place. `Lighting::probe_tuning` sets how they blend and sample.
  - Shading blends the 8 probes around each point, weighted by where each actually is, which
    way they face and whether their distances say they can see it, which keeps light from
    leaking through walls. The finest cascade covering a point lights it, blending into the
    next at its edge.

  It needs a `ShadowGrid` whose chunks are a multiple of 16 voxels wide, and
  `Lighting::global_illumination` turns it off and on. Probes spaced wider than a room can
  leak light into it, so coarser cascades suit far away terrain rather than interiors.
- **Sky and tonemapping**: with a `Lighting::sky`, a gradient from the horizon up to the
  zenith, the ground color below it, and the sun's disc are drawn wherever no mesh is,
  instead of the clear color, and chunks fade into it with distance by `Sky::haze_distance`.
  `Lighting::exposure` scales shaded colors, and `Lighting::tonemapping` brings those brighter
  than the display shows into its range, with Reinhard or ACES, rather than clipping them.
  Both happen as each pixel is shaded, so there's no extra pass.
- **Debug views**: `Lighting::debug_view` shows all the light reaching chunks, only the
  sun's and lights', only the rest, the ambient occlusion, or the probes drawn as cubes.
- **Winding**: front faces wind counter-clockwise on screen, backface culling skips the rest.
- **Cameras**: `DrawList::proj` maps into Vulkan's clip space, where y points down and depth
  runs from 0 to 1. Flip y in an OpenGL style projection, such as glam's `perspective_rh`,
  otherwise the image is mirrored and the wrong faces are culled.
- **The mesh pool** is defragmented when its free space is too scattered for a new mesh, and
  doubled when there isn't enough of it. Both copy every mesh into new buffers, stalling
  `create_mesh` until the copies finish, which is rare since the pool doubles each time.
  Without growth, `create_mesh` fails with `ERROR_OUT_OF_DEVICE_MEMORY` when it's full.
- **Settings**: a `RendererConfig` for `Renderer::with_config` sets the mesh pool's starting
  size, and starting values for growing it, vsync, occlusion culling and backface culling. At
  runtime `grow_mesh_pool`, `occlusion_culling`, `backface_culling` and `clear_color` are
  fields, vsync is `set_vsync`.
- **Headless**: `App::headless` renders into images of the renderer's own, with no window.

`App` and `Renderer` are `Send + Sync`, so they can be shared with other threads.

## Requirements

- A GPU and driver supporting Vulkan 1.3. The renderer picks a discrete GPU first, then
  integrated, then anything else, that can also present to the window.
- `drawIndirectFirstInstance`, `shaderDrawParameters`, `multiDrawIndirect` and
  `drawIndirectCount`, which every desktop GPU has: each indirect draw names its mesh by its
  first instance, which the mesh shaders read, and culling writes how many draws there are.
- The compiled shaders in `assets/shader` are committed, so building only needs `slangc`, from
  the Vulkan SDK, after editing a `.slang` file.
- Debug builds turn on the Vulkan validation layer when it's installed.

## Examples

```sh
cargo run -p vrs --example triangle
cargo run -p vrs --example cube
cargo run -p vrs --example materials --release
cargo run -p vrs --example terrain --release
cargo run -p vrs --example screenshot --release -- terrain.ppm
cargo run -p vrs --example stress --release
cargo run -p vrs --example stress --release -- --headless
```

- `triangle` is the classic Vulkan triangle, drawn as a colored mesh lit head on.
- `cube` circles a cube of 8³ voxels checkered in two types. Greedy meshing makes each side
  one quad, and the colors still come from the voxels.
- `materials` circles rows of red lacquer and gold blocks, each rougher than the last, with
  blocks glowing in six colors between them on a glossy checkered floor. Space pauses the
  camera, and L turns on and off a point light inside each glowing block.
- `terrain` is rolling hills of greedy meshed chunks, ringed by coarser chunks of the next
  level of detail fading into a hazy sky, under a low sun casting shadows and lit by two
  cascades of light probes, tonemapped with ACES,
  to fly around: click to look with the mouse and Escape to let go, WASD to move, Space and
  Ctrl for up and down, Shift to go faster, L to drop a light and Backspace to take them
  away. The title also shows how many chunks were drawn, the rest culled behind hills.
- `screenshot` renders the terrain without a window and saves the frame as a PPM image.

The windowed examples' titles show frames a second and how long frames take on the GPU,
split into updating the light probes, tracing shadows and drawing, from
`Renderer::debug_gpu_timings`.

The examples share a greedy mesher for 64³ chunks in `examples/common`, to show how a
mesher's output becomes a chunk mesh, the Vulkan projection, and the title.

Debug builds run the stress test much slower, but check every Vulkan validation message.

`stress` checks the renderer the way a voxel game uses it and exits with a failure code if
anything's wrong:

- Voxel chunks behind a wall, axis aligned and packed, are culled with different cameras. Draw
  counts read back from the GPU must match the meshes in view, and with occlusion culling,
  at least every mesh the wall can't hide.
- A wall facing away from the camera must be skipped by backface culling.
- Chunks are remeshed every frame against a nearly full mesh pool, while the window is
  resized and minimized.
- The mesh pool is made to defragment and to grow, after which every moved mesh must still
  draw correctly.
- Each scenario's last frame is captured. Every chunk's face is two voxel types, and a pixel in
  each half must show its type's color, which catches chunks colored by anything but their
  voxels, drawn from missing data, or culled while visible.
- A renderer failing partway through being built must leak nothing.
- Lighting is checked against scenes with known answers: a floor under a white sky must come
  out at its color, no light may leak into a sealed box, both at full and half shadow
  resolution, bounced light must reach the inside of a box with a hole in its roof, glass
  must dim it, the probes must catch up with changes within a few frames, glowing voxels and
  point lights must light their surroundings, walls of fine and coarse chunks must shadow
  lights, a light inside a glowing block must light past it, spot lights must light only
  their cone, a mirror must reflect what's out of view, and coarser cascades must reach
  further without leaking or darkening coarse chunks.
- Debug builds fail on any validation message, with synchronization validation on.

Synchronization validation can't see which vertices an indirect draw reads, and the pixel
check only sees races the GPU loses, so a missing barrier before vertex input can still pass.

### In CI

`.github/workflows/ci.yml` checks formatting, runs clippy with warnings as errors, the tests
and a docs build, then runs the stress test headless on Mesa's lavapipe, a Vulkan driver that
runs on the CPU, with the validation layers installed so it checks synchronization too.

The compiled shaders are committed, and `build.rs` keeps them when `slangc` isn't installed,
since a fresh checkout's file times can make them look out of date.

## Debug hooks

For tests and tools, these stall the GPU unless they say otherwise:

- `Renderer::debug_frame_stats` reads back how many meshes the last frame drew.
- `Renderer::debug_capture_next_frame` and `debug_read_capture` copy a frame back.
- `Renderer::debug_mesh_pool_stats` reports the mesh pool's size and how often it repacked.
- `Renderer::debug_gpu_timings` reports how long each `GpuStage` of a recent frame took on the
  GPU, from timestamps, without stalling.
- `vulkan::debug::message_counts` counts validation errors and warnings.

## Layout

| Module | |
| --- | --- |
| `renderer` | `Renderer`, building it, and each frame's stages and command recording |
| `renderer::lighting` | How chunks are lit: `Lighting`, the sky, tonemapping, debug views, and for each of the shadows, the light probes and the lights, its settings, its GPU resources and passes, and how each frame drives them |
| `renderer::material` | `Material`s and the palette they're uploaded in |
| `renderer::mesh_pool` | The buffers every mesh shares, the `Renderer`'s methods adding and removing meshes, their GPU records, staging, range allocator and repacking |
| `renderer::culling` | GPU culling, and the depth pyramid occlusion culling tests against |
| `renderer::shader`, `renderer::timings` | What every shader shares, the frame's bindings and uniforms, and each frame's GPU timestamps |
| `renderer::debug` | Hooks for tests and tools: frame stats, GPU timings and captures |
| `mesh` | Mesh ids, every mesh's vertex formats, and octahedral normals |
| `mesh::chunk` | Chunk meshes and their packed voxel types |
| `vulkan` | Vulkan building blocks: GPUs and queues, memory, buffers, images, pipelines, descriptors, commands, sync, and the surface and swapchain frames are drawn into |
| `app` | The instance, and the window or headless images it draws to |

The shaders in `assets/shader` share `common.slang`, which includes one file of `common/` for
each concept: the data shared with the CPU, the bindings, voxels, materials, ambient
occlusion, rays, the probe volume, the lights, lighting, the sky, tonemapping and shading.

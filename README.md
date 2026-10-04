# vrs

A Vulkan 1.3 renderer for voxel games, which culls meshes on the GPU.

Every mesh lives in one shared vertex buffer and one index buffer. Each frame a compute shader
frustum culls them, then occlusion culls them against a depth pyramid in two phases, and draws
what's visible with indirect draws the GPU counts itself. Meshes can be created and destroyed
every frame, as chunks are remeshed, without stalling.

## Usage

```rust,no_run
use ash::vk;
use vrs::{
    App,
    renderer::{DrawList, Renderer},
    vertex::Vertex,
};

// A windowed app takes the window's handles, see `App::new`.
let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
let app = App::headless(app_info).unwrap();
let mut renderer = Renderer::new(&app, 800, 600).unwrap();

// Uploaded by the next frame, then drawn every frame it's listed.
let color = [1.0, 0.0, 0.0];
let triangle = renderer
    .create_mesh(
        // Counter-clockwise on screen, so it faces the camera.
        &[0, 2, 1],
        &[
            Vertex { pos: [0.0, -0.5, 0.5], col: color },
            Vertex { pos: [0.5, 0.5, 0.5], col: color },
            Vertex { pos: [-0.5, 0.5, 0.5], col: color },
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

- **Meshes** are created with `create_mesh`, which stages their data straight away, and drawn
  by listing their `MeshId`s in a `DrawList`. Ids aren't `Copy`, so `destroy_mesh` taking one
  means a destroyed mesh can't be drawn again. Its space is reused once no frame in flight
  can still draw it. Indices are relative to each mesh's first vertex, and meshes must already
  be in world space.
- **Winding**: front faces wind counter-clockwise on screen, backface culling skips the rest.
- **Cameras**: `DrawList::proj` maps into Vulkan's clip space, where y points down and depth
  runs from 0 to 1. Flip y in an OpenGL style projection, such as glam's `perspective_rh`,
  otherwise the image is mirrored and the wrong faces are culled.
- **The mesh pool** is defragmented when its free space is too scattered for a new mesh, and
  doubled when there isn't enough of it. Both copy every mesh into new buffers, stalling
  `create_mesh` until the copies finish, which is rare since the pool doubles each time.
  Without growth, `create_mesh` fails with `ERROR_OUT_OF_DEVICE_MEMORY` when it's full.
- **Per frame geometry**, such as debug lines, goes in a `DrawList`'s `indices` and `vertices`.
- **Settings**: a `RendererConfig` for `Renderer::with_config` sets the mesh pool's starting
  size, and starting values for growing it, vsync, occlusion culling and backface culling. At
  runtime `grow_mesh_pool`, `occlusion_culling`, `backface_culling` and `clear_color` are
  fields, vsync is `set_vsync`.
- **Headless**: `App::headless` renders into images of the renderer's own, with no window.

`App` and `Renderer` are `Send + Sync`, such as for Bevy resources.

## Requirements

- A GPU and driver supporting Vulkan 1.3. The renderer picks a discrete GPU first, then
  integrated, then anything else, that can also present to the window.
- `multiDrawIndirect` and `drawIndirectCount` are used when supported, with fallbacks when not.
- The compiled shaders in `assets/shader` are committed, so building only needs `slangc`, from
  the Vulkan SDK, after editing a `.slang` file.
- Debug builds turn on the Vulkan validation layer when it's installed.

## Examples

```sh
cargo run -p vrs --example triangle
cargo run -p vrs --example stress
cargo run -p vrs --example stress -- --headless
```

`triangle` is the classic Vulkan triangle. `stress` checks the renderer the way a voxel game
uses it and exits with a failure code if anything's wrong:

- Chunk meshes behind a wall are culled with different cameras and draw paths, including the
  fallbacks for devices without `multiDrawIndirect` or `drawIndirectCount`. Draw counts read
  back from the GPU must match the meshes in view, and with occlusion culling, at least every
  mesh the wall can't hide.
- A wall facing away from the camera must be skipped by backface culling.
- Chunks are remeshed every frame against a nearly full mesh pool, while the window is
  resized and minimized.
- The mesh pool is made to defragment and to grow, after which every moved mesh must still
  draw correctly.
- Each scenario's last frame is captured, and every chunk's center pixel must show the color
  it should, which catches meshes drawn from missing data or culled while visible.
- A renderer failing partway through being built must leak nothing.
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

For tests and tools, these stall the GPU:

- `Renderer::debug_frame_stats` reads back how many meshes the last frame drew.
- `Renderer::debug_capture_next_frame` and `debug_read_capture` copy a frame back.
- `Renderer::debug_limit_draw_paths` draws as if the device lacked features.
- `Renderer::debug_mesh_pool_stats` reports the mesh pool's size and how often it repacked.
- `debug::message_counts` counts validation errors and warnings.

## Layout

| Module | |
| --- | --- |
| `renderer` | `Renderer`, building it, each frame's stages and command recording |
| `mesh` | Mesh ids and the shared mesh pool, its staging, range allocator and repacking |
| `cull`, `hiz` | GPU culling, and the depth pyramid occlusion culling tests against |
| `app`, `adapter`, `surface`, `swapchain` | The instance, GPUs, and what frames are drawn into |
| `buffer`, `image`, `pipeline`, `descriptors`, `commands`, `sync` | Vulkan building blocks |

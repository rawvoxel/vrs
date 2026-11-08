# vrs

A bare-bones Vulkan 1.3 layer over [ash](https://docs.rs/ash), for GPU-driven compute and
mesh-shader rendering. It holds no engine logic: plain structs of handles and free functions
that each make one Vulkan object or record one command, with that command's rules checked in
debug builds. `ash` is re-exported in full, as `vrs::ash`.

## Usage

```rust,no_run
use vrs::{
    App,
    ash::vk,
    vulkan::{
        commands,
        compute::{AddressedBuffer, HostAccess, HostBuffer},
        gpu, sync,
    },
};

// The instance is Vulkan 1.3 at least, whatever the application info asks for.
let mut app = App::headless(vk::ApplicationInfo::default()).unwrap();
let adapter = app.adapters_by_preference()[0].clone();
assert!(gpu::missing_features(&app, &adapter).is_empty());
// The `App` destroys the device: everything made with it goes first.
let device = unsafe { gpu::create_device(&mut app, &adapter) }.unwrap();

// One pool, command buffer and fence for blocking one-shots.
let pool = commands::create_command_pool(&device.device, &device.queue_family).unwrap();
let command_buffer =
    commands::create_command_buffer(&device.device, pool, vk::CommandBufferLevel::PRIMARY)
        .unwrap();
let fence = sync::create_fence(&device.device).unwrap();

let buffer =
    AddressedBuffer::new(&device.adapter, &device.device, 1024, vk::BufferUsageFlags::empty())
        .unwrap();
let mut words = [0u32; 256];
unsafe {
    commands::submit_and_wait(&device, command_buffer, fence, |command_buffer| {
        commands::cmd_fill_buffer(&device.device, command_buffer, buffer.handle(), 0, 1024, 7);
    })
    .unwrap();
    commands::read_back(&device, command_buffer, fence, buffer.handle(), 0, &mut words).unwrap();
}
assert!(words.iter().all(|&word| word == 7));

unsafe {
    buffer.destroy(&device.device);
    device.device.destroy_fence(fence, None);
    device.device.destroy_command_pool(pool, None);
}
drop(app);
```

- **`App`** is the instance, headless or with a window's surface. It enumerates the
  `Adapter`s, each with its name, extensions and compute subgroup sizes, and destroys the
  devices made from it when it's dropped, or one earlier with `App::destroy_device`.
- **`gpu::VulkanDevice`** is a plain struct of public fields: the device's functions, its
  adapter, its one queue (graphics and compute, with timestamps) and family, the mesh-shader
  and pipeline-executables functions, the instance for memory budgets, and whether shaders may
  read clocks. `gpu::create_device` makes one on an adapter for an `App`;
  `gpu::create_shared_device` makes one from another Vulkan user's create info (wgpu-hal's),
  merging its own extensions and features in, and its caller destroys it. Both enable the same
  set; `gpu::missing_features` says what an adapter lacks.
- **Resources**: `compute::AddressedBuffer` (device memory shaders reach by address),
  `compute::HostBuffer` (mapped memory to upload or read back), `compute::ComputePipelines`,
  `compute::PipelineCache`, `compute::TimestampPool`, `gpu::MeshPipeline` and
  `gpu::RenderImage`. Each is destroyed by its `unsafe fn destroy(&self, device)`, once the
  device is done with it, by whoever owns it.
- **Recording**: `commands::cmd_*`, one function per command, taking the device's functions,
  the command buffer and handles. `sync::cmd_memory_barrier` takes `(stage, access)` pairs,
  `sync::cmd_pipeline_barrier` orders everything, and `commands::record_host_read_barrier`
  makes device writes visible to the host.
- **Submitting**: `commands::begin_command_buffer`, `commands::submit` (optionally with
  semaphores), `sync::wait_for_fence`, `sync::fence_signaled`, `commands::submit_and_wait`
  and `commands::read_back`. Waits are the driver's own, as long as the work takes, and a lost
  device returns `ERROR_DEVICE_LOST`. Nothing locks the queue: its owner submits from one
  place at a time.
- **Validation**: with `VRS_VALIDATION=1` in the environment and the Khronos layer installed,
  every `App` turns validation on. Its errors and warnings go through the `log` facade, and
  `validation::assert_no_new_errors` fails a test on any error but the known ones, listed in
  `validation::KNOWN_ERRORS`.

## Tests

The tests need a real GPU, so they're ignored by default (CI has none): run them with
`cargo test -p vrs -- --ignored`. They make devices, write and read buffers through the device,
clear a rendering and copy it back, and check that the recording functions refuse what their
commands forbid (in debug builds).

// The frame descriptor set's resources every shader shares. Each matches its `bindings` const.

[[vk::binding(0, 0)]] ConstantBuffer<Uniforms> ubo;
[[vk::binding(1, 0)]] StructuredBuffer<uint> vertexWords;
[[vk::binding(2, 0)]] StructuredBuffer<MeshInfo> meshes;
// Every chunk's voxel types, laid out as `voxel::VoxelTypes` packs them, then its occupancy,
// then its probe cells' words and voxel masks, see `lighting::gi::bake_probe_cells`.
[[vk::binding(3, 0)]] StructuredBuffer<uint> typeWords;
// Materials indexed by voxel type.
[[vk::binding(4, 0)]] StructuredBuffer<Material> palette;
// The mesh index of the chunk at each chunk coordinate around the camera, or `NO_CHUNK`. A
// table for each level, finest first, of `lighting.shadowTable.w` chunks per side, x-fastest.
[[vk::binding(5, 0)]] StructuredBuffer<uint> shadowTable;
// Each probe slot's probe, cascade by cascade, slots x-fastest by probe cell modulo the
// cascade's size.
#ifdef GI_WRITES
[[vk::binding(6, 0)]] RWStructuredBuffer<Probe> probes;
#else
[[vk::binding(6, 0)]] StructuredBuffer<Probe> probes;
#endif
// Each probe's irradiance and distance maps, octahedral with a border, in rows of
// `gi.volumeCounts.w` probes. Distance maps hold the mean distance and its square.
[[vk::binding(7, 0)]] Sampler2D<float4> probeIrradiance;
[[vk::binding(8, 0)]] Sampler2D<float4> probeDepth;
// A bit for each shadow table entry, set when its chunk was added, removed or remeshed since
// last frame.
[[vk::binding(16, 0)]] StructuredBuffer<uint> changedChunks;
// The frame's lights, see `Renderer::lights`.
[[vk::binding(17, 0)]] StructuredBuffer<Light> lights;

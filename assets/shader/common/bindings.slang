// The frame descriptor set's resources every shader shares. Each matches its `bindings` const.

[[vk::binding(0, 0)]] ConstantBuffer<Uniforms> uniforms;
[[vk::binding(1, 0)]] StructuredBuffer<uint> vertexWords;
[[vk::binding(2, 0)]] StructuredBuffer<MeshInfo> meshes;
// Every chunk's voxel types, laid out as `mesh::chunk::VoxelTypes` packs them, then its occupancy,
// then its probe cells' words and voxel masks, see `renderer::lighting::probes::bake_probe_cells`.
[[vk::binding(3, 0)]] StructuredBuffer<uint> chunkWords;
// Materials indexed by voxel type.
[[vk::binding(4, 0)]] StructuredBuffer<Material> palette;
// Where each of the voxel grid's chunks is, hashed: a slot's chunk's key, see `chunkKey`, then
// its mesh index, or `NO_CHUNK` for none, `LightingUniforms::chunkTableMask` plus one slots.
[[vk::binding(5, 0)]] StructuredBuffer<uint2> chunkTable;
// Each probe slot's probe, cascade by cascade, slots x-fastest by probe cell modulo the
// cascade's size.
#ifdef PROBE_WRITES
[[vk::binding(6, 0)]] RWStructuredBuffer<Probe> probes;
#else
[[vk::binding(6, 0)]] StructuredBuffer<Probe> probes;
#endif
// Each probe's irradiance and distance maps, octahedral with a border, in rows of
// `probeVolume.atlasColumns` probes. Distance maps hold the mean distance and its square.
[[vk::binding(7, 0)]] Sampler2D<float4> probeIrradiance;
[[vk::binding(8, 0)]] Sampler2D<float4> probeDistances;
// A bit for each probe slot, set when a chunk was added, removed or remeshed since last frame
// over the probe's cell or a neighboring one.
[[vk::binding(16, 0)]] StructuredBuffer<uint> changedProbes;
// The frame's lights, see `Renderer::lights`.
[[vk::binding(17, 0)]] StructuredBuffer<Light> lights;

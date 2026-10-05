//! Adding and removing meshes, which the mesh pool holds.

use super::Renderer;
use crate::{
    lighting::gi::bake_probe_cells,
    mesh::{Index, LitVertex, Mesh, MeshId, MeshKind, NewMesh, Vertex, octahedral},
    voxel::{ChunkMesh, ChunkVertices, unpack_position},
    vulkan::bytes::slice_as_words,
};
use ash::{prelude::VkResult, vk};
use glam::Vec3;

impl Renderer {
    /// Adds a mesh with colors of its own, in world space, uploaded by the next `submit`.
    /// Indices are relative to the mesh's first vertex, and an empty mesh draws nothing.
    ///
    /// See `add_mesh` for when this stalls or fails.
    pub fn create_mesh(&mut self, indices: &[Index], vertices: &[Vertex]) -> VkResult<MeshId> {
        let vertex_words: Vec<u32> = vertices.iter().flat_map(colored_words).collect();
        self.add_mesh(&NewMesh {
            kind: MeshKind::Colored,
            vertex_words: &vertex_words,
            indices,
            type_words: &[],
            occupancy: &[],
            probes: &[],
            origin: Vec3::ZERO,
            scale: 1.0,
            voxels_per_edge: 0,
            bounds: bounds(vertices.iter().map(|vertex| Vec3::from(vertex.pos))),
        })
    }

    /// Adds a mesh lit like chunks, by the sun, the lights, and the probes or the sky and
    /// ground, as a rough surface of its vertices' colors, uploaded by the next `submit`.
    /// For what isn't voxels but sits among them, such as leaves and grass. It casts no
    /// shadows and isn't seen by the probes, which trace voxels, and it's backface culled like
    /// anything else, so a thin card seen from both sides needs both windings.
    ///
    /// See `add_mesh` for when this stalls or fails.
    pub fn create_lit_mesh(
        &mut self,
        indices: &[Index],
        vertices: &[LitVertex],
    ) -> VkResult<MeshId> {
        let vertex_words: Vec<u32> = vertices.iter().flat_map(lit_words).collect();
        self.add_mesh(&NewMesh {
            kind: MeshKind::Lit,
            vertex_words: &vertex_words,
            indices,
            type_words: &[],
            occupancy: &[],
            probes: &[],
            origin: Vec3::ZERO,
            scale: 1.0,
            voxels_per_edge: 0,
            bounds: bounds(vertices.iter().map(|vertex| Vec3::from(vertex.pos))),
        })
    }

    /// Adds a chunk, colored by its voxels through the palette, uploaded by the next `submit`.
    /// Its vertices are taken as they are, they're already in the GPU's format.
    ///
    /// See `add_mesh` for when this stalls or fails.
    pub fn create_chunk_mesh(&mut self, chunk: &ChunkMesh) -> VkResult<MeshId> {
        let edge = chunk.types.voxels_per_edge();
        let (kind, vertex_words) = match chunk.vertices {
            ChunkVertices::Greedy(vertices) => (MeshKind::Greedy, slice_as_words(vertices)),
            ChunkVertices::Packed(vertices) => (MeshKind::Packed, slice_as_words(vertices)),
        };
        let positions = vertex_words
            .chunks_exact(kind.words_per_vertex() as usize)
            .map(|words| {
                chunk.origin + unpack_position(words[0], words[1], edge) * chunk.voxel_size
            });
        self.add_mesh(&NewMesh {
            kind,
            vertex_words,
            indices: chunk.indices,
            type_words: chunk.types.words(),
            occupancy: &chunk.types.occupancy(),
            probes: &bake_probe_cells(chunk.types),
            origin: chunk.origin,
            scale: chunk.voxel_size,
            voxels_per_edge: edge,
            bounds: bounds(positions),
        })
    }

    /// Its space is reused once no frame in flight can still draw it.
    #[inline]
    pub fn destroy_mesh(&mut self, id: MeshId) {
        let frame_count = self.in_flight_frames.len();
        let last_frame = (self.current_frame + frame_count - 1) % frame_count;
        self.mesh_pool.destroy(id, last_frame);
    }

    /// Where the mesh lives in the shared buffers, and its bounds.
    ///
    /// # Panics
    /// If `id` came from another renderer.
    #[inline]
    pub fn mesh(&self, id: &MeshId) -> &Mesh {
        self.mesh_pool.get(id)
    }

    /// Stages `new` right away, which first waits for the device to finish the frame that
    /// last used the next frame's staging buffer. That's normally long done.
    ///
    /// When the pool's out of room it waits for the device to go idle, freeing destroyed
    /// meshes' space, then defragments, or grows when `grow_mesh_pool` allows. Those stall,
    /// but are rare. Fails with `ERROR_OUT_OF_DEVICE_MEMORY` when none of it makes room.
    fn add_mesh(&mut self, new: &NewMesh) -> VkResult<MeshId> {
        let frame = self.current_frame;
        self.wait_for_frame(frame)?;
        let out_of_room = |result: &VkResult<MeshId>| {
            matches!(result, Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY))
        };

        let mut created = self.stage_mesh(frame, new);
        if !out_of_room(&created) {
            return created;
        }
        // Frames in flight may still hold destroyed meshes' space.
        if self.mesh_pool.has_pending_releases() {
            unsafe { self.render_device.device_wait_idle()? };
            self.mesh_pool.release_all();
            created = self.stage_mesh(frame, new);
            if !out_of_room(&created) {
                return created;
            }
        }

        let Some(capacity) = self.mesh_pool.plan_repack(new, self.grow_mesh_pool) else {
            return created;
        };
        unsafe {
            self.render_device.device_wait_idle()?;
            self.mesh_pool.repack(
                &self.adapter,
                &self.render_device,
                self.graphics_queue.handle,
                self.init_cmd_buffer,
                self.init_cmd_fence,
                capacity,
            )?;
        }
        self.stage_mesh(frame, new)
    }

    #[inline]
    fn stage_mesh(&mut self, frame: usize, new: &NewMesh) -> VkResult<MeshId> {
        // SAFETY: `add_mesh` waited for `frame`, which readies its staging buffer.
        unsafe {
            self.mesh_pool
                .create(&self.adapter, &self.render_device, frame, new)
        }
    }
}

/// A colored vertex as the mesh shader pulls it: its position's floats, then its color as
/// linear 8 bit RGBA.
fn colored_words(vertex: &Vertex) -> [u32; 4] {
    let [r, g, b] = vertex
        .col
        .map(|channel| (channel.clamp(0.0, 1.0) * 255.0).round() as u32);
    let [x, y, z] = vertex.pos.map(f32::to_bits);
    [x, y, z, r | (g << 8) | (b << 16) | (255 << 24)]
}

/// A lit vertex's words: its position, color, and octahedral normal, as `mesh.slang` reads
/// them.
fn lit_words(vertex: &LitVertex) -> [u32; 5] {
    let [x, y, z, color] = colored_words(&Vertex {
        pos: vertex.pos,
        col: vertex.col,
    });
    let [u, v] = octahedral::encode(Vec3::from(vertex.normal).normalize_or(Vec3::Y));
    [x, y, z, color, u as u32 | ((v as u32) << 16)]
}

/// The box around `positions`, zero sized at the origin when there are none.
fn bounds(positions: impl Iterator<Item = Vec3>) -> (Vec3, Vec3) {
    positions
        .fold(None, |bounds, position| match bounds {
            None => Some((position, position)),
            Some((min, max)) => Some((position.min(min), position.max(max))),
        })
        .unwrap_or((Vec3::ZERO, Vec3::ZERO))
}

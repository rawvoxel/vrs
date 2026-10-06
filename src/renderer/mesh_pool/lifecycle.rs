//! The `Renderer`'s methods adding meshes to the pool and removing them.

use super::{MeshKind, MeshRecord, NewMesh};
use crate::{
    mesh::{
        Index, MeshId, Vertex,
        chunk::{ChunkMesh, ChunkVertices, boundary_faces},
        unpack_position,
    },
    renderer::{Renderer, lighting::probes::bake_probe_cells},
    vulkan::bytes::slice_as_words,
};
use ash::{prelude::VkResult, vk};
use glam::Vec3;

impl Renderer {
    /// Adds a mesh with colors of its own, in world space, uploaded by the next `submit`, lit
    /// like chunks, by the sun, the lights, and the probes or the sky and ground, as a rough
    /// surface. For what isn't voxels but sits among them, such as leaves and grass. It casts
    /// no shadows and isn't seen by the probes, which trace voxels, and it's backface culled
    /// like anything else, so a thin card seen from both sides needs both windings. Indices are
    /// relative to the mesh's first vertex, and an empty mesh draws nothing.
    ///
    /// See `add_mesh` for when this stalls or fails.
    pub fn create_colored_mesh(
        &mut self,
        indices: &[Index],
        vertices: &[Vertex],
    ) -> VkResult<MeshId> {
        let (vertex_words, bounds) = colored_words(vertices);
        self.add_colored_mesh(indices, &vertex_words, bounds)
    }

    /// [`Self::create_colored_mesh`] for a mesh `prepared` already, on any thread.
    pub fn add_prepared_colored_mesh(
        &mut self,
        prepared: &PreparedColoredMesh,
    ) -> VkResult<MeshId> {
        self.add_colored_mesh(&prepared.indices, &prepared.vertex_words, prepared.bounds)
    }

    fn add_colored_mesh(
        &mut self,
        indices: &[Index],
        vertex_words: &[u32],
        bounds: (Vec3, Vec3),
    ) -> VkResult<MeshId> {
        self.add_mesh(&NewMesh {
            kind: MeshKind::Colored,
            vertex_words,
            indices,
            type_words: &[],
            occupancy: &[],
            probes: &[],
            boundary_faces: &[],
            origin: Vec3::ZERO,
            scale: 1.0,
            voxels_per_edge: 0,
            bounds,
        })
    }

    /// Adds a chunk, colored by its voxels through the palette, uploaded by the next `submit`.
    /// Its vertices are taken as they are, they're already in the GPU's format.
    ///
    /// See `add_mesh` for when this stalls or fails.
    pub fn create_chunk_mesh(&mut self, chunk: &ChunkMesh) -> VkResult<MeshId> {
        self.add_prepared_chunk(chunk, &PreparedChunk::new(chunk))
    }

    /// [`Self::create_chunk_mesh`] for a chunk whose `prepared` was worked out already, on
    /// any thread.
    ///
    /// # Panics
    /// If `prepared` wasn't made from `chunk`'s own vertices and types.
    pub fn add_prepared_chunk(
        &mut self,
        chunk: &ChunkMesh,
        prepared: &PreparedChunk,
    ) -> VkResult<MeshId> {
        let (kind, vertex_words) = match chunk.vertices {
            ChunkVertices::AxisAligned(vertices) => {
                (MeshKind::AxisAligned, slice_as_words(vertices))
            }
            ChunkVertices::Packed(vertices) => (MeshKind::Packed, slice_as_words(vertices)),
        };
        assert_eq!(
            prepared.vertices,
            chunk.vertices.len(),
            "a chunk prepared from other vertices"
        );
        self.add_mesh(&NewMesh {
            kind,
            vertex_words,
            indices: chunk.indices,
            type_words: chunk.types.words(),
            occupancy: &prepared.occupancy,
            probes: &prepared.probes,
            boundary_faces: &prepared.boundary_faces,
            origin: chunk.origin,
            scale: chunk.voxel_size,
            voxels_per_edge: chunk.types.voxels_per_edge(),
            bounds: prepared.bounds,
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
    pub fn mesh(&self, id: &MeshId) -> &MeshRecord {
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
            unsafe { self.device.device_wait_idle()? };
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
            self.device.device_wait_idle()?;
            self.mesh_pool.repack(
                &self.adapter,
                &self.device,
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
                .create(&self.adapter, &self.device, frame, new)
        }
    }
}

/// What the renderer works out from a chunk before uploading it: its occupancy, its light
/// probes' cells, the voxels along its sides with faces out of it, and its bounds. Pure, so a
/// host can make it on any thread and add the chunk with `Renderer::add_prepared_chunk`.
#[derive(Clone, Debug)]
pub struct PreparedChunk {
    occupancy: Vec<u32>,
    probes: Vec<u32>,
    boundary_faces: Vec<u32>,
    bounds: (Vec3, Vec3),
    /// The chunk's vertex count, to catch one prepared from other vertices.
    vertices: usize,
}

impl PreparedChunk {
    pub fn new(chunk: &ChunkMesh) -> Self {
        let edge = chunk.types.voxels_per_edge();
        // Only axis aligned faces lie exactly in a chunk's sides, to tell which voxels have
        // faces out of it.
        let (kind, vertex_words, boundary_faces) = match chunk.vertices {
            ChunkVertices::AxisAligned(vertices) => (
                MeshKind::AxisAligned,
                slice_as_words(vertices),
                boundary_faces(edge, vertices, chunk.indices),
            ),
            ChunkVertices::Packed(vertices) => {
                (MeshKind::Packed, slice_as_words(vertices), Vec::new())
            }
        };
        let positions = vertex_words
            .chunks_exact(kind.words_per_vertex() as usize)
            .map(|words| {
                chunk.origin + unpack_position(words[0], words[1], edge) * chunk.voxel_size
            });
        Self {
            occupancy: chunk.types.occupancy(),
            probes: bake_probe_cells(chunk.types),
            boundary_faces,
            bounds: bounds(positions),
            vertices: chunk.vertices.len(),
        }
    }
}

/// A mesh with colors of its own as the renderer uploads it: its indices, its vertices in the
/// GPU's words, and its bounds. Pure, so a host can make it on any thread and add it with
/// `Renderer::add_prepared_colored_mesh`.
#[derive(Clone, Debug)]
pub struct PreparedColoredMesh {
    indices: Vec<Index>,
    vertex_words: Vec<u32>,
    bounds: (Vec3, Vec3),
}

impl PreparedColoredMesh {
    pub fn new(indices: Vec<Index>, vertices: &[Vertex]) -> Self {
        let (vertex_words, bounds) = colored_words(vertices);
        Self {
            indices,
            vertex_words,
            bounds,
        }
    }

    /// Whether it draws nothing.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// A colored mesh's vertices as the GPU's words, and their bounds.
fn colored_words(vertices: &[Vertex]) -> (Vec<u32>, (Vec3, Vec3)) {
    (
        vertices.iter().flat_map(Vertex::words).collect(),
        bounds(vertices.iter().map(|vertex| Vec3::from(vertex.position))),
    )
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

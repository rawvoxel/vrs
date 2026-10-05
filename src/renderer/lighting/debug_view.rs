//! Seeing how chunks are lit.

/// What chunk meshes show, for seeing how they're lit, see `Lighting::debug_view`. Matches the
/// `DEBUG_` constants in `mesh.slang`, but for `Probes`, which shades chunks as usual.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DebugView {
    /// Their colors, lit, as usual.
    #[default]
    Off,
    /// All the light reaching them, as if they were white.
    Light,
    /// Only the sun's and the lights'.
    Direct,
    /// Only the light from everywhere else: the probes', or the sky and ground's.
    Indirect,
    /// How much voxel ambient occlusion leaves of the indirect light.
    AmbientOcclusion,
    /// As usual, with the light probes drawn as small cubes, each face colored with the
    /// light the probe holds from its direction.
    Probes,
}

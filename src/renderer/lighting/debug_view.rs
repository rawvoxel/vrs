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
    /// How the trace pass found each pixel's surface, in green where it found the face the
    /// pixel shows and in red where it didn't, bright where the sun reaches it and dark where
    /// it doesn't. Where the sun's ray gave up rather than reaching the sky or a voxel, in
    /// yellow past the trace distance, magenta out of voxel steps and cyan out of chunk steps.
    /// Without the trace pass, black.
    Trace,
    /// The level of the voxel grid the trace pass found each pixel's face in: green for the
    /// finest, then blue, yellow and magenta. Where it found none, red where its search met no
    /// voxel and white where it started in one. Without the trace pass, black.
    Levels,
    /// Which cascade of light probes lights each pixel, blended where they blend: green for
    /// the finest, then blue, yellow and magenta. Red where a cascade covers it but none of its
    /// probes count, so it gets the sky's and ground's light instead, and gray past every
    /// cascade.
    Cascades,
}

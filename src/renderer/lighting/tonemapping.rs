//! Bringing shaded colors into the display's range.

/// How `Lighting::exposure`d colors brighter than the display shows are brought into its
/// range, see `Lighting::tonemapping`. Colored meshes keep their own colors.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum Tonemapping {
    /// Clipped, as the display does: colors are drawn as they're lit.
    #[default]
    None,
    /// Each color's brightness `l` becomes `l / (1 + l)`, keeping its hue. Flat, but never
    /// clips.
    Reinhard,
    /// Narkowicz's fit of the ACES filmic curve, with more contrast, and highlights washing
    /// out to white.
    Aces,
}

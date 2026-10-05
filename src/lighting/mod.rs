//! Lighting's GPU resources and passes: the light probes, the point and spot lights, and the
//! shadow pass. How they're set up each frame is `renderer::Lighting`'s.

pub(crate) mod gi;
pub(crate) mod lights;
pub(crate) mod shadows;

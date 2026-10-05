// Shared by every shader but the culling ones: the frame's bindings, decoding voxels, tracing
// rays through them, and lighting from the sun, the sky, the light probes and the lights,
// with ambient occlusion and glossy reflections.
//
// `gi.slang` defines `GI_WRITES` first, for the bindings only its compute shaders write.

#include "common/layout.slang"
#include "common/bindings.slang"
#include "common/math.slang"
#include "common/voxels.slang"
#include "common/materials.slang"
#include "common/occlusion.slang"
#include "common/rays.slang"
#include "common/probe_volume.slang"
#include "common/lights.slang"
#include "common/lighting.slang"
#include "common/sky.slang"
#include "common/tonemapping.slang"
#include "common/shading.slang"

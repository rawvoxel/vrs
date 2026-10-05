// Bringing shaded colors into the display's range.

// `renderer::Tonemapping`'s.
static const uint TONEMAP_REINHARD = 1;
static const uint TONEMAP_ACES = 2;

// A shaded color as it's displayed: exposed, then tonemapped.
float3 toDisplay(float3 color) {
    color *= ubo.lighting.options.z;
    switch (uint(ubo.lighting.options.w)) {
    case TONEMAP_REINHARD:
        return color / (1.0 + dot(color, LUMINANCE));
    case TONEMAP_ACES:
        return saturate(color * (2.51 * color + 0.03) / (color * (2.43 * color + 0.59) + 0.14));
    default:
        return color;
    }
}

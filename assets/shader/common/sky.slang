// The sky seen behind everything, and the haze fading distant surfaces into it.

bool hasSky() {
    return ubo.lighting.sky[2].w != 0.0;
}

// What's seen of the sky along `direction`: the horizon blending into the zenith above, and
// into the ground below, with the sun's disc unless `withSun` is false.
float3 skyRadiance(float3 direction, bool withSun) {
    float3 zenith = ubo.lighting.sky[0].rgb;
    float3 horizon = ubo.lighting.sky[1].rgb;
    float up = direction.y;
    float3 color = up >= 0.0
        ? lerp(horizon, zenith, sqrt(up))
        : lerp(horizon, ubo.lighting.groundColor.rgb, saturate(-up * 4.0));
    float sunCosine = ubo.lighting.sky[2].x;
    if (withSun && sunCosine < 1.0 && up > 0.0) {
        // A soft edge, a tenth of its radius wide, and a faint glow around it.
        float toSun = dot(direction, ubo.lighting.sunDirection.xyz);
        float edge = (1.0 - sunCosine) * 0.2;
        float disc = smoothstep(sunCosine - edge, sunCosine + edge, toSun);
        float glow = pow(saturate(toSun), 256.0) * 0.5;
        color += ubo.lighting.sunColor.rgb * (disc * ubo.lighting.sky[2].y + glow);
    }
    return color;
}

// `color`, seen from `fromCamera` away, fading into the sky's color behind it with distance.
float3 hazed(float3 color, float3 fromCamera) {
    float hazeDistance = ubo.lighting.sky[2].z;
    if (!hasSky() || hazeDistance <= 0.0) {
        return color;
    }
    float distance = length(fromCamera);
    float haze = 1.0 - exp2(-distance / hazeDistance);
    return lerp(color, skyRadiance(fromCamera / max(distance, 1e-6), false), haze);
}

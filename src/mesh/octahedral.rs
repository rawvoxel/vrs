//! Unit normals packed in one word, as vertices keep them.

use glam::{Vec2, Vec3};

/// A unit normal folded onto the unit square, +Y at the center, 16 bits per axis. A
/// zero-length normal packs as +Y.
pub fn encode(normal: Vec3) -> [u16; 2] {
    let normal = normal.normalize_or(Vec3::Y);
    let mut folded =
        Vec2::new(normal.x, normal.z) / (normal.x.abs() + normal.y.abs() + normal.z.abs());
    if normal.y < 0.0 {
        let sign = Vec2::new(
            if folded.x >= 0.0 { 1.0 } else { -1.0 },
            if folded.y >= 0.0 { 1.0 } else { -1.0 },
        );
        folded = (Vec2::ONE - Vec2::new(folded.y.abs(), folded.x.abs())) * sign;
    }
    let unit = (folded * 0.5 + Vec2::splat(0.5)) * 65535.0;
    [unit.x.round() as u16, unit.y.round() as u16]
}

/// Undoes `encode` on a packed normal word. Matches `octahedralDecode` in `mesh.slang`.
pub fn decode(word: u32) -> Vec3 {
    let encoded = Vec2::new((word & 0xffff) as f32, (word >> 16) as f32) / 65535.0;
    let folded = encoded * 2.0 - 1.0;
    let mut normal = Vec3::new(folded.x, 1.0 - folded.x.abs() - folded.y.abs(), folded.y);
    let below = (-normal.y).max(0.0);
    normal.x += if normal.x >= 0.0 { -below } else { below };
    normal.z += if normal.z >= 0.0 { -below } else { below };
    normal.normalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(normal: Vec3) -> Vec3 {
        let [u, v] = encode(normal);
        decode(u as u32 | ((v as u32) << 16))
    }

    #[test]
    fn axes_round_trip() {
        for axis in [
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
        ] {
            let decoded = round_trip(axis);
            assert!(decoded.distance(axis) < 1e-4, "{axis}: {decoded}");
        }
    }

    #[test]
    fn diagonals_round_trip() {
        let diagonal = Vec3::new(1.0, -2.0, 3.0).normalize();
        assert!(round_trip(diagonal).distance(diagonal) < 1e-3);
    }
}

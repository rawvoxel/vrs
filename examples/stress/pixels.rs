//! Comparing a captured frame with the colors the scene should show.
//!
//! Each chunk is two flat halves of different voxel types, so a pixel in each half shows that
//! type's color, the wall's when the wall is in front, or the clear color when the cell is
//! empty. A chunk colored by its face rather than its voxels, drawn from stale or missing
//! data, or culled while visible, shows the wrong colors.

use crate::scene::{CHUNK_SIZE, CHUNK_Z, EDGE, GRID, Scene, VOXEL_SIZE, WALL_COLOR, chunk_min};
use glam::{Mat4, Vec2};
use vrs::renderer::Capture;

/// Allowed difference per channel, for rounding.
const TOLERANCE: u8 = 2;
/// Pixels this close to the wall's edge could show either side, so they aren't checked.
const EDGE_MARGIN: f32 = 2.0;

/// How many pixels were checked, or what was wrong with them.
pub fn check(scene: &Scene, capture: &Capture, view_proj: Mat4) -> Result<usize, String> {
    let size = Vec2::new(capture.width as f32, capture.height as f32);
    // Vulkan's NDC y points down, like the image's rows.
    let to_pixel = |ndc: Vec2| (ndc * 0.5 + 0.5) * size;
    let wall = scene
        .wall_rect(view_proj)
        .map(|(lo, hi)| (to_pixel(lo), to_pixel(hi)));
    let [r, g, b, _] = scene.renderer.clear_color;
    let clear = [r, g, b];

    let mut checked = 0;
    let mut wrong = Vec::new();
    for cell in 0..GRID * GRID {
        // One voxel into each half, away from the chunk's edges and where the halves meet,
        // or the middle of an empty cell.
        let samples = match scene.colors[cell as usize] {
            Some([low, high]) => vec![
                (Vec2::new(1.0, EDGE as f32 / 2.0) * VOXEL_SIZE, low),
                (
                    Vec2::new(EDGE as f32 - 1.0, EDGE as f32 / 2.0) * VOXEL_SIZE,
                    high,
                ),
            ],
            None => vec![(Vec2::splat(CHUNK_SIZE / 2.0), clear)],
        };
        for (offset, color) in samples {
            let world = (chunk_min(cell) + offset).extend(CHUNK_Z);
            let pixel = to_pixel(view_proj.project_point3(world).truncate());
            if pixel.cmplt(Vec2::ZERO).any() || pixel.cmpge(size).any() {
                continue;
            }
            let expected = match wall {
                Some((lo, hi)) if near_edge(pixel, lo, hi) => continue,
                Some((lo, hi)) if pixel.cmpge(lo).all() && pixel.cmplt(hi).all() => WALL_COLOR,
                _ => color,
            };
            let Some(actual) = capture.rgba8(pixel.x as u32, pixel.y as u32) else {
                return Err(format!("can't read {:?} pixels", capture.format));
            };
            let expected = encode(expected, capture.is_srgb());
            checked += 1;
            if (0..3).any(|i| actual[i].abs_diff(expected[i]) > TOLERANCE) {
                wrong.push(format!(
                    "cell {cell} at ({}, {}) is {:?}, expected {expected:?}",
                    pixel.x as u32,
                    pixel.y as u32,
                    &actual[..3]
                ));
            }
        }
    }

    match wrong.is_empty() {
        true => Ok(checked),
        false => Err(format!(
            "{} of {checked} pixels are wrong, such as {}",
            wrong.len(),
            wrong[..wrong.len().min(3)].join("; ")
        )),
    }
}

fn near_edge(pixel: Vec2, lo: Vec2, hi: Vec2) -> bool {
    let inside_margin = |lo: f32, hi: f32, p: f32| p > lo - EDGE_MARGIN && p < hi + EDGE_MARGIN;
    let near = |edge: f32, p: f32| (p - edge).abs() < EDGE_MARGIN;
    (inside_margin(lo.y, hi.y, pixel.y) && (near(lo.x, pixel.x) || near(hi.x, pixel.x)))
        || (inside_margin(lo.x, hi.x, pixel.x) && (near(lo.y, pixel.y) || near(hi.y, pixel.y)))
}

/// A linear color as stored bytes, sRGB encoded for `_SRGB` formats.
fn encode(color: [f32; 3], srgb: bool) -> [u8; 3] {
    color.map(|channel| {
        let stored = match srgb {
            true if channel <= 0.003_130_8 => channel * 12.92,
            true => 1.055 * channel.powf(1.0 / 2.4) - 0.055,
            false => channel,
        };
        (stored.clamp(0.0, 1.0) * 255.0).round() as u8
    })
}

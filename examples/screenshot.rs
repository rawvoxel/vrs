//! Renders the terrain example's world without a window and saves the frame as an image.
//!
//! Run with `cargo run -p vrs --example screenshot --release -- [path]`, which writes
//! `screenshot.ppm` by default. PPM is the simplest image format there is, so it needs no
//! image library. Most image editors and viewers open it, such as GIMP or IrfanView.

#[path = "common/greedy.rs"]
mod greedy;
#[path = "common/terrain.rs"]
mod terrain;

use anyhow::{Context, Result};
use ash::vk;
use std::{fs, path::PathBuf};
use vrs::{
    App,
    mesh::MeshId,
    renderer::{Capture, DrawList, Renderer},
};

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

/// The frame as a binary PPM: a short text header, then each pixel's red, green and blue
/// bytes, sRGB encoded as viewers expect.
fn encode_ppm(capture: &Capture) -> Result<Vec<u8>> {
    let mut ppm = format!("P6\n{} {}\n255\n", capture.width, capture.height).into_bytes();
    for y in 0..capture.height {
        for x in 0..capture.width {
            let [r, g, b, _] = capture
                .rgba8(x, y)
                .with_context(|| format!("Can't read {:?} pixels.", capture.format))?;
            let rgb = [r, g, b];
            match capture.is_srgb() {
                true => ppm.extend(rgb),
                false => ppm.extend(rgb.map(srgb_encode)),
            }
        }
    }
    Ok(ppm)
}

/// A linear byte as an sRGB encoded one.
fn srgb_encode(linear: u8) -> u8 {
    let linear = linear as f32 / 255.0;
    let encoded = match linear <= 0.003_130_8 {
        true => linear * 12.92,
        false => 1.055 * linear.powf(1.0 / 2.4) - 0.055,
    };
    (encoded * 255.0).round() as u8
}

fn main() -> Result<()> {
    let path = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "screenshot.ppm".to_string()),
    );
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"vrs screenshot")
        .api_version(vk::API_VERSION_1_3);
    let app = App::headless(app_info)?;
    let mut renderer = Renderer::with_config(&app, WIDTH, HEIGHT, terrain::config())?;
    let chunks = terrain::create(&mut renderer)?;

    let (view, proj) = terrain::view_proj(
        terrain::START_POSITION,
        terrain::forward(terrain::START_YAW, terrain::START_PITCH),
        WIDTH as f32 / HEIGHT as f32,
    );
    let meshes: Vec<&MeshId> = chunks.iter().collect();
    let draws = DrawList {
        meshes: &meshes,
        view,
        proj,
        ..Default::default()
    };
    // The first frame uploads the chunks and draws them, the capture is of the second.
    renderer.submit(WIDTH, HEIGHT, &draws)?;
    if !renderer.debug_capture_next_frame() {
        anyhow::bail!("This device's frames can't be copied back.");
    }
    renderer.submit(WIDTH, HEIGHT, &draws)?;
    let capture = renderer
        .debug_read_capture()?
        .context("The frame wasn't captured.")?;

    fs::write(&path, encode_ppm(&capture)?)
        .with_context(|| format!("Couldn't write {}.", path.display()))?;
    println!("Saved {}x{} to {}.", WIDTH, HEIGHT, path.display());

    // The renderer must be dropped before the app.
    drop(renderer);
    drop(app);
    Ok(())
}

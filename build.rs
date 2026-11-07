use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    // Included by `mesh.slang`, `probe_update.slang`, `trace_pass.slang`, `probe_overlay.slang` and
    // `sky_pass.slang`, and including each of `common/`.
    println!("cargo:rerun-if-changed=assets/shader/common.slang");
    println!("cargo:rerun-if-changed=assets/shader/common");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"));
    compile_slang_spv(&out_dir, "cull.slang", "cull.spv", &["cullMeshes"]);
    compile_slang_spv(
        &out_dir,
        "depth_pyramid.slang",
        "depth_pyramid.spv",
        &["reduceDepth"],
    );
    compile_slang_spv(
        &out_dir,
        "mesh.slang",
        "mesh.spv",
        &["meshVert", "meshFrag"],
    );
    compile_slang_spv(
        &out_dir,
        "trace_pass.slang",
        "trace_pass.spv",
        &["cullLights", "traceSurfaces"],
    );
    compile_slang_spv(
        &out_dir,
        "probe_overlay.slang",
        "probe_overlay.spv",
        &["probeVert", "probeFrag"],
    );
    compile_slang_spv(
        &out_dir,
        "sky_pass.slang",
        "sky_pass.spv",
        &["skyVert", "skyFrag"],
    );
    compile_slang_spv(
        &out_dir,
        "probe_update.slang",
        "probe_update.spv",
        &[
            "classifyProbes",
            "planRays",
            "traceProbeRays",
            "updateIrradiance",
            "updateDistances",
        ],
    );
}

/// Compiles `slang_name` from `assets/shader` into `out_dir`'s `spv_name`, which the renderer
/// includes from there. Cargo only reruns this when a shader source changes, or on a fresh build.
#[rustfmt::skip]
fn compile_slang_spv(out_dir: &Path, slang_name: &str, spv_name: &str, entries: &[&str]) {
    let slang_path = Path::new("assets/shader").join(slang_name);
    let spv_path = out_dir.join(spv_name);
    println!("cargo:rerun-if-changed={}", slang_path.display());

    // Run `slangc` with every entry point
    let mut command = Command::new("slangc");
    command.args([
        slang_path.to_str().unwrap(),
        "-target", "spirv", "-profile", "spirv_1_4",
        "-emit-spirv-directly", "-fvk-use-entrypoint-name",
    ]);
    for entry in entries {
        command.args(["-entry", entry]);
    }
    let status = command
        .args(["-o", spv_path.to_str().unwrap()])
        .status()
        .unwrap_or_else(|error| panic!("Failed to run slangc, which {spv_name} needs: {error}"));
    if !status.success() {
        panic!("slangc failed to compile {slang_name}.");
    }
}

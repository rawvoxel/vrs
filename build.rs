use std::{path::Path, process::Command};

fn main() {
    // Included by `mesh.slang` and `gi.slang`.
    println!("cargo:rerun-if-changed=assets/shader/common.slang");
    compile_slang_spv("shader.slang", "slang.spv", &["vertMain", "fragMain"]);
    compile_slang_spv("cull.slang", "cull.spv", &["cullMain"]);
    compile_slang_spv("hiz.slang", "hiz.spv", &["reduceMain"]);
    compile_slang_spv("mesh.slang", "mesh.spv", &["meshVert", "meshFrag"]);
    compile_slang_spv(
        "gi.slang",
        "gi.spv",
        &[
            "classifyProbes",
            "planRays",
            "traceProbeRays",
            "updateIrradiance",
            "updateDepth",
        ],
    );
}

/// Cargo only reruns this when a shader source changes, or on a fresh build.
#[rustfmt::skip]
fn compile_slang_spv(slang_name: &str, spv_name: &str, entries: &[&str]) {
    let shader_dir = Path::new("assets/shader");
    let slang_path = shader_dir.join(slang_name);
    let spv_path = shader_dir.join(spv_name);
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
    let status = match command.args(["-o", spv_path.to_str().unwrap()]).status() {
        Ok(status) => status,
        // The compiled shader is committed, so without `slangc`, such as on CI, keep it.
        Err(err) if spv_path.exists() => {
            println!("cargo:warning=Using the committed {spv_name}, slangc couldn't run: {err}");
            return;
        }
        Err(err) => panic!("Failed to run slangc, which {spv_name} needs: {err}"),
    };

    if !status.success() {
        panic!("Shader compilation failed.");
    }
}

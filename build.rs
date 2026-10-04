use std::{fs, path::Path, process::Command, time::SystemTime};

fn main() {
    compile_slang_spv("shader.slang", "slang.spv", &["vertMain", "fragMain"]);
    compile_slang_spv("cull.slang", "cull.spv", &["cullMain"]);
    compile_slang_spv("hiz.slang", "hiz.spv", &["reduceMain"]);
}

#[rustfmt::skip]
fn compile_slang_spv(slang_name: &str, spv_name: &str, entries: &[&str]) {
    let shader_dir = Path::new("assets/shader");
    fs::create_dir_all(shader_dir).unwrap();
    let shader_dir = shader_dir.canonicalize().expect("Failed to canonicalize shader directory!");

    let slang_path = shader_dir.join(slang_name);
    let spv_path = shader_dir.join(spv_name);

    // Re-run if the shader source changes
    println!("cargo:rerun-if-changed={}", slang_path.display());

    let slang_meta = fs::metadata(&slang_path);
    let spv_meta = fs::metadata(&spv_path);

    let needs_compilation = match spv_meta {
        Ok(spv_meta) => {
            let slang_modified_time = slang_meta.and_then(|m| m.modified()).unwrap_or(SystemTime::UNIX_EPOCH);
            let spv_modified_time = spv_meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
            slang_modified_time > spv_modified_time
        }
        Err(_) => true,
    };

    if !needs_compilation {
        return;
    }

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
        .expect("Failed to run slangc!");

    if !status.success() {
        panic!("Shader compilation failed.");
    }
}

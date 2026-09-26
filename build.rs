use std::{fs, path::Path, process::Command, time::SystemTime};

fn main() {
    compile_slang_spv();
}

#[rustfmt::skip]
fn compile_slang_spv() {
    let shader_dir = Path::new("assets/shader");
    fs::create_dir_all(shader_dir).unwrap();
    let shader_dir = shader_dir.canonicalize().expect("Failed to canonicalize shader directory!");

    let slang_path = shader_dir.join("shader.slang");
    let spv_path = shader_dir.join("slang.spv");

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

    // Run `slangc` with both entry points
    let status = Command::new("slangc")
        .args([
            slang_path.to_str().unwrap(),
            "-target", "spirv", "-profile", "spirv_1_4",
            "-emit-spirv-directly", "-fvk-use-entrypoint-name",
            "-entry", "vertMain", "-entry", "fragMain", "-o",
            spv_path.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to run slangc!");

    if !status.success() {
        panic!("Shader compilation failed.");
    }
}

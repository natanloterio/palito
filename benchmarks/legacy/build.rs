use std::{env, fs, path::Path, process::Command};

fn run(command: &mut Command) {
    let status = command
        .status()
        .expect("could not launch native build command");
    assert!(status.success(), "native build command failed: {command:?}");
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if env::var_os("CARGO_FEATURE_LLAMA_CPP").is_none() {
        return;
    }
    assert_eq!(
        env::var("CARGO_CFG_TARGET_OS").unwrap(),
        "linux",
        "v0.1 supports Linux"
    );
    let root = env::current_dir().unwrap();
    let source = root.parent().unwrap().join("llama.cpp");
    let revision = fs::read_to_string("native/llama-revision.txt").unwrap();
    let actual = Command::new("git")
        .args(["-C", source.to_str().unwrap(), "rev-parse", "HEAD"])
        .output()
        .expect("git is required to validate the llama.cpp checkout");
    assert!(
        actual.status.success(),
        "missing sibling llama.cpp checkout"
    );
    assert_eq!(
        String::from_utf8(actual.stdout).unwrap().trim(),
        revision.trim(),
        "llama.cpp revision differs from native/llama-revision.txt"
    );
    for path in ["native", "build.rs"] {
        println!("cargo:rerun-if-changed={path}");
    }
    // Include source changes, not just HEAD, in Cargo's native rebuild decision.
    for path in [
        "include",
        "src",
        "ggml",
        "common",
        "tools/llama-bench",
        "CMakeLists.txt",
        ".git/HEAD",
        ".git/index",
    ] {
        println!("cargo:rerun-if-changed={}", source.join(path).display());
    }
    println!("cargo:rerun-if-env-changed=LI_BUILD_JOBS");
    let jobs = env::var("LI_BUILD_JOBS").unwrap_or_else(|_| "4".into());
    assert!(
        jobs.parse::<usize>().is_ok_and(|n| n > 0),
        "LI_BUILD_JOBS must be positive"
    );
    let build = Path::new(&env::var("OUT_DIR").unwrap()).join("native");
    run(Command::new("cmake")
        .arg("-S")
        .arg(root.join("native"))
        .arg("-B")
        .arg(&build)
        .arg(format!("-DLLAMA_SOURCE_DIR={}", source.display()))
        .args([
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_CUDA=OFF",
            "-DGGML_NATIVE=ON",
            "-DGGML_OPENMP=OFF",
            "-DGGML_BLAS=OFF",
        ]));
    run(Command::new("cmake").arg("--build").arg(&build).args([
        "--parallel",
        &jobs,
        "--target",
        "li_bridge",
        "li-reference",
        "llama-bench",
    ]));
    let bin = build.join("bin");
    println!("cargo:rustc-link-search=native={}", bin.display());
    println!("cargo:rustc-link-lib=dylib=li_bridge");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", bin.display());
    println!("cargo:rustc-env=LI_NATIVE_BIN={}", bin.display());
    println!("cargo:rustc-env=LI_LLAMA_REVISION={}", revision.trim());
    println!(
        "cargo:rustc-env=LI_BUILD_PROFILE={}",
        env::var("PROFILE").unwrap()
    );
}

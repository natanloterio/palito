//! Linux: let libpalito.so and the palito binary find the NVRTC shipped next
//! to them (the bundle's lib/) without LD_LIBRARY_PATH. cudarc opens NVRTC by
//! name from inside this crate, so the caller's RUNPATH is searched before the
//! system library cache (which may hold a different CUDA version).
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,$ORIGIN");
        println!("cargo:rustc-link-arg-bins=-Wl,-rpath,$ORIGIN/../lib");
    }
}

//! Dynamic shared memory for kernels whose tiles exceed the 48 KB static limit.
//! The device reports each kernel's size (`mmq_dynamic_shared`, sizeof of its
//! shared structs), the functions are opted in to it, and `launch_shared!` passes
//! it at launch.

use super::failure;
use crate::Result;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, LaunchConfig, PushKernelArg,
    sys::CUfunction_attribute_enum,
};
use std::{collections::HashMap, sync::Arc};

/// Kernels sized by `mmq_dynamic_shared`, in its output order.
pub(super) const DYNAMIC_MMQ: [&str; 3] =
    ["mmq_c1_pipe128", "mmq_c1_pipe128_splitk", "mmq_c1_pipe64"];

/// Queries the dynamic sizes and opts the loaded kernels in to them.
pub(super) fn configure(
    context: &Arc<CudaContext>,
    module: &Arc<CudaModule>,
    functions: &HashMap<&'static str, CudaFunction>,
) -> Result<HashMap<&'static str, u32>> {
    let stream = context.default_stream();
    let query = module
        .load_function("mmq_dynamic_shared")
        .map_err(failure)?;
    let mut bytes = stream
        .alloc_zeros::<i32>(DYNAMIC_MMQ.len())
        .map_err(failure)?;
    let mut args = stream.launch_builder(&query);
    args.arg(&mut bytes);
    // One thread writes DYNAMIC_MMQ.len() ints.
    unsafe { args.launch(LaunchConfig::for_num_elems(1)) }.map_err(failure)?;
    let sizes = stream.clone_dtoh(&bytes).map_err(failure)?;
    let mut shared = HashMap::new();
    for (name, size) in DYNAMIC_MMQ.into_iter().zip(sizes) {
        let size = validate(name, size)?;
        let Some(function) = functions.get(name) else {
            continue;
        };
        function
            .set_attribute(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                size as i32,
            )
            .map_err(failure)?;
        shared.insert(name, size);
    }
    Ok(shared)
}

/// A reported size must be positive and within the 227 KB per-block hardware limit.
fn validate(name: &str, size: i32) -> Result<u32> {
    if !(1..=227 * 1024).contains(&size) {
        return Err(failure(format!(
            "{name}: invalid dynamic shared size {size}"
        )));
    }
    Ok(size as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_are_validated() {
        assert_eq!(validate("k", 78_000).ok(), Some(78_000));
        assert!(validate("k", 0).is_err());
        assert!(validate("k", 300 * 1024).is_err());
    }

    #[test]
    fn dynamic_kernels_are_unique() {
        let mut names = DYNAMIC_MMQ.to_vec();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), DYNAMIC_MMQ.len());
    }
}

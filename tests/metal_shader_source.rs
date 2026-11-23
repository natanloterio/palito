//! Source-level Metal checks that run on every platform.

const SHADERS: &str = include_str!("../src/metal/shaders.metal");
const POSITION_ATTRIBUTES: [&str; 2] = [
    "threadgroup_position_in_grid",
    "thread_position_in_threadgroup",
];

fn attribute_type(signature: &str, attribute: &str) -> Option<String> {
    let marker = format!("[[{attribute}]]");
    let line = signature.lines().find(|line| line.contains(&marker))?;
    line.split_whitespace().next().map(str::to_owned)
}

#[test]
fn kernel_position_attributes_share_one_type() {
    let mismatched: Vec<_> = SHADERS
        .split("kernel void ")
        .skip(1)
        .filter_map(|kernel| {
            let (signature, _) = kernel.split_once('{')?;
            let [group, thread] =
                POSITION_ATTRIBUTES.map(|attribute| attribute_type(signature, attribute));
            let name = signature.split('(').next()?;
            (group.is_some() && thread.is_some() && group != thread).then(|| name.to_owned())
        })
        .collect();
    assert!(mismatched.is_empty(), "mismatched kernels: {mismatched:?}");
}

//! Baseline: the optimal 2-bit scalar (Lloyd-Max) quantizer for N(0, 1).

/// Expected distortion of `lloyd_max_2bit` on N(0, 1) samples.
pub const LLOYD_MAX_2BIT_MSE: f64 = 0.1175;

const INNER_LEVEL: f32 = 0.4528;
const OUTER_LEVEL: f32 = 1.5104;
/// Decision threshold: the midpoint of the inner and outer levels.
const THRESHOLD: f32 = 0.9816;

pub fn lloyd_max_2bit(value: f32) -> f32 {
    let magnitude = if value.abs() < THRESHOLD {
        INNER_LEVEL
    } else {
        OUTER_LEVEL
    };
    magnitude.copysign(value)
}

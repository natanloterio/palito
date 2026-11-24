//! Packed trellis tiles for GPU decoding.
//!
//! A `16 x 16` tile is one trellis sequence of `T = 256` weights (index
//! `t = row * 16 + col`). Its bits form one MSB-first stream: the `L`-bit start
//! state, then `K` bits per later weight. The state of weight `t` is the
//! `L`-bit window at bit offset `K * t`, so any weight decodes without the
//! ones before it. Without tail-biting a tile takes `L + K (T - 1)` bits,
//! rounded up to whole 32-bit words.

use super::{
    codebook::one_mad,
    ldlq::TILE,
    trellis::{Encoded, Trellis},
};
use crate::{Error, Result};
use rayon::prelude::*;

pub const TILE_WEIGHTS: usize = TILE * TILE;

/// Words per packed tile.
pub fn tile_words(state_bits: u32, symbol_bits: u32) -> usize {
    let bits = state_bits as usize + symbol_bits as usize * (TILE_WEIGHTS - 1);
    bits.div_ceil(32)
}

fn put_bits(words: &mut [u32], offset: usize, width: u32, value: u32) {
    for bit in 0..width as usize {
        if value >> (width as usize - 1 - bit) & 1 == 1 {
            let position = offset + bit;
            words[position / 32] |= 1 << (31 - position % 32);
        }
    }
}

/// Packs one encoded tile (`TILE_WEIGHTS` weights).
pub fn pack_tile(encoded: &Encoded, state_bits: u32, symbol_bits: u32) -> Result<Vec<u32>> {
    if encoded.symbols.len() != TILE_WEIGHTS - 1 {
        return Err(Error::Invalid(format!("a tile has {TILE_WEIGHTS} weights")));
    }
    let mut words = vec![0u32; tile_words(state_bits, symbol_bits)];
    put_bits(&mut words, 0, state_bits, encoded.start);
    for (index, &symbol) in encoded.symbols.iter().enumerate() {
        let offset = state_bits as usize + index * symbol_bits as usize;
        put_bits(&mut words, offset, symbol_bits, u32::from(symbol));
    }
    Ok(words)
}

/// State of weight `t` of a packed tile.
pub fn state_at(words: &[u32], t: usize, state_bits: u32, symbol_bits: u32) -> u32 {
    let offset = t * symbol_bits as usize;
    let word = offset / 32;
    let high = u64::from(words[word]) << 32;
    let low = words.get(word + 1).copied().map_or(0, u64::from);
    let window = (high | low) << (offset % 32);
    (window >> (64 - state_bits)) as u32
}

/// A matrix in the rotated domain as packed tiles, row group by row group
/// (tile `(g, k)` at index `g * cols / 16 + k`), with one scale.
pub struct PackedMatrix {
    pub rows: usize,
    pub cols: usize,
    pub state_bits: u32,
    pub symbol_bits: u32,
    pub sigma: f32,
    pub words: Vec<u32>,
}

impl PackedMatrix {
    /// Trellis-encodes each tile of `values` (row-major, already rotated)
    /// after dividing by `sigma`. No LDLQ: this layout test does not need it.
    pub fn encode(
        values: &[f32],
        rows: usize,
        cols: usize,
        trellis_bits: (u32, u32),
        sigma: f32,
    ) -> Result<Self> {
        let (state_bits, symbol_bits) = trellis_bits;
        if !rows.is_multiple_of(TILE) || !cols.is_multiple_of(TILE) || values.len() != rows * cols {
            return Err(Error::Invalid(format!(
                "packed tiles need {TILE}-aligned shapes"
            )));
        }
        let trellis = Trellis::new(state_bits, symbol_bits, 1.0)?;
        let col_tiles = cols / TILE;
        let tiles: Vec<Vec<u32>> = (0..rows / TILE * col_tiles)
            .into_par_iter()
            .map(|index| {
                let (group, block) = (index / col_tiles, index % col_tiles);
                let tile: Vec<f32> = (0..TILE_WEIGHTS)
                    .map(|t| {
                        values[(group * TILE + t / TILE) * cols + block * TILE + t % TILE] / sigma
                    })
                    .collect();
                pack_tile(&trellis.quantize(&tile), state_bits, symbol_bits)
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            rows,
            cols,
            state_bits,
            symbol_bits,
            sigma,
            words: tiles.concat(),
        })
    }

    /// Dense row-major reconstruction (the CPU reference).
    pub fn decode(&self) -> Vec<f32> {
        let words_per_tile = tile_words(self.state_bits, self.symbol_bits);
        let col_tiles = self.cols / TILE;
        let mut out = vec![0.0; self.rows * self.cols];
        for (index, tile) in self.words.chunks_exact(words_per_tile).enumerate() {
            let (group, block) = (index / col_tiles, index % col_tiles);
            for t in 0..TILE_WEIGHTS {
                let state = state_at(tile, t, self.state_bits, self.symbol_bits);
                out[(group * TILE + t / TILE) * self.cols + block * TILE + t % TILE] =
                    self.sigma * one_mad(state);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(count: usize) -> Vec<f32> {
        (0..count)
            .map(|i| ((i * 7919 % 1000) as f32 - 500.0) / 300.0)
            .collect()
    }

    #[test]
    fn packed_tile_decodes_like_the_trellis() {
        for (state_bits, symbol_bits) in [(12, 2), (12, 3), (8, 1)] {
            let trellis = Trellis::new(state_bits, symbol_bits, 1.0).unwrap();
            let encoded = trellis.quantize(&values(TILE_WEIGHTS));
            let words = pack_tile(&encoded, state_bits, symbol_bits).unwrap();
            assert_eq!(words.len(), tile_words(state_bits, symbol_bits));
            let expected = trellis.decode(&encoded);
            for (t, &value) in expected.iter().enumerate() {
                assert_eq!(
                    one_mad(state_at(&words, t, state_bits, symbol_bits)),
                    value,
                    "L{state_bits} K{symbol_bits} t{t}"
                );
            }
        }
    }

    #[test]
    fn three_bit_tiles_take_25_words() {
        assert_eq!(tile_words(12, 3), 25);
        assert_eq!(tile_words(12, 2), 17);
    }

    #[test]
    fn packed_matrix_round_trips_tile_order() {
        let (rows, cols) = (2 * TILE, 3 * TILE);
        let w = values(rows * cols);
        let sigma = 0.8;
        let packed = PackedMatrix::encode(&w, rows, cols, (12, 3), sigma).unwrap();
        let dense = packed.decode();
        let trellis = Trellis::new(12, 3, 1.0).unwrap();
        // Tile (1, 2) decoded directly must land at rows 16..32, cols 32..48.
        let tile: Vec<f32> = (0..TILE_WEIGHTS)
            .map(|t| w[(TILE + t / TILE) * cols + 2 * TILE + t % TILE] / sigma)
            .collect();
        let direct = trellis.decode(&trellis.quantize(&tile));
        for t in 0..TILE_WEIGHTS {
            assert_eq!(
                dense[(TILE + t / TILE) * cols + 2 * TILE + t % TILE],
                sigma * direct[t]
            );
        }
        let error: f32 = w
            .iter()
            .zip(&dense)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / w.iter().map(|a| a * a).sum::<f32>();
        assert!(error < 0.1, "relative error {error}");
    }
}

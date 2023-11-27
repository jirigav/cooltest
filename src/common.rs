use clap::Parser;
use std::fs;

pub(crate) fn z_score(sample_size: usize, positive: usize, p: f64) -> f64 {
    ((positive as f64) - p * (sample_size as f64)) / f64::sqrt(p * (1.0 - p) * (sample_size as f64))
}

#[derive(Parser)]
#[command(version)]
pub(crate) struct Args {
    /// Path of file with input data.
    pub(crate) data_source: String,

    /// Length of block of data.
    #[arg(short, long, default_value_t = 128)]
    pub(crate) block_size: usize,

    /// Number of explored distinguisher branches.
    #[arg(short, long, default_value_t = 10)]
    pub(crate) k: usize,

    /// Minimal difference between expected and actual count of a given pattern in data.
    #[arg(short, long, default_value_t = 100)]
    pub(crate) min_difference: usize,

    /// Number of polynomials from first phase used in the greedy search.
    #[arg(short, long, default_value_t = 100)]
    pub(crate) n: usize,

    /// The degree of polynomials evaluated in the first phase.
    #[arg(short, long, default_value_t = 2)]
    pub(crate) deg: usize,
}

pub(crate) fn bit_value_in_block(bit: usize, block: &[u8]) -> bool {
    let (byte_index, offset) = (bit / 8, bit % 8);
    ((block[byte_index] >> offset) & 1) == 1
}

pub(crate) fn bits_block_eval(bits: &[usize], block: &[u8]) -> usize {
    let mut result = 0;

    for (i, b) in bits.iter().enumerate() {
        if bit_value_in_block(*b, block) {
            result += 2_usize.pow(i as u32);
        }
    }
    result
}

pub(crate) fn load_data(path: &str, block_size: usize) -> Vec<Vec<u8>> {
    let len_of_block_in_bytes = block_size / 8;
    fs::read(path)
        .unwrap()
        .chunks(len_of_block_in_bytes)
        .map(<[u8]>::to_vec)
        .collect()
}

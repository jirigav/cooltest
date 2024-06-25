use clap::Parser;
use pyo3::prelude::*;
use std::fs;

pub(crate) fn z_score(sample_size: usize, positive: usize, p: f64) -> f64 {
    ((positive as f64) - p * (sample_size as f64)) / f64::sqrt(p * (1.0 - p) * (sample_size as f64))
}

#[derive(Parser, Debug)]
#[command(version)]
pub(crate) struct Args {
    /// Path of file with input data.
    pub(crate) data_source: String,

    /// Length of block of data.
    #[arg(short, long, default_value_t = 128)]
    pub(crate) block: usize,

    /// Number of bits in histograms in brute-force search.
    #[arg(short, long, default_value_t = 2)]
    pub(crate) k: usize,

    /// Number of best histograms taken for the second step.
    #[arg(short, long, default_value_t = 1)]
    pub(crate) top: usize,

    /// Number of histograms combined in second step.
    #[arg(short, long, default_value_t = 1)]
    pub(crate) max_bits: usize,

    /// Significance level
    #[arg(short, long, default_value_t = 0.0001)]
    pub(crate) alpha: f64,

    /// Number of threads for multi-thread run. 0 means that efficient single thread implementation is used.
    #[arg(short, long, default_value_t = 0)]
    pub(crate) threads: usize,
}

pub(crate) fn bits_block_eval(bits: &[usize], block: &[u8]) -> usize {
    let mut result = 0;

    for (i, b) in bits.iter().enumerate() {
        if bit_value_in_block(*b, block) {
            result += 1 << i;
        }
    }
    result
}

pub(crate) fn bit_value_in_block(bit: usize, block: &[u8]) -> bool {
    let (byte_index, offset) = (bit / 8, 7 - (bit % 8));
    ((block[byte_index] >> offset) & 1) == 1
}

#[derive(Clone)]
pub(crate) struct Data {
    pub(crate) data: Vec<Vec<u128>>,
    pub(crate) _mask: u128,
    pub(crate) _num_of_blocks: u32,
}

pub(crate) fn multi_eval(bits: &[usize], data: &Data) -> usize {
    let mut result = vec![u128::MAX; data.data[0].len()];

    for b in bits.iter() {
        result = result
            .iter()
            .zip(&data.data[*b])
            .map(|(a, b)| a & b)
            .collect();
    }

    let r = result
        .iter()
        .map(|x| x.count_ones() as usize)
        .sum::<usize>();
    r
}

fn load_data(path: &str, block_size: usize) -> Vec<Vec<u8>> {
    let len_of_block_in_bytes = block_size / 8;
    let mut data: Vec<_> = fs::read(path)
        .unwrap()
        .chunks(len_of_block_in_bytes)
        .map(<[u8]>::to_vec)
        .collect();
    if data[data.len() - 1].len() != len_of_block_in_bytes {
        println!("Data are not aligned with block size, dropping last block!");
        data.pop();
    }
    data
}

pub(crate) fn prepare_data(data_source: &str, block_size: usize) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let data = load_data(data_source, block_size);

    let (tr_data, testing_data) = data.split_at(data.len() / 2);

    (tr_data.to_vec(), testing_data.to_vec())
}

/// Returns data transformed into vectors of u64, where i-th u64 contains values of 64 i-th bits of consecutive blocks.
pub(crate) fn transform_data(data: &[Vec<u8>]) -> Data {
    let mut result = Vec::new();
    let block_size = data[0].len() * 8;
    for blocks in data.chunks(128) {
        let mut ints = vec![0_u128; block_size];

        for (e, block) in blocks.iter().enumerate() {
            for (i, int) in ints.iter_mut().enumerate().take(block_size) {
                if bit_value_in_block(i, block) {
                    *int += 1_u128 << e;
                }
            }
        }
        result.push(ints);
    }
    let mask = if data.len() % 128 == 0 {
        u128::MAX
    } else {
        2_u128.pow((data.len() % 128) as u32) - 1
    };
    let mut d = Vec::new();

    for i in 0..(data[0].len() * 8) {
        let mut bit = Vec::new();
        for block in result.iter() {
            bit.push(block[i])
        }
        d.push(bit);
    }
    Data {
        data: d,
        _mask: mask,
        _num_of_blocks: data.len() as u32,
    }
}

pub(crate) fn p_value(positive: usize, sample_size: usize, probability: f64) -> f64 {
    Python::with_gil(|py| {
        let scipy_stats = PyModule::import(py, "scipy.stats").expect("SciPy not installed! Use `pip install scipy` to install the library.");
        let result: f64 = scipy_stats
            .getattr("binomtest")
            .expect("Scipy binomtest not found! Make sure that your version os SciPy is >=1.7.0.")
            .call1((positive, sample_size, probability, "two-sided"))
            .unwrap()
            .getattr("pvalue")
            .unwrap()
            .extract()
            .unwrap();
        result
    })
}

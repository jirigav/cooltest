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

    /// Number of explored branches of distinguishers in greedy search.
    #[arg(short, long, default_value_t = 10)]
    pub(crate) k: usize,

    /// Number of bits in histograms in brute-force search.
    #[arg(short, long, default_value_t = 2)]
    pub(crate) deg: usize,

    /// Option whether the input data should be divided into training, validation and testing data.
    #[arg(long, short)]
    pub(crate) validation_and_testing_split: bool,

    /// Maximal number of bits in a histogram during greedy search.
    #[arg(long, short, default_value_t = 10)]
    pub(crate) max_bits: usize,

    /// When some histogram reaches this value of p-value, greedy search ends.
    #[arg(long, short, default_value_t = 0.0)]
    pub(crate) stop_p_value: f64,

    /// When the change between histograms between consecutive rounds reaches this value the search stops.
    #[arg(long, short, default_value_t = 0.0)]
    pub(crate) stop_change: f64,

    /// Significance level
    #[arg(short, long, default_value_t = 0.0004)]
    pub(crate) alpha: f64,
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

pub(crate) fn bit_value_in_block(bit: usize, block: &[u8]) -> bool {
    let (byte_index, offset) = (bit / 8, bit % 8);
    ((block[byte_index] >> offset) & 1) == 1
}

pub(crate) fn count_combinations(n: usize, r: usize) -> usize {
    if r > n {
        0
    } else {
        (1..=r.min(n - r)).fold(1, |acc, val| acc * (n - val + 1) / val)
    }
}

pub(crate) struct Data {
    pub(crate) data: Vec<Vec<u128>>,
    pub(crate) mask: u128,
    pub(crate) _num_of_blocks: usize,
}

pub(crate) fn multi_eval(
    bits_signs: usize,
    bits: &[usize],
    tr_data: &[u128],
    mask: u128,
    is_last: bool,
) -> u128 {
    let mut result = u128::MAX;

    for (i, b) in bits.iter().enumerate() {
        if ((bits_signs >> i) & 1) == 1 {
            result &= tr_data[*b];
        } else {
            result &= tr_data[*b] ^ u128::MAX;
        }
    }
    if is_last {
        result &= mask;
    }
    result
}

pub(crate) fn multi_eval_count(
    bits_signs: usize,
    bits: &[usize],
    tr_data: &[u128],
    mask: u128,
    is_last: bool,
) -> u32 {
    multi_eval(bits_signs, bits, tr_data, mask, is_last).count_ones()
}

fn load_data(path: &str, block_size: usize) -> Vec<Vec<u8>> {
    let len_of_block_in_bytes = block_size / 8;
    fs::read(path)
        .unwrap()
        .chunks(len_of_block_in_bytes)
        .map(<[u8]>::to_vec)
        .collect()
}

pub(crate) fn prepare_data(
    data_source: &str,
    block_size: usize,
    halving: bool,
    validation: bool,
) -> (Vec<Vec<u8>>, Option<Vec<Vec<u8>>>, Option<Vec<Vec<u8>>>) {
    let data = load_data(data_source, block_size);
    let training_data;
    let mut testing_data_option = None;
    let mut validation_data_option = None;

    if validation {
        let (tr_data, testing_data) = data.split_at(data.len() / 3);
        let (val_data, test_data) = testing_data.split_at(testing_data.len() / 2);
        testing_data_option = Some(test_data.to_vec());
        validation_data_option = Some(val_data.to_vec());
        training_data = tr_data.to_vec();
    } else if halving {
        let (tr_data, testing_data) = data.split_at(data.len() / 2);
        testing_data_option = Some(testing_data.to_vec());
        training_data = tr_data.to_vec();
    } else {
        training_data = data;
    }
    (training_data, validation_data_option, testing_data_option)
}

/// Returns data transformed into vectors of u64, where i-th u64 contains values of 64 i-th bits of consecutive blocks.
pub(crate) fn transform_data(data: &Vec<Vec<u8>>) -> Data {
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
    Data {
        data: result,
        mask,
        _num_of_blocks: data.len(),
    }
}

pub(crate) fn p_value(positive: usize, sample_size: usize, probability: f64) -> f64 {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let scipy = PyModule::import(py, "scipy").unwrap();
        let result: f64 = scipy
            .getattr("stats")
            .unwrap()
            .getattr("binom_test")
            .unwrap()
            .call1((positive, sample_size, probability, "two-sided"))
            .unwrap()
            .extract()
            .unwrap();
        result
    })
}

pub(crate) fn p_value_to_z_score(p_value: f64) -> f64 {
    if p_value == 0.0 {
        return f64::MAX;
    }
    pyo3::prepare_freethreaded_python();
    f64::abs(Python::with_gil(|py| {
        let scipy = PyModule::import(py, "scipy").unwrap();
        let result: f64 = scipy
            .getattr("stats")
            .unwrap()
            .getattr("norm")
            .unwrap()
            .getattr("ppf")
            .unwrap()
            .call1((p_value,))
            .unwrap()
            .extract()
            .unwrap();
        result
    }))
}

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
    pub(crate) block_size: usize,

    /// Number of explored pattern branches.
    #[arg(short, long, default_value_t = 100)]
    pub(crate) k: usize,

    /// Minimal difference between expected and actual count of a given pattern in data.
    #[arg(short, long, default_value_t = 100)]
    pub(crate) min_difference: usize,

    /// Number of patterns combined into a multipattern.
    #[arg(long, default_value_t = 50)]
    pub(crate) top_n: usize,

    /// Number of patterns combined into a multipattern.
    #[arg(short, long, default_value_t = 2)]
    pub(crate) patterns_combined: usize,

    /// Length of patterns evaluated in the first phase.
    #[arg(short, long, default_value_t = 2)]
    pub(crate) base_pattern_size: usize,

    /// Option whether the input data should be divided into training, validation and testing data.
    #[arg(long, short)]
    pub(crate) validation_and_testing_split: bool,

    /// Option whether histogram should be used as an alternative evaluation method.
    #[arg(long)]
    pub(crate) hist: bool,

    /// Option whether histogram should be used as an alternative evaluation method.
    #[arg(long, short)]
    pub(crate) config: bool,
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
    pub(crate) num_of_blocks: usize,
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
) -> (Data, Option<Data>, Option<Data>) {
    let data = load_data(data_source, block_size);
    let training_data;
    let mut testing_data_option = None;
    let mut validation_data_option = None;

    if validation {
        let (tr_data, testing_data) = data.split_at(data.len() / 3);
        let (val_data, test_data) = testing_data.split_at(testing_data.len() / 2);
        testing_data_option = Some(transform_data(test_data.to_vec()));
        validation_data_option = Some(transform_data(val_data.to_vec()));
        training_data = transform_data(tr_data.to_vec());
    } else if halving {
        let (tr_data, testing_data) = data.split_at(data.len() / 2);
        testing_data_option = Some(transform_data(testing_data.to_vec()));
        training_data = transform_data(tr_data.to_vec());
    } else {
        training_data = transform_data(data);
    }
    (training_data, validation_data_option, testing_data_option)
}

/// Returns data transformed into vectors of u64, where i-th u64 contains values of 64 i-th bits of consecutive blocks.
pub(crate) fn transform_data(data: Vec<Vec<u8>>) -> Data {
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
        num_of_blocks: data.len(),
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

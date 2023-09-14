use clap::Parser;
use pyo3::prelude::*;
use std::fs;

pub(crate) type Data = Vec<Vec<u8>>;

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
    #[arg(short, long, default_value_t = 10)]
    pub(crate) k: usize,

    /// Minimal difference between expected and actual count of a given pattern in data.
    #[arg(short, long, default_value_t = 100)]
    pub(crate) min_difference: usize,

    /// Number of patterns combined into a multipattern.
    #[arg(short, long, default_value_t = 2)]
    pub(crate) patterns_combined: usize,

    /// Length of patterns evaluated in the first phase.
    #[arg(short, long, default_value_t = 2)]
    pub(crate) base_pattern_size: usize,

    /// Option whether the input data should be halved into training and testing data.
    #[arg(long)]
    pub(crate) halving: bool,

    /// Option whether the input data should be divided into training, validation and testing data.
    #[arg(long, short)]
    pub(crate) validation_and_testing_split: bool,

    /// Option whether histogram should be used as an alternative evaluation method.
    #[arg(long)]
    pub(crate) hist: bool,
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

fn load_data(path: &str, block_size: usize) -> Data {
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
    let mut training_data = load_data(data_source, block_size);
    let mut testing_data_option = None;
    let mut validation_data_option = None;

    if validation {
        let (tr_data, testing_data) = training_data.split_at(training_data.len() / 3);
        let (val_data, test_data) = testing_data.split_at(testing_data.len() / 2);
        testing_data_option = Some(test_data.to_vec());
        validation_data_option = Some(val_data.to_vec());
        training_data = tr_data.to_vec();
    } else if halving {
        let (tr_data, testing_data) = training_data.split_at(training_data.len() / 2);
        testing_data_option = Some(testing_data.to_vec());
        training_data = tr_data.to_vec();
    }
    (training_data, validation_data_option, testing_data_option)
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

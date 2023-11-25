use clap::Parser;
use itertools::Itertools;
use pyo3::prelude::*;
use std::fs;

pub(crate) fn z_score(sample_size: usize, positive: usize, p: f64) -> f64 {
    ((positive as f64) - p * (sample_size as f64)) / f64::sqrt(p * (1.0 - p) * (sample_size as f64))
}

#[derive(Parser, Debug)]
#[command(version)]
pub(crate) struct Args {
    /// Path to file with input data.
    pub(crate) data_source: String,

    /// Block size in bits.
    #[arg(short, long, default_value_t = 128)]
    pub(crate) block: usize,

    /// If the value is greater than 1, CoolTest looks for distinguisher on block size that is a multiple of 'block' and utilizes all such tuples of consecutive blocks.
    #[arg(long, default_value_t = 1)]
    pub(crate) block_size_multiple: usize,

    /// Number of explored branches of distinguishers in greedy search.
    #[arg(short, long, default_value_t = 100)]
    pub(crate) k: usize,

    /// Minimal difference between expected and actual count of a given pattern in data.
    #[arg(short, long, default_value_t = 100)]
    pub(crate) min_difference: usize,

    /// Number of distinguishers tested in combinations for a multipattern.
    #[arg(long, default_value_t = 50)]
    pub(crate) top_n: usize,

    /// Maximal number of bits (variables) used in a distinguishers.
    #[arg(long)]
    pub(crate) max_bits: Option<usize>,

    /// Number of patterns combined into a multipattern.
    #[arg(short, long, default_value_t = 2)]
    pub(crate) patterns_combined: usize,

    /// Degree of distinguishers evaluated in the first phase.
    #[arg(short, long, default_value_t = 2)]
    pub(crate) deg: usize,

    /// Option whether the input data should be divided into training, validation and testing data.
    #[arg(long, short)]
    pub(crate) validation_and_testing_split: bool,

    /// Option whether histogram should be used as a strengthening method.
    #[arg(long)]
    pub(crate) hist: bool,

    /// Config file with list of CoolTest configurations to run.
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

fn load_data(path: &str, block_size: usize, block_size_multiple: usize) -> Vec<Vec<u8>> {
    let len_of_block_in_bytes = (block_size * block_size_multiple) / 8;
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

pub(crate) fn prepare_data(
    data_source: &str,
    block_size: usize,
    block_size_multiple: usize,
    halving: bool,
    validation: bool,
) -> (Data, Option<Data>, Option<Data>) {
    let data = load_data(data_source, block_size, block_size_multiple);
    let training_data;
    let mut testing_data_option = None;
    let mut validation_data_option = None;

    if validation {
        let (tr_data, testing_data) = data.split_at(data.len() / 3);
        let (val_data, test_data) = testing_data.split_at(testing_data.len() / 2);
        testing_data_option = Some(transform_data(test_data.to_vec()));
        validation_data_option = Some(transform_data(val_data.to_vec()));
        training_data = transform_training_data(tr_data.to_vec(), block_size, block_size_multiple);
    } else if halving {
        let (tr_data, testing_data) = data.split_at(data.len() / 2);
        testing_data_option = Some(transform_data(testing_data.to_vec()));
        training_data = transform_training_data(tr_data.to_vec(), block_size, block_size_multiple);
    } else {
        training_data = transform_training_data(data, block_size, block_size_multiple);
    }
    println!(
        "tr {}, te {}",
        training_data.data.len(),
        testing_data_option.as_ref().unwrap().data.len()
    );
    (training_data, validation_data_option, testing_data_option)
}

fn transform_training_data(
    data: Vec<Vec<u8>>,
    block_size: usize,
    block_size_multiple: usize,
) -> Data {
    if block_size_multiple == 1 {
        return transform_data(data);
    }

    let data_flattened: Vec<Vec<u8>> = data
        .into_iter()
        .flatten()
        .collect_vec()
        .chunks(block_size / 8)
        .map(<[u8]>::to_vec)
        .collect();

    let mut data_duplicated = Vec::new();

    for i in 0..(data_flattened.len() - block_size_multiple + 1) {
        let mut block: Vec<u8> = Vec::new();
        for j in 0..block_size_multiple {
            block.append(&mut data_flattened[i + j].clone());
        }
        data_duplicated.push(block);
    }

    transform_data(data_duplicated)
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

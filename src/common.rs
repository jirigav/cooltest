use clap::{Parser, Subcommand};
use rand::seq::SliceRandom;
use std::fs::File;
use std::{fs, io::Write};

pub(crate) fn z_score(sample_size: usize, positive: usize, p: f64) -> f64 {
    ((positive as f64) - p * (sample_size as f64)) / f64::sqrt(p * (1.0 - p) * (sample_size as f64))
}

#[derive(Parser)]
#[command(version)]
pub(crate) struct Args {
    #[clap(subcommand)]
    pub(crate) tool: Subcommands,
}

#[derive(Subcommand)]
#[command(version)]
pub(crate) enum Subcommands {
    /// Simple tool to shuffle blocks of given length of the input file and save shuffled data into output file.
    ShuffleData {
        /// Length of block of data.
        block_size: usize,

        /// Path to input file.
        input_file_path: String,

        /// Path to where the optput of this function should be stored.
        output_file_path: String,
    },

    /// Tool for finding frequent patterns in data.
    Bottomup {
        /// Path of file with input data.
        data_source: String,

        /// Length of block of data.
        #[arg(short, long, default_value_t = 128)]
        block_size: usize,

        /// Number of explored pattern branches.
        #[arg(short, long, default_value_t = 10)]
        k: usize,

        /// Minimal difference between expected and actual count of a given pattern in data.
        #[arg(short, long, default_value_t = 100)]
        min_difference: usize,

        /// Number of patterns combined into a multipattern.
        #[arg(short, long, default_value_t = 2)]
        patterns_combined: usize,

        /// Length of patterns evaluated in the first phase.
        #[arg(short, long, default_value_t = 2)]
        base_pattern_size: usize,

        /// Option whether the input data should be halved into training and testing data.
        #[arg(long)]
        halving: bool,

        /// Option whether the input data should be divided into training, validation and testing data.
        #[arg(long, short)]
        validation_and_testing_split: bool,

        /// Option whether histogram should be used as an alternative evaluation method.
        #[arg(long)]
        hist: bool,
    },
    /// Tool similar to bottom up with base_pattern_size=1, but with distinguishers constructed as boolean polynomials and the ability to find also infrequent patterns.
    Polyup {
        /// Path of file with input data.
        data_source: String,

        /// Length of block of data.
        #[arg(short, long, default_value_t = 128)]
        block_size: usize,

        /// Number of explored pattern branches.
        #[arg(short, long, default_value_t = 10)]
        k: usize,

        /// Number of bits considered for the patterns.
        #[arg(short, long, default_value_t = 64)]
        n: usize,

        /// Minimal difference between expected and actual count of a given pattern in data.
        #[arg(short, long, default_value_t = 100)]
        min_difference: usize,

        /// Option whether the input data should be halved into training and testing data.
        #[arg(long)]
        halving: bool,
    },
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

pub(crate) fn shuffle_data(data_in_path: &str, data_out_path: &str, block_size: usize) {
    let mut data = load_data(data_in_path, block_size);
    let mut rng = rand::thread_rng();
    data.shuffle(&mut rng);

    let mut file_out = File::create(data_out_path).unwrap();

    file_out
        .write_all(&data.iter().flatten().copied().collect::<Vec<u8>>())
        .unwrap();
}

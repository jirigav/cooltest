use clap::Parser;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Parser, Debug, Serialize, Deserialize, Clone)]
#[command(version)]
pub(crate) struct Args {
    /// Path of file with input data.
    pub(crate) data_source: String,

    /// Length of block of data.
    #[arg(short, long, default_value_t = 128)] // Changing the default value changes autotest
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

    /// Path where json output should be stored. If no path provided, json output is not stored.
    #[arg(short, long)]
    pub(crate) json: Option<String>,

    #[clap(subcommand)]
    pub subcommand: Option<SubCommand>,
}

#[derive(Parser, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub(crate) enum SubCommand {
    /// Evaluate a given distinguisher on given data and report p-value.
    Evaluate {
        /// Path of file with distinguisher which should be evaluated.
        #[arg(short, long)]
        dis_path: String,
    },
    Autotest {},
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

#[derive(Clone, PartialEq, Debug)]
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

pub(crate) fn prepare_data(
    data_source: &str,
    block_size: usize,
    training_data: bool,
) -> (Vec<Vec<u8>>, Option<Vec<Vec<u8>>>) {
    let data = load_data(data_source, block_size);
    if !training_data {
        (data, None)
    } else {
        let (tr_data, testing_data) = data.split_at(data.len() / 2);

        (tr_data.to_vec(), Some(testing_data.to_vec()))
    }
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

pub(crate) fn z_score(sample_size: usize, positive: usize, p: f64) -> f64 {
    ((positive as f64) - p * (sample_size as f64)) / f64::sqrt(p * (1.0 - p) * (sample_size as f64))
}

pub(crate) fn p_value(sample_size: usize, positive: usize, probability: f64) -> f64 {
    Python::with_gil(|py| {
        let scipy_stats = PyModule::import(py, "scipy.stats")
            .expect("SciPy not installed! Use `pip install scipy` to install the library.");
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

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::approx_eq;

    #[test]
    fn test_p_value() {
        assert!(approx_eq!(f64, p_value(1, 1, 0.0), 0.0));
        assert!(approx_eq!(f64, p_value(1, 1, 1.0), 1.0));
        assert!(approx_eq!(f64, p_value(1, 1, 0.5), 1.0));
        assert!(approx_eq!(f64, p_value(1, 1, 0.25), 0.25));
        assert!(approx_eq!(f64, p_value(8064, 675, 0.85), 0.0));
        assert!(approx_eq!(f64, p_value(1245, 872, 0.51), 2.519147904123094e-42));
        assert!(approx_eq!(f64, p_value(3952, 3009, 0.87), 1.6048354143177452e-76));
        assert!(approx_eq!(f64, p_value(6395, 1774, 0.32), 1.4633129278540793e-13));
        assert!(approx_eq!(f64, p_value(7716, 969, 0.76), 0.0));
        assert!(approx_eq!(f64, p_value(4231, 1225, 0.75), 0.0));
        assert!(approx_eq!(f64, p_value(2295, 1187, 0.02), 0.0));
        assert!(approx_eq!(f64, p_value(2228, 1993, 0.61), 8.219896711580438e-200));
        assert!(approx_eq!(f64, p_value(5936, 4649, 0.97), 0.0));
        assert!(approx_eq!(f64, p_value(711, 342, 0.2), 5.29655579272766e-63));
    }

    #[test]
    fn test_z_score() {
        assert!(z_score(1, 1, 1.0).is_nan());
        assert_eq!(z_score(8852, 7609, 0.74), 25.649318571444642);
        assert_eq!(z_score(8838, 7708, 0.99), -111.35627996866052);
        assert_eq!(z_score(1040, 1037, 0.34), 44.73494199130066);
        assert_eq!(z_score(5204, 1855, 0.85), -99.71004790616179);
        assert_eq!(z_score(8878, 386, 0.19), -35.1917063646087);
        assert_eq!(z_score(8377, 1181, 0.49), -63.9013271819615);
        assert_eq!(z_score(9682, 2871, 0.11), 58.65959381857785);
        assert_eq!(z_score(6615, 343, 0.21), -31.579543090478786);
        assert_eq!(z_score(4997, 4918, 0.41), 82.52637218309836);
        assert_eq!(z_score(9609, 1609, 0.28), -24.57254813392147);
    }

    #[test]
    fn test_bit_value_in_block() {
        assert_eq!(bit_value_in_block(0, &[2_u8.pow(7)]), true);
        assert_eq!(bit_value_in_block(0, &[2_u8.pow(6)]), false);
        assert_eq!(bit_value_in_block(1, &[2_u8.pow(6)]), true);
        assert_eq!(bit_value_in_block(2, &[2_u8.pow(5)]), true);
        assert_eq!(bit_value_in_block(3, &[2_u8.pow(4)]), true);
        assert_eq!(bit_value_in_block(4, &[2_u8.pow(3)]), true);
        assert_eq!(bit_value_in_block(5, &[2_u8.pow(2)]), true);
        assert_eq!(bit_value_in_block(6, &[2_u8.pow(1)]), true);
        assert_eq!(bit_value_in_block(7, &[2_u8.pow(0)]), true);

        assert_eq!(bit_value_in_block(8, &[0, 2_u8.pow(7)]), true);
        assert_eq!(bit_value_in_block(0, &[0, 2_u8.pow(7)]), false);
        assert_eq!(bit_value_in_block(8, &[0, 0]), false);

        assert_eq!(
            bit_value_in_block(103, &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
            true
        );
    }

    #[test]
    fn test_transform_data() {
        assert_eq!(
            transform_data(&[vec![0, 0], vec![0, 1], vec![1, 0]]),
            Data {
                data: vec![
                    vec![0],
                    vec![0],
                    vec![0],
                    vec![0],
                    vec![0],
                    vec![0],
                    vec![0],
                    vec![4],
                    vec![0],
                    vec![0],
                    vec![0],
                    vec![0],
                    vec![0],
                    vec![0],
                    vec![0],
                    vec![2]
                ],
                _mask: 7,
                _num_of_blocks: 3
            }
        )
    }
}

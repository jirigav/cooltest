use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(not(feature = "scipy"))]
use binomtest::*;

#[cfg(feature = "scipy")]
use pyo3::prelude::*;

pub(crate) type Res<T> = Result<T, Box<dyn Error>>;

static QUIET: AtomicBool = AtomicBool::new(false);

pub(crate) fn set_quiet(quiet: bool) {
    QUIET.store(quiet, Ordering::Relaxed);
}

pub(crate) fn is_quiet() -> bool {
    QUIET.load(Ordering::Relaxed)
}

/// Prints progress and status information to stderr, so that stdout carries only results.
/// Suppressed entirely by `--quiet`.
macro_rules! status {
    ($($arg:tt)*) => {
        if !$crate::common::is_quiet() {
            eprintln!($($arg)*);
        }
    };
}
pub(crate) use status;

#[derive(Parser, Debug, Serialize, Deserialize, Clone)]
#[command(version)]
pub(crate) struct Args {
    /// Path of file with input data.
    pub(crate) data_source: String,

    /// Length of block of data.
    #[arg(short, long, default_value_t = 128, value_parser = parse_block_size)]
    // Changing the default value changes autotest
    pub(crate) block: usize,

    /// Number of bits in histograms in brute-force search.
    #[arg(short, long, default_value_t = 2, value_parser = parse_k)]
    pub(crate) k: usize,

    /// Significance level
    #[arg(short, long, default_value_t = 0.0001, value_parser = parse_alpha)]
    pub(crate) alpha: f64,

    /// Number of threads for multi-thread run. 0 uses an optimized single-thread algorithm (recommended for small inputs). 1 uses the multi-thread code with a single thread. Values ≥2 run in parallel.
    #[arg(short, long, default_value_t = 0)]
    pub(crate) threads: usize,

    /// Path where json output should be stored. If no path provided, json output is not stored.
    #[arg(short, long)]
    pub(crate) json: Option<String>,

    /// Suppress progress bars and status messages; print only the results.
    #[arg(short, long, default_value_t = false)]
    pub(crate) quiet: bool,

    /// Exit with status 1 when the randomness hypothesis is rejected.
    #[arg(long, default_value_t = false)]
    pub(crate) exit_code: bool,

    #[clap(subcommand)]
    pub subcommand: Option<SubCommand>,
}

#[derive(Parser, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub(crate) enum SubCommand {
    /// Evaluate a given distinguisher on given data and report p-value.
    ///
    /// The whole input file is used for evaluation. To keep the p-value valid, the file must be
    /// disjoint from the data the distinguisher was trained on.
    Evaluate {
        /// Path of file with distinguisher which should be evaluated.
        #[arg(short, long)]
        dis_path: String,
    },

    /// Test data automatically with multiple configurations, optionally with the use of user-provided block size (-b)
    Autotest {},
}

fn parse_block_size(s: &str) -> Result<usize, String> {
    let block: usize = s.parse().map_err(|e| format!("{e}"))?;
    if block == 0 {
        return Err("Block size must be positive".to_string());
    }
    if !block.is_multiple_of(8) {
        return Err("Block size must be divisible by 8".to_string());
    }
    Ok(block)
}

/// Upper bound on `k`. The search allocates `2^k` histogram bins per candidate, so anything near
/// this limit is already impractical
pub(crate) const MAX_K: usize = 20;

fn parse_k(s: &str) -> Result<usize, String> {
    let k: usize = s.parse().map_err(|e| format!("{e}"))?;
    if k == 0 {
        return Err("k must be at least 1".to_string());
    }
    if k > MAX_K {
        return Err(format!("k must be at most {MAX_K}, typical values are 1..=4").to_string());
    }
    Ok(k)
}

fn parse_alpha(s: &str) -> Result<f64, String> {
    let alpha: f64 = s.parse().map_err(|e| format!("{e}"))?;
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err("Significance level must be in the open interval (0, 1)".to_string());
    }
    Ok(alpha)
}

/// Checks the argument combinations that a per-argument parser cannot see.
pub(crate) fn validate_args(args: &Args) -> Res<()> {
    if args.k > args.block {
        return Err(format!(
            "k = {} is larger than block size {}: a block does not have that many bits",
            args.k, args.block
        )
        .into());
    }
    Ok(())
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
    /// `data[bit][word]` holds the value of `bit` for 128 consecutive blocks, one per bit of the
    /// `u128`.
    pub(crate) data: Vec<Vec<u128>>,
    /// Marks the valid bits of the final word, which is only partially filled unless the block
    /// count is a multiple of 128.
    pub(crate) mask: u128,
    pub(crate) num_of_blocks: usize,
}

pub(crate) fn multi_eval(bits: &[usize], data: &Data) -> usize {
    let mut result = vec![u128::MAX; data.data[0].len()];

    for b in bits.iter() {
        for (r, d) in result.iter_mut().zip(&data.data[*b]) {
            *r &= d;
        }
    }

    result
        .iter()
        .map(|x| x.count_ones() as usize)
        .sum::<usize>()
}

pub(crate) fn load_data(path: &str) -> Res<Vec<u8>> {
    status!("Loading data from {path}...");
    let data = fs::read(path).map_err(|e| format!("failed to read '{path}': {e}"))?;
    status!("Data loaded, {} bytes.", data.len());
    Ok(data)
}

/// Truncates `raw` to a whole number of blocks.
/// If training is set to true, the data is split into training and testing halves, otherwise the whole data is used for testing.
pub(crate) fn whole_blocks(raw: &[u8], block_size: usize, training: bool) -> Res<(&[u8], &[u8])> {
    let block_bytes = block_size / 8;
    let n_blocks = raw.len() / block_bytes;
    if n_blocks < 2 {
        return Err(format!(
            "input is {} bytes, which is less than two {block_size}-bit blocks",
            raw.len()
        )
        .into());
    }
    if !raw.len().is_multiple_of(block_bytes) {
        status!("Data are not aligned with block size, dropping last block!");
    }
    status!("Block size {block_size}, {n_blocks} blocks.");
    if training {
        Ok((&raw[..(n_blocks / 2) * block_bytes], &raw[(n_blocks / 2) * block_bytes..n_blocks * block_bytes]))
    } else {
        Ok((&raw[..n_blocks * block_bytes], &[]))
    }
}

/// Smallest number of testing blocks that could ever produce a p-value below `alpha`.
///
/// The most extreme outcome puts every block in the selected bin set, giving a p-value of `p^n`
/// where `p >= 1/2` is unknown before the search. Using the most favourable `p = 1/2` yields a
/// lower bound on the required sample size.
pub(crate) fn min_blocks_for_alpha(alpha: f64) -> usize {
    (1.0 / alpha).log2().ceil() as usize
}

pub(crate) fn warn_if_underpowered(testing_blocks: usize, alpha: f64) {
    let needed = min_blocks_for_alpha(alpha);
    if testing_blocks < needed {
        status!(
            "WARNING: only {testing_blocks} testing blocks. At alpha = {alpha:.0e} the randomness \
             hypothesis cannot be rejected with fewer than {needed} blocks, no matter what the \
             data look like."
        );
    }
}

pub(crate) fn new_progress_bar(total: u64, prefix: &str) -> ProgressBar {
    if is_quiet() {
        return ProgressBar::hidden();
    }
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template(
            "{prefix} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
        )
        .unwrap()
        .progress_chars("#>-"),
    );
    pb.set_prefix(prefix.to_string());
    pb
}

/// Number of iterations between progress-bar updates.
pub(crate) const PROGRESS_STEP: u64 = 4096;

/// Transforms blocks into a bit-sliced layout: `data[i]` holds bit `i` of 128 consecutive blocks
/// packed into each `u128`, so a candidate can be evaluated on 128 blocks per instruction.
pub(crate) fn transform_data(data: &[u8], block_size: usize) -> Data {
    let block_bytes = block_size / 8;
    let n_blocks = data.len() / block_bytes;
    let n_words = n_blocks.div_ceil(128);

    let mut sliced = vec![vec![0_u128; n_words]; block_size];

    let pb = new_progress_bar(n_words as u64, "Transforming data");
    for (word, group) in data.chunks(block_bytes * 128).enumerate() {
        for (e, block) in group.chunks_exact(block_bytes).enumerate() {
            let block_bit = 1_u128 << e;
            for (byte_index, &byte) in block.iter().enumerate() {
                if byte == 0 {
                    continue;
                }
                let base = byte_index * 8;
                for offset in 0..8 {
                    if (byte >> (7 - offset)) & 1 == 1 {
                        sliced[base + offset][word] |= block_bit;
                    }
                }
            }
        }
        if (word as u64).is_multiple_of(PROGRESS_STEP) {
            pb.set_position(word as u64);
        }
    }
    pb.finish_and_clear();

    let mask = if n_blocks.is_multiple_of(128) {
        u128::MAX
    } else {
        (1_u128 << (n_blocks % 128)) - 1
    };

    Data {
        data: sliced,
        mask,
        num_of_blocks: n_blocks,
    }
}

pub(crate) fn z_score(sample_size: usize, positive: usize, p: f64) -> f64 {
    ((positive as f64) - p * (sample_size as f64)) / f64::sqrt(p * (1.0 - p) * (sample_size as f64))
}

#[cfg(feature = "scipy")]
pub(crate) fn p_value(sample_size: usize, positive: usize, probability: f64) -> f64 {
    Python::with_gil(|py| {
        let scipy_stats = PyModule::import(py, "scipy.stats")
            .expect("SciPy not installed! Use `pip install scipy` to install the library.");
        let result: f64 = scipy_stats
            .getattr("binomtest")
            .expect("Scipy binomtest not found! Make sure that your version of SciPy is >=1.7.0.")
            .call1((positive, sample_size, probability, "greater"))
            .unwrap()
            .getattr("pvalue")
            .unwrap()
            .extract()
            .unwrap();
        result
    })
}

#[cfg(not(feature = "scipy"))]
pub(crate) fn p_value(sample_size: usize, positive: usize, probability: f64) -> f64 {
    binomial_test(
        positive as u64,
        sample_size as u64,
        probability,
        Alternative::Greater,
    )
    .unwrap()
}

/// Deterministic pseudorandom bytes for tests. splitmix64 passes the standard batteries, so data
/// drawn from it should not be rejected; using a fixed generator keeps the statistical tests
/// reproducible without an rng dependency.
#[cfg(test)]
pub(crate) fn splitmix64_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    let mut out = Vec::with_capacity(len + 8);
    while out.len() < len {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        out.extend_from_slice(&z.to_le_bytes());
    }
    out.truncate(len);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() <= f64::max(b.abs() * 1e-6, 1e-12)
    }

    /// Reference values from `scipy.stats.binomtest(k, n, p, "greater").pvalue`.
    #[test]
    fn test_p_value() {
        // Single trial, exact by hand.
        assert!(approx_eq(p_value(1, 1, 0.0), 0.0));
        assert!(approx_eq(p_value(1, 1, 1.0), 1.0));
        assert!(approx_eq(p_value(1, 1, 0.5), 0.5));
        assert!(approx_eq(p_value(1, 1, 0.25), 0.25));

        // Upper tail: observed well above expectation.
        assert!(approx_eq(p_value(1245, 872, 0.51), 1.581616064854676e-42));
        assert!(approx_eq(p_value(2228, 1993, 0.61), 4.2305090286988354e-200));
        assert!(approx_eq(p_value(711, 342, 0.2), 4.111347623990106e-63));
        assert!(approx_eq(p_value(2295, 1187, 0.02), 0.0));

        // Mid-range, where an error in scale or continuity would show up.
        assert!(approx_eq(p_value(1000, 520, 0.5), 0.10872414660207075));
        assert!(approx_eq(p_value(1000, 510, 0.5), 0.2739863729617383));
        assert!(approx_eq(p_value(64, 20, 0.25), 0.15605708100416849));
        assert!(approx_eq(p_value(200, 60, 0.25), 0.06247223105646413));
        assert!(approx_eq(p_value(10000, 2560, 0.25), 0.08496741896324542));

        // Lower tail: a two-sided test would call these extreme, a one-sided "greater" must not.
        assert!(approx_eq(p_value(8064, 675, 0.85), 1.0));
        assert!(approx_eq(p_value(3952, 3009, 0.87), 1.0));
        assert!(approx_eq(p_value(7716, 969, 0.76), 1.0));
        assert!(approx_eq(p_value(4231, 1225, 0.75), 1.0));
        assert!(approx_eq(p_value(5936, 4649, 0.97), 1.0));
        assert!(approx_eq(p_value(6395, 1774, 0.32), 0.9999999999999359));
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
        assert!(bit_value_in_block(0, &[2_u8.pow(7)]));
        assert!(!bit_value_in_block(0, &[2_u8.pow(6)]));
        assert!(bit_value_in_block(1, &[2_u8.pow(6)]));
        assert!(bit_value_in_block(2, &[2_u8.pow(5)]));
        assert!(bit_value_in_block(3, &[2_u8.pow(4)]));
        assert!(bit_value_in_block(4, &[2_u8.pow(3)]));
        assert!(bit_value_in_block(5, &[2_u8.pow(2)]));
        assert!(bit_value_in_block(6, &[2_u8.pow(1)]));
        assert!(bit_value_in_block(7, &[2_u8.pow(0)]));

        assert!(bit_value_in_block(8, &[0, 2_u8.pow(7)]));
        assert!(!bit_value_in_block(0, &[0, 2_u8.pow(7)]));
        assert!(!bit_value_in_block(8, &[0, 0]));

        assert!(bit_value_in_block(
            103,
            &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ));
    }

    #[test]
    fn test_transform_data() {
        assert_eq!(
            transform_data(&[0, 0, 0, 1, 1, 0], 16),
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
                mask: 7,
                num_of_blocks: 3
            }
        )
    }

    /// The bit-sliced layout must agree with the direct per-block bit reads used at evaluation
    /// time, for block counts on both sides of the 128-block word boundary.
    #[test]
    fn test_transform_data_matches_direct_reads() {
        for block_size in [8, 16, 64] {
            let block_bytes = block_size / 8;
            for n_blocks in [1, 2, 127, 128, 129, 300] {
                let raw: Vec<u8> = (0..n_blocks * block_bytes)
                    .map(|i| (i as u64).wrapping_mul(0x9e3779b97f4a7c15) as u8)
                    .collect();
                let sliced = transform_data(&raw, block_size);

                assert_eq!(sliced.num_of_blocks, n_blocks);
                for (e, block) in raw.chunks_exact(block_size / 8).enumerate() {
                    for bit in 0..block_size {
                        let from_slice = (sliced.data[bit][e / 128] >> (e % 128)) & 1 == 1;
                        assert_eq!(from_slice, bit_value_in_block(bit, block));
                    }
                }

                // No bit may be set past the block count in the final word.
                for bit in 0..block_size {
                    let last = sliced.data[bit].last().unwrap();
                    assert_eq!(last & !sliced.mask, 0);
                }
            }
        }
    }

    #[test]
    fn test_split_data() {
        let raw: Vec<u8> = (0..100).collect();

        // 100 bytes, 2-byte blocks => 50 blocks, split 25/25.
        let (train, test) = whole_blocks(&raw, 16, true).unwrap();
        assert_eq!(train, &raw[..50]);
        assert_eq!(test, &raw[50..]);

        // Odd block count: the extra block goes to the testing half.
        let (train, test) = whole_blocks(&raw[..30], 32, true).unwrap();
        assert_eq!(train.len() + 4, test.len());

        // Trailing partial block is dropped rather than silently mis-parsed.
        let (train, test) = whole_blocks(&raw[..99], 16, true).unwrap();
        assert_eq!(train.len() + test.len(), 98);

        // Fewer than two blocks is an error, not a panic.
        assert!(whole_blocks(&raw[..15], 128, true).is_err());
        assert!(whole_blocks(&[], 8, true).is_err());
    }

    #[test]
    fn test_arg_validation() {
        assert!(parse_block_size("0").is_err());
        assert!(parse_block_size("12").is_err());
        assert!(parse_block_size("128").is_ok());
        assert!(parse_k("0").is_err());
        assert!(parse_k("1").is_ok());
        assert!(parse_k("25").is_err());
        assert!(parse_alpha("0").is_err());
        assert!(parse_alpha("1").is_err());
        assert!(parse_alpha("0.0001").is_ok());
    }

    #[test]
    fn test_min_blocks_for_alpha() {
        assert_eq!(min_blocks_for_alpha(0.5), 1);
        assert_eq!(min_blocks_for_alpha(0.0001), 14);
    }
}

use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use rayon::iter::*;

use crate::common::{bits_block_eval, multi_eval, transform_data, z_score, Data};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct Histogram {
    pub(crate) bits: Vec<usize>,
    pub(crate) sorted_indices: Vec<usize>,
    pub(crate) best_division: usize,
    #[serde(skip_serializing, default)]
    pub(crate) z_score: f64,
    pub(crate) block_size: usize,
}

impl Histogram {
    pub(crate) fn from_bins(bits: Vec<usize>, bins: &[usize], block_size: usize) -> Histogram {
        let mut indices = (0..2_usize.pow(bits.len() as u32)).collect_vec();
        indices.sort_by(|a, b| bins[*b].cmp(&bins[*a]));

        let mut max_z = 0.0;
        let mut best_i = 0;
        let prob = 2.0_f64.powf(-(bits.len() as f64));
        let total: usize = bins.iter().sum();
        let mut count = 0;

        for i in 1..2_usize.pow(bits.len() as u32) {
            count += bins[indices[i - 1]];
            let z = z_score(total, count, prob * (i as f64)).abs();
            if z > max_z {
                max_z = z;
                best_i = i;
            }
        }
        Histogram {
            bits: bits.to_vec(),
            sorted_indices: indices,
            best_division: best_i,
            z_score: max_z,
            block_size,
        }
    }

    pub(crate) fn evaluate(&self, data: &[Vec<u8>]) -> (usize, Vec<usize>) {
        let mut hist2 = vec![0; 2_usize.pow(self.bits.len() as u32)];
        for block in data {
            hist2[bits_block_eval(&self.bits, block)] += 1;
        }
        let mut count = 0;
        for k in 0..self.best_division {
            count += hist2[self.sorted_indices[k]];
        }
        (count, hist2)
    }
}

impl std::fmt::Debug for Histogram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Histogram")
            .field("bits", &self.bits)
            .field("sorted indices", &self.sorted_indices)
            .field("best_division", &self.best_division)
            .field("z_score", &self.z_score)
            .finish()
    }
}

fn first_zero_bit(k: usize) -> usize {
    (!k).trailing_zeros() as usize
}

fn choose(n: usize, r: usize) -> usize {
    if r > n {
        0
    } else {
        (1..=r).fold(1, |acc, val| acc * (n - val + 1) / val)
    }
}

fn compute_index(bits: &[usize], block_size: usize) -> usize {
    let mut result = 0;
    let mut j = 0;
    let mut i = 0;

    while i < block_size && j < bits.len() {
        if i < bits[j] {
            result += choose(block_size - i - 1, bits.len() - j - 1)
        } else {
            j += 1;
        }
        i += 1;
    }
    result
}

fn compute_bins(
    bits: &[usize],
    data: &Data,
    d: usize,
    hists: &[Vec<usize>],
    bins: &mut [usize],
    block_size: usize,
) {
    let ones = multi_eval(bits, data);

    let value = 2_usize.pow(d as u32) - 1;

    bins[value] = ones;

    let mut bits_without: Vec<Vec<usize>> = Vec::with_capacity(d);
    for ind in 0..d {
        let mut b = bits.to_owned();
        b.remove(ind);
        bits_without.push(b);
    }

    for k in (0..value).rev() {
        // find first zero in bin's index k and replace if with one. i.e. obtain index with distance 1 for which the bin value is already computed
        let mut k2 = k;
        let ind = first_zero_bit(k);
        k2 ^= 1 << ind; // flip bit to one

        let n = (k2 & ((1 << ind) - 1)) + ((k2 >> (ind + 1)) << ind); // remove ind-th bit from the number

        let prev = hists[compute_index(&bits_without[ind], block_size)][n]; // result from prev layer

        bins[k] = prev - bins[k2];
    }
}

fn new_progress_bar(total: u64, prefix: &str) -> ProgressBar {
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

fn brute_force(data: &Data, block_size: usize, k: usize) -> Histogram {
    println!("Searching for distinguisher...");
    let mut hists: Vec<Vec<usize>> = Vec::new();
    for i in 0..block_size {
        let ones = multi_eval(&[i], data);
        hists.push(vec![(data._num_of_blocks as usize) - ones, ones])
    }

    for d in 2..k {
        let total = choose(block_size, d) as u64;
        let pb = new_progress_bar(total, &format!("Distinguisher search (layer {d}/{k})"));
        let mut new_hists = Vec::with_capacity(2_usize.pow(k as u32));

        for bits in (0..block_size).combinations(d) {
            let mut bins = vec![0; 2_usize.pow(d as u32)];
            compute_bins(&bits, data, d, &hists, &mut bins, block_size);

            new_hists.push(bins);
            pb.inc(1);
        }
        pb.finish_and_clear();
        hists = new_hists;
    }
    if k > 1 {
        let total = choose(block_size, k) as u64;
        let pb = new_progress_bar(total, &format!("Distinguisher search (layer {k}/{k})"));
        let mut best_hist = Histogram::from_bins(vec![0], &[1, 1], block_size);
        let mut bins = vec![0; 2_usize.pow(k as u32)];
        for bits in (0..block_size).combinations(k) {
            compute_bins(&bits, data, k, &hists, &mut bins, block_size);
            let hist = Histogram::from_bins(bits, &bins, block_size);
            if hist.z_score.abs() > best_hist.z_score.abs() {
                best_hist = hist;
            }
            pb.inc(1);
        }
        pb.finish_and_clear();
        best_hist
    } else {
        let bits = (0..block_size).combinations(k).collect_vec();
        let mut best: Vec<_> = hists
            .into_iter()
            .enumerate()
            .map(|(i, bins)| Histogram::from_bins(bits[i].clone(), &bins, block_size))
            .collect();

        best.sort_by(|a, b| b.z_score.partial_cmp(&a.z_score).unwrap());
        best.into_iter().next().unwrap()
    }
}

pub(crate) fn bottomup(data: &[Vec<u8>], block_size: usize, k: usize, threads: usize) -> Histogram {
    let res = if threads == 0 {
        brute_force(&transform_data(data), block_size, k)
    } else {
        brute_force_threads(&transform_data(data), block_size, k, threads)
    };

    println!("Distinguisher: {:?}", res);
    res
}

pub(crate) fn multi_eval_neg(
    bits: &[usize],
    data: &Data,
    neg_data: &Data,
    mut negs: usize,
) -> usize {
    let mut result = vec![u128::MAX; data.data[0].len()];

    for b in bits.iter() {
        let src = if negs % 2 == 0 {
            &neg_data.data[*b]
        } else {
            &data.data[*b]
        };
        for (r, d) in result.iter_mut().zip(src) {
            *r &= d;
        }
        negs >>= 1;
    }

    result
        .iter()
        .map(|x| x.count_ones() as usize)
        .sum::<usize>()
}

fn brute_force_threads(data: &Data, block_size: usize, k: usize, threads: usize) -> Histogram {
    println!("Searching for distinguisher...");

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("Failed to create thread pool");

    let mut neg_data = data.clone();
    neg_data.data = neg_data
        .data
        .iter()
        .map(|x| x.iter().map(|a| a ^ u128::MAX).collect())
        .collect();
    neg_data.data.iter_mut().for_each(|x| {
        let l = x.len();
        x[l - 1] &= data._mask;
    });

    let total = choose(block_size, k) as u64;
    let pb = new_progress_bar(total, "Distinguisher search (threaded)");

    let best_hist = pool.install(|| {
        (0..block_size)
            .combinations(k)
            .par_bridge()
            .map(|bits| {
                let mut bins = vec![0; 2_usize.pow(k as u32)];
                for (i, bin) in bins.iter_mut().enumerate() {
                    *bin = multi_eval_neg(&bits, data, &neg_data, i);
                }
                pb.inc(1);
                Histogram::from_bins(bits, &bins, block_size)
            })
            .reduce(
                || Histogram::from_bins(vec![0], &[1, 1], block_size),
                |a, b| {
                    if a.z_score.abs() >= b.z_score.abs() {
                        a
                    } else {
                        b
                    }
                },
            )
    });
    pb.finish_and_clear();

    best_hist
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_choose() {
        assert_eq!(choose(0, 0), 1);
        assert_eq!(choose(5, 6), 0);
        assert_eq!(choose(8, 6), 28);
        assert_eq!(choose(1, 1), 1);
        assert_eq!(choose(50, 33), 9847379391150);
        assert_eq!(choose(14, 9), 2002);
    }

    #[test]
    fn test_first_zero_bit() {
        assert_eq!(first_zero_bit(0b0), 0);
        assert_eq!(first_zero_bit(0b000010), 0);
        assert_eq!(first_zero_bit(0b1010101010), 0);
        assert_eq!(first_zero_bit(0b101), 1);
        assert_eq!(first_zero_bit(0b1011), 2);
        assert_eq!(first_zero_bit(0b10111), 3);
        assert_eq!(first_zero_bit(0b101111), 4);
        assert_eq!(first_zero_bit(0b1011111), 5);
        assert_eq!(first_zero_bit(0b10111111), 6);
        assert_eq!(first_zero_bit(0b101111111), 7);
        assert_eq!(first_zero_bit(0b1011111111), 8);
        assert_eq!(first_zero_bit(0b10111111111), 9);
        assert_eq!(first_zero_bit(0b101111111111), 10);
    }
}

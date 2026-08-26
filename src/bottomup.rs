use indicatif::ProgressBar;
use itertools::Itertools;
use rayon::iter::*;
use std::cmp::Ordering;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use crate::common::{
    bits_block_eval, multi_eval, new_progress_bar, status, transform_data, z_score, Data, Res,
    MAX_K, PROGRESS_STEP,
};
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
        let mut indices = (0..bins.len()).collect_vec();
        indices.sort_by(|a, b| bins[*b].cmp(&bins[*a]));

        let mut max_z = 0.0;

        let mut best_i = 1;
        let prob = 2.0_f64.powf(-(bits.len() as f64));
        let total: usize = bins.iter().sum();
        let mut count = 0;

        for i in 1..bins.len() {
            count += bins[indices[i - 1]];
            let z = z_score(total, count, prob * (i as f64)).abs();
            if z > max_z {
                max_z = z;
                best_i = i;
            }
        }
        debug_assert!(best_i >= 1 && best_i < bins.len().max(2));
        Histogram {
            bits,
            sorted_indices: indices,
            best_division: best_i,
            z_score: max_z,
            block_size,
        }
    }

    /// Rejects distinguishers that would index out of bounds during evaluation. Needed because a
    /// `Histogram` can arrive from a user-supplied JSON file rather than from the search.
    pub(crate) fn validate(&self) -> Res<()> {
        if self.block_size == 0 || !self.block_size.is_multiple_of(8) {
            return Err(format!("invalid block size {} in distinguisher", self.block_size).into());
        }
        if self.bits.is_empty() || self.bits.len() > MAX_K {
            return Err(format!(
                "distinguisher must select between 1 and {MAX_K} bits, got {}",
                self.bits.len()
            )
            .into());
        }
        if self.bits.windows(2).any(|w| w[0] >= w[1]) {
            return Err("distinguisher bits must be strictly increasing".into());
        }
        if let Some(bit) = self.bits.iter().find(|b| **b >= self.block_size) {
            return Err(format!(
                "distinguisher selects bit {bit}, outside a {}-bit block",
                self.block_size
            )
            .into());
        }
        let n_bins = 1_usize << self.bits.len();
        if self.sorted_indices.len() != n_bins {
            return Err(format!(
                "distinguisher has {} sorted indices, expected {n_bins}",
                self.sorted_indices.len()
            )
            .into());
        }
        let mut seen = vec![false; n_bins];
        for i in &self.sorted_indices {
            if *i >= n_bins || std::mem::replace(&mut seen[*i], true) {
                return Err(
                    "distinguisher sorted indices are not a permutation of the bins".into(),
                );
            }
        }
        if self.best_division == 0 || self.best_division >= n_bins {
            return Err(format!(
                "distinguisher division {} must be between 1 and {}",
                self.best_division,
                n_bins - 1
            )
            .into());
        }
        Ok(())
    }

    pub(crate) fn evaluate(&self, data: &[u8]) -> (usize, Vec<usize>) {
        let mut bins = vec![0; 1_usize << self.bits.len()];
        for block in data.chunks_exact(self.block_size / 8) {
            bins[bits_block_eval(&self.bits, block)] += 1;
        }
        let count = self.sorted_indices[..self.best_division]
            .iter()
            .map(|i| bins[*i])
            .sum();
        (count, bins)
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

/// Ranking key for candidate distinguishers. NaN (reachable only for an empty sample) sorts below
/// every real score instead of winning, as it would under `total_cmp`.
fn score(hist: &Histogram) -> f64 {
    let z = hist.z_score.abs();
    if z.is_nan() {
        f64::NEG_INFINITY
    } else {
        z
    }
}

/// True when `a` should beat `b`. Ties break on `bits` lexicographically rather than on arrival
/// order, so the parallel search is reproducible and agrees with the sequential one.
fn better(a: &Histogram, b: &Histogram) -> bool {
    match score(a).total_cmp(&score(b)) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => a.bits < b.bits,
    }
}

fn keep_best(a: Histogram, b: Histogram) -> Histogram {
    if better(&b, &a) {
        b
    } else {
        a
    }
}

fn first_zero_bit(k: usize) -> usize {
    (!k).trailing_zeros() as usize
}

/// Binomial coefficient
fn choose(n: usize, r: usize) -> usize {
    if r > n {
        return 0;
    }
    let mut acc: usize = 1;
    for val in 1..=r {
        acc = match acc.checked_mul(n - val + 1) {
            Some(v) => v / val,
            None => return usize::MAX,
        };
    }
    acc
}

/// Rank of `bits` in the lexicographic ordering of `block_size choose bits.len()` combinations,
/// matching the order `itertools::combinations` yields them in. This is what lets a layer address
/// the previous layer's histograms by index rather than by search.
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

fn brute_force(data: &Data, block_size: usize, k: usize) -> Option<Histogram> {
    status!("Searching for distinguisher...");
    let mut hists: Vec<Vec<usize>> = Vec::new();
    for i in 0..block_size {
        let ones = multi_eval(&[i], data);
        hists.push(vec![data.num_of_blocks - ones, ones])
    }

    for d in 2..k {
        let total = choose(block_size, d) as u64;
        let pb = new_progress_bar(total, &format!("Distinguisher search (layer {d}/{k})"));
        let mut new_hists = Vec::with_capacity(total as usize);

        for (i, bits) in (0..block_size).combinations(d).enumerate() {
            let mut bins = vec![0; 2_usize.pow(d as u32)];
            compute_bins(&bits, data, d, &hists, &mut bins, block_size);

            new_hists.push(bins);
            if (i as u64).is_multiple_of(PROGRESS_STEP) {
                pb.set_position(i as u64);
            }
        }
        pb.finish_and_clear();
        hists = new_hists;
    }
    if k > 1 {
        let total = choose(block_size, k) as u64;
        let pb = new_progress_bar(total, &format!("Distinguisher search (layer {k}/{k})"));
        let mut best_hist: Option<Histogram> = None;
        let mut bins = vec![0; 2_usize.pow(k as u32)];
        for (i, bits) in (0..block_size).combinations(k).enumerate() {
            compute_bins(&bits, data, k, &hists, &mut bins, block_size);
            let hist = Histogram::from_bins(bits, &bins, block_size);
            if best_hist.as_ref().is_none_or(|best| better(&hist, best)) {
                best_hist = Some(hist);
            }
            if (i as u64).is_multiple_of(PROGRESS_STEP) {
                pb.set_position(i as u64);
            }
        }
        pb.finish_and_clear();
        best_hist
    } else {
        (0..block_size)
            .combinations(k)
            .zip(hists)
            .map(|(bits, bins)| Histogram::from_bins(bits, &bins, block_size))
            .reduce(keep_best)
    }
}

pub(crate) fn multi_eval_neg(
    bits: &[usize],
    data: &Data,
    neg_data: &Data,
    mut negs: usize,
) -> usize {
    let mut result = vec![u128::MAX; data.data[0].len()];

    for b in bits.iter() {
        let src = if negs.is_multiple_of(2) {
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

/// Bitwise complement of `data`, with the padding past the final block cleared so that complemented
/// bits of nonexistent blocks are not counted.
pub(crate) fn negate(data: &Data) -> Data {
    let mut neg = data.clone();
    neg.data = neg
        .data
        .iter()
        .map(|x| x.iter().map(|a| a ^ u128::MAX).collect())
        .collect();
    neg.data.iter_mut().for_each(|x| {
        if let Some(last) = x.last_mut() {
            *last &= data.mask;
        }
    });
    neg
}

fn brute_force_threads(
    data: &Data,
    block_size: usize,
    k: usize,
    threads: usize,
) -> Option<Histogram> {
    status!("Searching for distinguisher...");

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("Failed to create thread pool");

    let neg_data = negate(data);

    let total = choose(block_size, k) as u64;
    let pb = new_progress_bar(total, "Distinguisher search (threaded)");
    let done = AtomicU64::new(0);

    let best_hist = pool.install(|| {
        (0..block_size)
            .combinations(k)
            .par_bridge()
            .map(|bits| {
                let mut bins = vec![0; 2_usize.pow(k as u32)];
                for (i, bin) in bins.iter_mut().enumerate() {
                    *bin = multi_eval_neg(&bits, data, &neg_data, i);
                }
                advance(&pb, &done);
                Histogram::from_bins(bits, &bins, block_size)
            })
            .reduce_with(keep_best)
    });
    pb.finish_and_clear();

    best_hist
}

/// Batches progress updates: each `ProgressBar::inc` takes a lock, which under `par_bridge` is
/// contended by every worker on every candidate.
fn advance(pb: &ProgressBar, done: &AtomicU64) {
    if done
        .fetch_add(1, AtomicOrdering::Relaxed)
        .is_multiple_of(PROGRESS_STEP)
    {
        pb.inc(PROGRESS_STEP);
    }
}

pub(crate) fn bottomup(data: &[u8], block_size: usize, k: usize, threads: usize) -> Res<Histogram> {
    let transformed = transform_data(data, block_size);

    // The two searches compute the same histograms with opposite trade-offs. The incremental one
    // derives 2^k - 1 of every candidate's bins by subtraction from the previous layer, so it
    // touches the data 2^k times less often, but must keep that layer in memory. The direct one
    // stores nothing and re-scans once per bin, so it only pays off past 2^k threads.
    let res = if threads == 0 {
        brute_force(&transformed, block_size, k)
    } else {
        brute_force_threads(&transformed, block_size, k, threads)
    };

    let res = res.ok_or_else(|| {
        format!("no distinguisher found: there are no {k}-bit subsets of a {block_size}-bit block")
    })?;

    status!("Distinguisher: {res:?}");
    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{splitmix64_bytes, transform_data};

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
    fn test_choose_saturates_instead_of_wrapping() {
        // C(4096, 20) is far past usize::MAX; the old unchecked multiply wrapped to a small,
        // plausible-looking number.
        assert_eq!(choose(4096, 20), usize::MAX);
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

    /// `compute_index` must reproduce the position at which `itertools::combinations` yields each
    /// combination; the layered search addresses the previous layer by that index alone.
    #[test]
    fn test_compute_index_matches_combination_order() {
        for block_size in [4, 8, 16, 32] {
            for k in 1..=4.min(block_size) {
                for (expected, bits) in (0..block_size).combinations(k).enumerate() {
                    assert_eq!(
                        compute_index(&bits, block_size),
                        expected,
                        "block_size={block_size} k={k} bits={bits:?}"
                    );
                }
            }
        }
    }

    /// Counts bins one block at a time. Deliberately naive: this is the specification that both
    /// search algorithms must reproduce.
    fn reference_bins(bits: &[usize], data: &[u8], block_size: usize) -> Vec<usize> {
        let mut bins = vec![0; 1_usize << bits.len()];
        for block in data.chunks_exact(block_size / 8) {
            bins[bits_block_eval(bits, block)] += 1;
        }
        bins
    }

    /// Cross-checks the bin-subtraction recursion and the direct complement-mask evaluation against
    /// the naive reference, over block counts on both sides of the 128-block word boundary. This
    /// covers `compute_index`, `first_zero_bit`, and the `compute_bins` recursion at once, and pins
    /// the two search paths to each other.
    #[test]
    fn test_bins_match_reference() {
        for block_size in [8, 16, 32] {
            let max_k = if block_size <= 16 { 4 } else { 3 };
            for n_blocks in [1, 5, 127, 128, 129] {
                let raw = splitmix64_bytes(n_blocks * block_size / 8, 0xC001_D00D);
                let data = transform_data(&raw, block_size);
                let neg_data = negate(&data);

                let mut hists: Vec<Vec<usize>> = (0..block_size)
                    .map(|i| {
                        let ones = multi_eval(&[i], &data);
                        vec![data.num_of_blocks - ones, ones]
                    })
                    .collect();

                for (i, bins) in hists.iter().enumerate() {
                    assert_eq!(
                        *bins,
                        reference_bins(&[i], &raw, block_size),
                        "layer 1 mismatch: block_size={block_size} n_blocks={n_blocks} bit={i}"
                    );
                }

                for d in 2..=max_k {
                    let mut new_hists = Vec::new();
                    for bits in (0..block_size).combinations(d) {
                        let expected = reference_bins(&bits, &raw, block_size);

                        let mut bins = vec![0; 1_usize << d];
                        compute_bins(&bits, &data, d, &hists, &mut bins, block_size);
                        assert_eq!(
                            bins, expected,
                            "compute_bins: block_size={block_size} n_blocks={n_blocks} bits={bits:?}"
                        );

                        let direct: Vec<usize> = (0..1_usize << d)
                            .map(|i| multi_eval_neg(&bits, &data, &neg_data, i))
                            .collect();
                        assert_eq!(
                            direct, expected,
                            "multi_eval_neg: block_size={block_size} n_blocks={n_blocks} bits={bits:?}"
                        );

                        new_hists.push(bins);
                    }
                    hists = new_hists;
                }
            }
        }
    }

    #[test]
    fn test_from_bins_never_selects_an_empty_division() {
        // Flat bins give every division a z-score of exactly zero; the division still has to be a
        // real one, or evaluation reports a verdict drawn from no bins.
        for k in 1..=4 {
            let bins = vec![10_usize; 1 << k];
            let hist = Histogram::from_bins((0..k).collect(), &bins, 64);
            assert!(hist.best_division >= 1);
            assert!(hist.best_division < bins.len());
            hist.validate().unwrap();
        }
        // An empty sample makes every z-score NaN.
        let hist = Histogram::from_bins(vec![0, 1], &[0, 0, 0, 0], 64);
        assert_eq!(hist.best_division, 1);
        hist.validate().unwrap();
    }

    #[test]
    fn test_better_breaks_ties_deterministically() {
        let a = Histogram::from_bins(vec![0, 1], &[4, 3, 2, 1], 64);
        let b = Histogram::from_bins(vec![0, 2], &[4, 3, 2, 1], 64);
        assert_eq!(a.z_score, b.z_score);
        // Equal scores: the lexicographically smaller bit set wins, whichever way round it is asked.
        assert!(better(&a, &b));
        assert!(!better(&b, &a));
        assert_eq!(keep_best(a.clone(), b.clone()).bits, vec![0, 1]);
        assert_eq!(keep_best(b, a).bits, vec![0, 1]);
    }

    #[test]
    fn test_validate_rejects_malformed_distinguishers() {
        let good = Histogram::from_bins(vec![0, 1], &[9, 5, 3, 1], 64);
        good.validate().unwrap();

        let mut h = good.clone();
        h.best_division = 0;
        assert!(h.validate().is_err());

        let mut h = good.clone();
        h.best_division = 4;
        assert!(h.validate().is_err());

        let mut h = good.clone();
        h.sorted_indices = vec![0, 0, 1, 2];
        assert!(h.validate().is_err());

        let mut h = good.clone();
        h.sorted_indices = vec![0, 1, 2, 9];
        assert!(h.validate().is_err());

        let mut h = good.clone();
        h.bits = vec![1, 0];
        assert!(h.validate().is_err());

        let mut h = good.clone();
        h.bits = vec![0, 64];
        assert!(h.validate().is_err());

        let mut h = good.clone();
        h.block_size = 12;
        assert!(h.validate().is_err());
    }

    /// Both searches must return the same distinguisher, including on ties.
    #[test]
    fn test_search_paths_agree() {
        for block_size in [8, 16, 32] {
            for k in 1..=3.min(block_size) {
                for seed in [1_u64, 42, 12345] {
                    let raw = splitmix64_bytes(200 * block_size / 8, seed);
                    let data = transform_data(&raw, block_size);

                    let sequential = brute_force(&data, block_size, k).unwrap();
                    let parallel = brute_force_threads(&data, block_size, k, 4).unwrap();

                    assert_eq!(
                        sequential.bits, parallel.bits,
                        "block_size={block_size} k={k} seed={seed}"
                    );
                    assert_eq!(sequential.best_division, parallel.best_division);
                    assert_eq!(sequential.sorted_indices, parallel.sorted_indices);
                    assert_eq!(sequential.z_score, parallel.z_score);
                }
            }
        }
    }

    #[test]
    fn test_no_candidates_returns_none() {
        let raw = splitmix64_bytes(64, 7);
        let data = transform_data(&raw, 8);
        assert!(brute_force(&data, 8, 9).is_none());
        assert!(brute_force_threads(&data, 8, 9, 2).is_none());
        assert!(bottomup(&raw, 8, 9, 0).is_err());
    }
}

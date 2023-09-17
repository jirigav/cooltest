use crate::common::{multi_eval, multi_eval_count, z_score, Data};
use core::fmt;
use itertools::Itertools;
use rayon::prelude::*;
use std::{collections::HashSet, fmt::Debug};

#[derive(Debug, Clone)]
pub(crate) struct Histogram {
    pub(crate) bits: Vec<usize>,
    pub(crate) _bins: Vec<usize>,
    pub(crate) sorted_indices: Vec<usize>,
    pub(crate) best_division: usize,
    pub(crate) z_score: f64,
}

impl Histogram {
    pub(crate) fn get_hist(bits: &Vec<usize>, data: &Data) -> Histogram {
        let mut hist = vec![0; 2_usize.pow(bits.len() as u32)];

        let mut it = data.data.iter().peekable();
        while let Some(blocks) = it.next() {
            let is_last = it.peek().is_none();
            hist.par_iter_mut().enumerate().for_each(|(i, h)| {
                *h += multi_eval_count(i, bits, blocks, data.mask, is_last) as usize;
            })
        }

        let mut indices = (0..2_usize.pow(bits.len() as u32)).collect_vec();
        indices.sort_by(|a, b| hist[*b].cmp(&hist[*a]));

        let mut max_z = 0.0;
        let mut best_i = 0;
        let prob = 2.0_f64.powf(-(bits.len() as f64));

        for i in 1..2_usize.pow(bits.len() as u32) {
            let mut count = 0;
            for k in 0..i {
                count += hist[indices[k]];
            }
            let z = z_score(data.num_of_blocks, count, prob * (i as f64));
            if z > max_z {
                max_z = z;
                best_i = i;
            }
        }
        Histogram {
            bits: bits.to_vec(),
            _bins: hist,
            sorted_indices: indices,
            best_division: best_i,
            z_score: max_z,
        }
    }

    pub(crate) fn evaluate(&self, data: &Data) -> usize {
        let mut hist2 = vec![0; 2_usize.pow(self.bits.len() as u32)];

        let mut it = data.data.iter().peekable();
        while let Some(blocks) = it.next() {
            let is_last = it.peek().is_none();
            hist2.par_iter_mut().enumerate().for_each(|(i, h)| {
                *h += multi_eval_count(i, &self.bits, blocks, data.mask, is_last) as usize;
            })
        }
        let mut count = 0;
        for k in 0..self.best_division {
            count += hist2[self.sorted_indices[k]];
        }
        count
    }
}

pub(crate) trait Distinguisher {
    fn evaluate(&self, blocks: &[u128], mask: u128, is_last: bool) -> u32;

    fn forget_count(&mut self);

    fn increase_count(&mut self, n: usize);

    fn get_count(&self) -> usize;

    fn z_score(&mut self, samples: usize) -> f64;

    fn get_z_score(&self) -> Option<f64>;
}

#[derive(Debug, Clone)]
pub(crate) struct Pattern {
    pub(crate) length: usize,
    pub(crate) bits: Vec<usize>,
    pub(crate) values: Vec<bool>,
    pub(crate) count: Option<usize>,
    pub(crate) z_score: Option<f64>,
    pub(crate) validation_z_score: Option<f64>,
    pub(super) bits_signs: usize,
}

impl Pattern {
    pub(crate) fn add_bit(&mut self, bit: usize, value: bool) {
        if self.bits.contains(&bit) {
            assert_eq!(
                value,
                self.values[self.bits.iter().position(|x| *x == bit).unwrap()]
            );
        } else {
            self.length += 1;

            let mut bits_values = self
                .bits
                .clone()
                .into_iter()
                .zip(self.values.clone())
                .collect::<Vec<_>>();
            bits_values.push((bit, value));
            bits_values.sort_by(|a, b| a.0.cmp(&b.0));
            (self.bits, self.values) = bits_values.into_iter().unzip();
            self.bits_signs = self
                .values
                .iter()
                .enumerate()
                .map(|(i, v)| if *v { 2_usize.pow(i as u32) } else { 0 })
                .sum();
            self.count = None;
            self.z_score = None;
        }
    }

    pub(crate) fn evaluate_raw(&self, blocks: &[u128], mask: u128, is_last: bool) -> u128 {
        multi_eval(self.bits_signs, &self.bits, blocks, mask, is_last)
    }
}

impl Distinguisher for Pattern {
    fn evaluate(&self, blocks: &[u128], mask: u128, is_last: bool) -> u32 {
        multi_eval_count(self.bits_signs, &self.bits, blocks, mask, is_last)
    }

    fn z_score(&mut self, samples: usize) -> f64 {
        let p = 2.0_f64.powf(-(self.length as f64));
        assert!((0.0..=1.0).contains(&p));
        let z = z_score(samples, self.count.unwrap(), p);
        self.z_score = Some(z);
        z
    }

    fn forget_count(&mut self) {
        self.count = None;
        self.z_score = None;
        self.validation_z_score = None;
    }

    fn increase_count(&mut self, n: usize) {
        if let Some(count) = self.count {
            self.count = Some(count + n);
        } else {
            self.count = Some(n);
        }
    }

    fn get_count(&self) -> usize {
        if let Some(count) = self.count {
            count
        } else {
            0
        }
    }

    fn get_z_score(&self) -> Option<f64> {
        self.z_score
    }
}

impl PartialEq for Pattern {
    fn eq(&self, other: &Self) -> bool {
        self.bits == other.bits && self.values == other.values
    }
}

impl fmt::Display for Pattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bits: {:?}\nValues: {:?}\n", self.bits, self.values)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MultiPattern {
    patterns: Vec<Pattern>,
    pub(crate) probability: f64,
    pub(crate) z_score: Option<f64>,
    count: Option<usize>,
}

fn union_probability(patterns: &[&Pattern]) -> f64 {
    if any_pairwise_disjoint_patterns(patterns) {
        return 0.0;
    }
    let bits: HashSet<usize> = patterns
        .iter()
        .flat_map(|x| x.bits.iter().copied())
        .collect();
    2.0_f64.powf(-(bits.len() as f64))
}

impl MultiPattern {
    pub(crate) fn new(patterns: Vec<Pattern>) -> MultiPattern {
        let mut probability: f64 = 0.0;
        let mut sgn = 1.0;
        for i in 1..=patterns.len() {
            for ps in patterns.iter().combinations(i) {
                probability += sgn * union_probability(&ps);
            }
            sgn *= -1.0;
        }
        MultiPattern {
            patterns,
            probability,
            z_score: None,
            count: None,
        }
    }
}

impl Distinguisher for MultiPattern {
    fn evaluate(&self, blocks: &[u128], mask: u128, is_last: bool) -> u32 {
        self.patterns
            .iter()
            .map(|p| p.evaluate_raw(blocks, mask, is_last))
            .fold(u128::MIN, |acc, x| acc | x)
            .count_ones()
    }

    fn forget_count(&mut self) {
        self.count = None;
    }

    fn increase_count(&mut self, n: usize) {
        if let Some(count) = self.count {
            self.count = Some(count + n);
        } else {
            self.count = Some(n);
        }
    }

    fn get_count(&self) -> usize {
        if let Some(count) = self.count {
            count
        } else {
            0
        }
    }

    fn z_score(&mut self, sample_size: usize) -> f64 {
        self.z_score = Some(z_score(sample_size, self.get_count(), self.probability));
        self.z_score.unwrap()
    }

    fn get_z_score(&self) -> Option<f64> {
        self.z_score
    }
}

pub(crate) fn best_multi_pattern(data: &Data, patterns: &[Pattern], n: usize) -> MultiPattern {
    let mut best_mp: Option<MultiPattern> = None;
    let mut max_z = 0.0;
    let patterns_sets = patterns.iter().combinations(n).collect_vec();

    let mut mps = patterns_sets
        .par_iter()
        .map(|ps| MultiPattern::new(ps.iter().map(|x| x.to_owned().clone()).collect_vec()))
        .collect::<Vec<_>>();

    let mut it = data.data.iter().peekable();
    while let Some(blocks) = it.next() {
        let is_last = it.peek().is_none();
        mps.par_iter_mut()
            .for_each(|mp| mp.increase_count(mp.evaluate(blocks, data.mask, is_last) as usize));
    }
    for mut mp in mps {
        let z = mp.z_score(data.num_of_blocks);
        if f64::abs(z) > f64::abs(max_z) {
            best_mp = Some(mp);
            max_z = z;
        }
    }
    best_mp.unwrap()
}

fn any_pairwise_disjoint_patterns(patterns: &[&Pattern]) -> bool {
    for ps in patterns.iter().combinations(2) {
        let mut disjoint_pair = false;
        for (i, b1) in ps[0].bits.iter().enumerate() {
            for (j, b2) in ps[1].bits.iter().enumerate() {
                if b1 == b2 && ps[0].values[i] != ps[1].values[j] {
                    disjoint_pair = true;
                    break;
                }
            }
            if disjoint_pair {
                break;
            }
        }
        if disjoint_pair {
            return true;
        }
    }
    false
}

pub(crate) fn evaluate_distinguisher<P: Distinguisher + ?Sized>(
    distinguisher: &mut P,
    data: &Data,
) -> f64 {
    let len_m1 = data.data.len() - 1;
    distinguisher.forget_count();
    distinguisher.increase_count(
        data.data
            .iter()
            .enumerate()
            .map(|(i, blocks)| distinguisher.evaluate(blocks, data.mask, i == len_m1))
            .sum::<u32>() as usize,
    );
    distinguisher.z_score(data.num_of_blocks)
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand::Rng;

    use crate::common::transform_data;

    use super::Distinguisher;
    use super::MultiPattern;
    use super::Pattern;

    #[test]
    fn double_pattern_probability() {
        let dp = MultiPattern::new(
            [
                Pattern {
                    length: 2,
                    bits: vec![1, 3],
                    values: vec![true, false],
                    count: None,
                    z_score: None,
                    validation_z_score: None,
                    bits_signs: 1,
                },
                Pattern {
                    length: 3,
                    bits: vec![0, 1, 2],
                    values: vec![true, true, true],
                    count: None,
                    z_score: None,
                    validation_z_score: None,
                    bits_signs: 7,
                },
            ]
            .to_vec(),
        );
        assert_eq!(5.0 / 16.0, dp.probability);
    }

    #[test]
    fn multi_pattern_probability() {
        for _i in 0..100 {
            for k in 1..10 {
                let mut patterns: Vec<Pattern> = Vec::new();
                for _p in 0..k {
                    let n_bits = rand::thread_rng().gen_range(1..=16);
                    let mut bits: Vec<usize> = Vec::new();
                    let mut values: Vec<bool> = Vec::new();
                    for _b in 0..n_bits {
                        let b = rand::thread_rng().gen_range(0..16);
                        if !bits.contains(&b) {
                            bits.push(b);
                            values.push(rand::random());
                        }
                    }
                    let p = Pattern {
                        length: bits.len(),
                        bits,
                        values: values.clone(),
                        count: None,
                        z_score: None,
                        validation_z_score: None,
                        bits_signs: values
                            .iter()
                            .enumerate()
                            .map(|(i, v)| if *v { 2_usize.pow(i as u32) } else { 0 })
                            .sum(),
                    };
                    patterns.push(p);
                }
                let mp = MultiPattern::new(patterns);
                let tr_data = transform_data(
                    (0..2_usize.pow(16))
                        .map(|x| x.to_le_bytes().to_vec())
                        .collect_vec(),
                );
                let len_m1 = tr_data.data.len() - 1;

                let count: u32 = tr_data
                    .data
                    .iter()
                    .enumerate()
                    .map(|(i, blocks)| mp.evaluate(&blocks, tr_data.mask, len_m1 == i))
                    .sum();

                assert_eq!(mp.probability, (count as f64) / 2.0_f64.powf(16.0));
            }
        }
    }
}

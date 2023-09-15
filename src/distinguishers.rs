use crate::common::{bits_block_eval, z_score, Data};
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
        for block in data {
            hist[bits_block_eval(bits, block)] += 1;
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
            let z = z_score(data.len(), count, prob * (i as f64));
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
        for block in data {
            hist2[bits_block_eval(&self.bits, block)] += 1;
        }
        let mut count = 0;
        for k in 0..self.best_division {
            count += hist2[self.sorted_indices[k]];
        }
        count
    }
}

pub(crate) trait Distinguisher {
    fn evaluate(&self, block: &[u8]) -> bool;

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
            self.bits_signs = self.values.iter().enumerate().map(|(i, v)| {if *v {2_usize.pow(i as u32)} else {0}}).sum();
            self.count = None;
            self.z_score = None;
        }
    }
}

impl Distinguisher for Pattern {
    fn evaluate(&self, block: &[u8]) -> bool {
        for (val, b) in self.values.iter().zip(&self.bits) {
            let (byte_index, offset) = (b / 8, b % 8);
            if u8::from(*val) != ((block[byte_index] >> offset) & 1) {
                return false;
            }
        }
        true
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
    fn evaluate(&self, block: &[u8]) -> bool {
        self.patterns.iter().any(|p| p.evaluate(block))
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
    for ps in patterns.iter().combinations(n) {
        let mut mp = MultiPattern::new(ps.iter().map(|x| x.to_owned().clone()).collect_vec());
        mp.count = Some(data.par_iter().filter(|block| mp.evaluate(block)).count());
        let z = mp.z_score(data.len());
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
    data: &[Vec<u8>],
) -> f64 {
    distinguisher.forget_count();
    distinguisher.increase_count(
        data.iter()
            .filter(|block| distinguisher.evaluate(block))
            .count(),
    );
    distinguisher.z_score(data.len())
}

#[cfg(test)]
mod tests {
    use rand::Rng;

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
                        bits_signs: values.iter().enumerate().map(|(i, v)| {if *v {2_usize.pow(i as u32)} else {0}}).sum(),
                    };
                    patterns.push(p);
                }
                let mp = MultiPattern::new(patterns);

                let count = (0..2_usize.pow(16))
                    .filter(|x| mp.evaluate(&x.to_le_bytes()))
                    .count();

                assert_eq!(mp.probability, (count as f64) / 2.0_f64.powf(16.0));
            }
        }
    }
}

use crate::common::*;
use rayon::prelude::*;
use std::{collections::HashSet, fmt::Debug};

pub(crate) trait GeneralizedPattern {
    fn evaluate(&self, block: &[u8]) -> bool;

    fn forget_count(&mut self);

    fn increase_count(&mut self, n: usize);

    fn get_count(&self) -> usize;

    fn z_score(&mut self, samples: usize) -> f64;

    fn get_z_score(&self) -> Option<f64>;
}

#[derive(Debug)]
pub(crate) struct Pattern {
    pub(crate) length: usize,
    pub(crate) bits: Vec<usize>,
    pub(crate) values: Vec<bool>,
    pub(crate) count: Option<usize>,
    pub(crate) z_score: Option<f64>,
}

impl Pattern {
    pub(crate) fn add_bit(&mut self, bit: usize, value: bool) {
        if self.bits.contains(&bit) {
            assert_eq!(
                value,
                self.values[self.bits.iter().position(|x| *x == bit).unwrap()]
            )
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
            self.count = None;
            self.z_score = None;
        }
    }
}

impl GeneralizedPattern for Pattern {
    fn evaluate(&self, block: &[u8]) -> bool {
        for (val, b) in self.values.iter().zip(&self.bits) {
            let (byte_index, offset) = (b / 8, b % 8);
            if (*val as u8) != ((block[byte_index] >> offset) & 1) {
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

impl Clone for Pattern {
    fn clone(&self) -> Self {
        Pattern {
            length: self.length,
            bits: self.bits.clone(),
            values: self.values.clone(),
            count: self.count,
            z_score: self.z_score,
        }
    }
}

impl PartialEq for Pattern {
    fn eq(&self, other: &Self) -> bool {
        self.bits == other.bits && self.values == other.values
    }
}

#[derive(Debug)]
pub(crate) struct DoublePattern {
    pattern1: Pattern,
    pattern2: Pattern,
    probability: f64,
    pub(crate) z_score: Option<f64>,
    count: Option<usize>,
}

impl DoublePattern {
    pub(crate) fn new(pattern1: Pattern, pattern2: Pattern) -> DoublePattern {
        let mut probability: f64 =
            2.0_f64.powf(-(pattern1.length as f64)) + 2.0_f64.powf(-(pattern2.length as f64));
        let mut disjoint = false;
        for (i, b1) in pattern1.bits.iter().enumerate() {
            for (j, b2) in pattern2.bits.iter().enumerate() {
                if b1 == b2 && pattern1.values[i] != pattern2.values[j] {
                    disjoint = true;
                    break;
                }
            }
            if disjoint {
                break;
            }
        }

        if !disjoint {
            let mut union = pattern1.bits.iter().collect::<HashSet<_>>();
            union.extend(pattern2.bits.iter());
            probability -= 2.0_f64.powf(-(union.len() as f64));
        }
        DoublePattern {
            pattern1,
            pattern2,
            probability,
            z_score: None,
            count: None,
        }
    }

    pub(crate) fn evaluate(&self, block: &[u8]) -> bool {
        self.pattern1.evaluate(block) || self.pattern2.evaluate(block)
    }
}

impl GeneralizedPattern for DoublePattern {
    fn evaluate(&self, block: &[u8]) -> bool {
        self.pattern1.evaluate(block) || self.pattern2.evaluate(block)
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

pub(crate) fn best_double_pattern(data: &[Vec<u8>], patterns: &[Pattern]) -> DoublePattern {
    let mut best_double_pattern: DoublePattern =
        DoublePattern::new(patterns[0].clone(), patterns[0].clone());
    best_double_pattern.z_score = Some(0.0);
    for i in 0..patterns.len() {
        for j in i..patterns.len() {
            let mut double_pattern = DoublePattern::new(patterns[i].clone(), patterns[j].clone());

            double_pattern.increase_count(
                data.par_iter()
                    .map(|block| double_pattern.evaluate(block))
                    .filter(|x| *x)
                    .count(),
            );

            let z = double_pattern.z_score(data.len());

            if f64::abs(z) > f64::abs(best_double_pattern.z_score.unwrap()) {
                best_double_pattern = double_pattern;
            }
        }
    }
    best_double_pattern
}

pub(crate) fn evaluate_pattern<P: GeneralizedPattern + ?Sized>(
    pattern: &mut P,
    data: &[Vec<u8>],
) -> f64 {
    pattern.forget_count();
    pattern.increase_count(
        data.iter()
            .map(|block| pattern.evaluate(block))
            .filter(|x| *x)
            .count(),
    );
    pattern.z_score(data.len())
}

#[cfg(test)]
mod tests {
    use super::DoublePattern;
    use super::Pattern;

    #[test]
    fn double_pattern_probability() {
        let dp = DoublePattern::new(
            Pattern {
                length: 2,
                bits: vec![1, 3],
                values: vec![true, false],
                count: None,
                z_score: None,
            },
            Pattern {
                length: 3,
                bits: vec![0, 1, 2],
                values: vec![true, true, true],
                count: None,
                z_score: None,
            },
        );
        assert_eq!(5.0 / 16.0, dp.probability);
    }
}

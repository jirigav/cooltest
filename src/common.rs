use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs::File;
use std::{fs, io::Write};
use clap::Parser;

//pub(crate) const LEN_OF_BLOCK_IN_BYTES: usize = 16;
//pub(crate) const LEN_OF_BLOCK: usize = LEN_OF_BLOCK_IN_BYTES * 8;

pub(crate) fn z_score(sample_size: usize, positive: usize, p: f64) -> f64 {
    ((positive as f64) - p * (sample_size as f64)) / f64::sqrt(p * (1.0 - p) * (sample_size as f64))
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub(crate) struct Args {
    
    #[arg(short, long)]
    pub(crate) data_source: String,

    #[arg(short, long)]
    pub(crate) block_size: usize,

    #[arg(short, long, default_value_t = 10)]
    pub(crate) k: usize,

    #[arg(short, long, default_value_t = 64)]
    pub(crate) n: usize,

    #[arg(short, long, default_value_t = 10)]
    pub(crate) min_count: usize,
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
    pub(crate) fn z(&mut self, samples: usize) -> f64 {
        let p = 2.0_f64.powf(-(self.length as f64));
        assert!((0.0..=1.0).contains(&p));
        let z = z_score(samples, self.count.unwrap(), p);
        self.z_score = Some(z);
        z
    }

    pub(crate) fn evaluate(&self, block: &[u8]) -> bool {
        for (val, b) in self.values.iter().zip(&self.bits) {
            let (byte_index, offset) = (b / 8, b % 8);
            if (*val as u8) != ((block[byte_index] >> offset) & 1) {
                return false;
            }
        }
        true
    }

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

#[derive(Debug, Clone)]
pub(crate) struct DoublePattern {
    pattern1: Pattern,
    pattern2: Pattern,
    probability: f64,
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
        }
    }

    pub(crate) fn evaluate(&self, block: &[u8]) -> bool {
        self.pattern1.evaluate(block) || self.pattern2.evaluate(block)
    }

    pub(crate) fn z_score(&self, sample_size: usize, positive: usize) -> f64 {
        z_score(sample_size, positive, self.probability)
    }
}

pub(crate) fn best_double_patterns(
    data: &[Vec<u8>],
    patterns: &[Pattern],
    k: usize,
) -> Vec<DoublePattern> {
    let mut double_patterns: Vec<(f64, DoublePattern)> = Vec::new();
    for i in 0..patterns.len() {
        for j in i..patterns.len() {
            let double_pattern = DoublePattern::new(patterns[i].clone(), patterns[j].clone());

            let count = data
                .par_iter()
                .map(|block| double_pattern.evaluate(block))
                .filter(|x| *x)
                .count();

            let z = double_pattern.z_score(data.len(), count);

            double_patterns.push((z, double_pattern));
        }
    }
    double_patterns.sort_by(|a, b| f64::abs(b.0).partial_cmp(&f64::abs(a.0)).unwrap());
    double_patterns
        .iter()
        .take(k)
        .map(|(_z, double_pattern)| double_pattern.clone())
        .collect()
}

pub(crate) fn bit_value_in_block(bit: &usize, block: &[u8]) -> bool {
    let (byte_index, offset) = (bit / 8, bit % 8);
    ((block[byte_index] >> offset) & 1) == 1
}

pub(crate) fn bits_block_eval(bits: Vec<usize>, block: &[u8]) -> usize {
    let mut result = 0;

    for (i, b) in bits.iter().enumerate() {
        if bit_value_in_block(b, block) {
            result += 2_usize.pow(i as u32)
        }
    }
    result
}

pub(crate) fn load_data(path: &str, block_size: usize) -> Vec<Vec<u8>> {
    let len_of_block_in_bytes = block_size/8;
    fs::read(path)
        .unwrap()
        .chunks(len_of_block_in_bytes)
        .map(|x| x.to_vec())
        .collect()
}

pub(crate) fn evaluate_pattern(pattern: &mut Pattern, path: &str, block_size: usize) -> f64 {
    let data = load_data(path, block_size);
    pattern.count = Some(
        data.par_iter()
            .map(|block| pattern.evaluate(block))
            .filter(|x| *x)
            .count(),
    );
    pattern.z(data.len())
}

pub(crate) fn evaluate_double_pattern(double_pattern: &DoublePattern, path: &str, block_size: usize) -> f64 {
    let data = load_data(path, block_size);
    let count = data
        .par_iter()
        .map(|block| double_pattern.evaluate(block))
        .filter(|x| *x)
        .count();
    double_pattern.z_score(data.len(), count)
}

pub(crate) fn split_data(data_in_path: &str, train_data_path: &str, test_data_path: &str, booltest_path: &str, block_size: usize) {
    let mut data = load_data(data_in_path, block_size);
    let mut rng = rand::thread_rng();
    data.shuffle(&mut rng);

    let mut booltest_file = File::create(booltest_path).unwrap();

    booltest_file.write_all(&data.iter().flatten().copied().collect::<Vec<u8>>()).unwrap();

    let mut train_file = File::create(train_data_path).unwrap();
    let mut test_file = File::create(test_data_path).unwrap();

    let (training_data, testing_data) = data.split_at(data.len() / 2);
    assert_eq!(training_data.len(), testing_data.len());

    train_file
        .write_all(&training_data.iter().flatten().copied().collect::<Vec<u8>>())
        .unwrap();
    test_file
        .write_all(&testing_data.iter().flatten().copied().collect::<Vec<u8>>())
        .unwrap();
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

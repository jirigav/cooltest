use std::time::Instant;

use itertools::Itertools;
use rayon::prelude::*;

use crate::common::{
    bits_block_eval, count_combinations, multi_eval_count, p_value_to_z_score, transform_data,
    z_score, Data,
};

#[derive(Debug, Clone)]
pub(crate) struct Histogram {
    pub(crate) bits: Vec<usize>,
    pub(crate) _bins: Vec<usize>,
    pub(crate) sorted_indices: Vec<usize>,
    pub(crate) best_division: usize,
    pub(crate) z_score: f64,
}

impl Histogram {
    pub(crate) fn get_hist(bits: &Vec<usize>, data: &[Vec<u8>]) -> Histogram {
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

    pub(crate) fn from_bins(bits: Vec<usize>, bins: Vec<usize>) -> Histogram {
        let mut indices = (0..2_usize.pow(bits.len() as u32)).collect_vec();
        indices.sort_by(|a, b| bins[*b].cmp(&bins[*a]));

        let mut max_z = 0.0;
        let mut best_i = 0;
        let prob = 2.0_f64.powf(-(bits.len() as f64));

        for i in 1..2_usize.pow(bits.len() as u32) {
            let mut count = 0;
            for k in 0..i {
                count += bins[indices[k]];
            }
            let z = z_score(bins.iter().sum(), count, prob * (i as f64));
            if z > max_z {
                max_z = z;
                best_i = i;
            }
        }
        Histogram {
            bits: bits.to_vec(),
            _bins: bins,
            sorted_indices: indices,
            best_division: best_i,
            z_score: max_z,
        }
    }

    pub(crate) fn evaluate(&self, data: &[Vec<u8>]) -> usize {
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

fn phase_one(data: &Data, block_size: usize, base_degree: usize, k: usize) -> Vec<Histogram> {
    let dises_per_bits = 2_usize.pow(base_degree as u32);
    let num_of_dises = count_combinations(block_size, base_degree) * dises_per_bits;
    let mut counts: Vec<usize> = vec![0; num_of_dises];
    let bits = (0..block_size).combinations(base_degree).collect_vec();

    let mut it = data.data.iter().peekable();
    while let Some(blocks) = it.next() {
        let is_last = it.peek().is_none();

        counts.par_iter_mut().enumerate().for_each(|(i, c)| {
            let bits_index = i / dises_per_bits;
            let bits_signs = i % dises_per_bits;
            *c += multi_eval_count(bits_signs, &bits[bits_index], blocks, data.mask, is_last)
                as usize;
        })
    }

    let mut best: Vec<_> = counts
        .into_par_iter()
        .chunks(dises_per_bits)
        .enumerate()
        .map(|(i, bins)| Histogram::from_bins(bits[i].clone(), bins))
        .collect();

    best.sort_by(|a, b| b.z_score.partial_cmp(&a.z_score).unwrap());
    best.into_iter().take(k).collect()
}

fn phase_two(
    data: &[Vec<u8>],
    block_size: usize,
    mut top_k: Vec<Histogram>,
    max_bits: usize,
    stop_z: f64,
) -> Vec<Histogram> {
    let mut final_bins: Vec<Histogram> = Vec::new();
    let mut length = top_k[0].bits.len();
    while !top_k.is_empty() && length < max_bits {
        length += 1;
        let mut new_top: Vec<Histogram> = Vec::new();

        for (i, hist) in top_k
            .par_iter()
            .map(|hist| {
                let mut best_imp: Option<Histogram> = None;
                for bit in 0..block_size {
                    if hist.bits.contains(&bit) {
                        continue;
                    }
                    let mut new_bits = hist.bits.clone();
                    new_bits.push(bit);
                    let new_hist = Histogram::get_hist(&new_bits.to_vec(), data);
                    if new_hist.z_score < hist.z_score {
                        continue;
                    }

                    if best_imp.is_none()
                        || f64::abs(best_imp.as_ref().unwrap().z_score) < f64::abs(new_hist.z_score)
                    {
                        best_imp = Some(new_hist);
                    }
                }
                best_imp
            })
            .enumerate()
            .collect::<Vec<_>>()
        {
            if let Some(imp) = hist {
                new_top.push(imp);
            } else {
                final_bins.push(top_k[i].clone());
            }
        }
        top_k = new_top;
        if top_k.iter().any(|h| f64::abs(h.z_score) > stop_z) {
            println!("stop z: {stop_z}");
            break;
        }
    }
    final_bins.extend(top_k);
    final_bins
}

pub(crate) fn bottomup(
    data: &Vec<Vec<u8>>,
    block_size: usize,
    k: usize,
    base_degree: usize,
    max_bits: usize,
    stop_p_value: f64,
) -> Histogram {
    let mut start = Instant::now();
    let top_k = phase_one(&transform_data(data), block_size, base_degree, k);
    println!("Phase one in {:?}", start.elapsed());
    start = Instant::now();
    let mut r = phase_two(
        data,
        block_size,
        top_k,
        max_bits,
        p_value_to_z_score(stop_p_value),
    );
    println!("Phase two in {:?}", start.elapsed());
    r.sort_unstable_by(|a, b| {
        f64::abs(b.z_score)
            .partial_cmp(&f64::abs(a.z_score))
            .unwrap()
    });
    println!("{:?}", r[0]);
    r[0].clone()
}

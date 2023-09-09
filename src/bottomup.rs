use std::time::Instant;

use itertools::Itertools;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::common::{bits_block_eval, z_score};

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
}

fn phase_one(data: &[Vec<u8>], block_size: usize, deg: usize, k: usize) -> Vec<Histogram> {
    let mut top_k: Vec<Histogram> = Vec::new();
    for bits in (0..block_size).combinations(deg) {
        let hist = Histogram::get_hist(&bits, data);
        if top_k.len() < k {
            top_k.push(hist);
        } else if top_k[0].z_score < hist.z_score {
            top_k[0] = hist;
        }
        top_k.sort_unstable_by(|a, b| {
            f64::abs(a.z_score)
                .partial_cmp(&f64::abs(b.z_score))
                .unwrap()
        });
    }
    top_k
}

fn phase_two(
    data: &[Vec<u8>],
    block_size: usize,
    mut top_k: Vec<Histogram>,
    max_bits: usize,
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
    }
    final_bins.extend(top_k);
    final_bins
}

pub(crate) fn bottomup(
    data: &[Vec<u8>],
    block_size: usize,
    k: usize,
    base_degree: usize,
    max_bits: usize,
) -> Histogram {
    let mut start = Instant::now();
    let top_k = phase_one(data, block_size, base_degree, k);
    println!("Phase one in {:?}", start.elapsed());
    start = Instant::now();
    let mut r = phase_two(data, block_size, top_k, max_bits);
    println!("Phase two in {:?}", start.elapsed());
    r.sort_unstable_by(|a, b| {
        f64::abs(b.z_score)
            .partial_cmp(&f64::abs(a.z_score))
            .unwrap()
    });
    println!("{:?}", r[0]);
    r[0].clone()
}

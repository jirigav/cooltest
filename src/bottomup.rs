use std::{fmt::DebugStruct, ops::Index, time::Instant};

use itertools::Itertools;
use rayon::{prelude::*, vec};

use crate::common::{
    bits_block_eval, count_combinations, multi_eval_count, transform_data,
    z_score, Data,
};

#[derive(Clone)]
pub(crate) struct Histogram {
    pub(crate) bits: Vec<usize>,
    pub(crate) _bins: Vec<usize>,
    pub(crate) sorted_indices: Vec<usize>,
    pub(crate) best_division: usize,
    pub(crate) z_score: f64,
    pub(crate) changes: Vec<f64>,
}

impl Histogram {

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
            changes: Vec::new(),
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

impl std::fmt::Debug for Histogram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Histogram")
            .field("bits", &self.bits)
            .field("best_division", &self.best_division)
            .field("z_score", &self.z_score)
            .field("changes", &self.changes)
            .finish()
    }
}


fn num_to_bits(mut k: usize, l: usize) -> Vec<bool> {
    let mut tf = Vec::new();
    while k != 0 {
        tf.push(k%2 == 1);
        k /= 2;
    }
    
    while tf.len() < l{
        tf.push(false);
    }
    tf
}

fn bits_to_num(tf: Vec<bool>) -> usize {
    let mut pow2 = 1;
    let mut res = 0;
    for b in tf{
        if b{
            res += pow2;
        }
        pow2 *= 2;
    }
    res
}

fn alt_phase_one(data: &Data, block_size: usize, deg: usize, k: usize) -> Vec<Histogram> {

    let mut start = Instant::now();

    let mut hists: Vec<Vec<usize>> = Vec::new();
    for i in 0..block_size{
        let mut ones: usize = 0;
        let mut it = data.data.iter().peekable();
        while let Some(blocks) = it.next() {
            let is_last = it.peek().is_none();
            ones += multi_eval_count(0, &vec![i], blocks, data.mask, is_last) as usize;
        }
        hists.push(vec![(data.num_of_blocks as usize) - ones, ones])
    }
    println!("One bits {:?}", start.elapsed());
    start = Instant::now();

    let mut prev_bits: Vec<Vec<usize>> = (0..block_size).map(|x| vec![x]).collect();
    for d in 2..=deg{
        let bits_vects = (0..block_size).combinations(d).collect_vec();

        let mut new_hists = Vec::new();

        for bits in bits_vects.iter(){
            let mut bins = vec![0;2_usize.pow(d as u32)];
            let mut ones: usize = 0;
            let mut it = data.data.iter().peekable();
            while let Some(blocks) = it.next() {
                let is_last = it.peek().is_none();
                ones += multi_eval_count(0, bits, blocks, data.mask, is_last) as usize;
            }

            let value = 2_usize.pow(d as u32) - 1;

            bins[value] = ones;

            for k in (0..value).rev(){
                let mut tf = num_to_bits(k, bits.len());
                let ind = tf.iter().position(|x| *x == false).unwrap();
                tf[ind] = true;
                let val = bits_to_num(tf.clone()); // previous result on the same level

                tf.remove(ind);
                let mut bits2 = bits.clone();
                bits2.remove(ind);

                let prev = hists[prev_bits.iter().position(|x| *x == bits2).unwrap()][bits_to_num(tf)]; // result from prev layer

                bins[k] = (prev - bins[val]) as usize;
            }

            new_hists.push(bins);   
        }
        hists = new_hists;
        prev_bits = bits_vects;
    }
    println!("main part {:?}", start.elapsed());
    start = Instant::now();
    hists.iter_mut().for_each(|x| x.reverse());
    let bits = (0..block_size).combinations(deg).collect_vec();
    let mut best: Vec<_> = hists
    .into_iter()
    .enumerate()
    .map(|(i, bins)| Histogram::from_bins(bits[i].clone(), bins))
    .collect();

    best.sort_by(|a, b| b.z_score.partial_cmp(&a.z_score).unwrap());
    println!("rest {:?}", start.elapsed());
    start = Instant::now();
    best.into_iter().take(k).collect()
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

pub(crate) fn bottomup(
    data: &Vec<Vec<u8>>,
    block_size: usize,
    base_degree: usize,
) -> Histogram {
    let start = Instant::now();
    let top_k = alt_phase_one(&transform_data(data), block_size, base_degree, 1);
    println!("Phase one in {:?}", start.elapsed());

    println!("{:?}", top_k[0]);
    top_k[0].clone()
}



#[cfg(test)]
mod tests {
    use super::{num_to_bits, bits_to_num};

    #[test]
    fn exploration() {
        for i in 0..8{
            println!("{}, {:?}", i, num_to_bits(i, 3));
            assert_eq!(bits_to_num(num_to_bits(i, 3)), i as usize)
        }
    }
}
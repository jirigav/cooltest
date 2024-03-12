use std::time::{Duration, Instant};

use itertools::Itertools;
use rayon::iter::*;

use crate::common::{
    bits_block_eval, multi_eval, transform_data, z_score, Data
};

#[derive(Clone)]
pub(crate) struct Histogram {
    pub(crate) bits: Vec<usize>,
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
            let z = z_score(data.len(), count, prob * (i as f64)).abs();
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
        }
    }


    pub(crate) fn from_bins(bits: Vec<usize>, bins: &Vec<usize>) -> Histogram {
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
            let z = z_score(bins.iter().sum(), count, prob * (i as f64)).abs();
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
            .field("sorted indices", &self.sorted_indices)
            .field("best_division", &self.best_division)
            .field("z_score", &self.z_score)
            .finish()
    }
}


fn first_zero_bit(mut k: usize) -> usize {
    let mut i = 0;
    while k !=0 && k%2 == 1 {
        k >>= 1;
        i += 1;
    }
    i
}

fn choose(n: usize, r: usize) -> usize {
    if r > n {
        0
    } else {
        (1..=r).fold(1, |acc, val| acc * (n - val + 1) / val)
    }
}


fn compute_index(bits: &Vec<usize>, block_size: usize) -> usize{
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

fn compute_bins(bits: &Vec<usize>, data: &Data, d: usize, hists: &Vec<Vec<usize>>, bins: &mut Vec<usize>, block_size: usize, t: &mut Duration) {
    let ones;
    ones = multi_eval( bits, data, t);

    let value = 2_usize.pow(d as u32) - 1;

    bins[value] = ones;

    for k in (0..value).rev(){
        // find first zero in bin's index k and replace if with one. i.e. obtain index with distance 1 for which the bin value is already computed
        let mut k2 = k;
        let ind = first_zero_bit(k); 
        k2 ^= 1 << ind; // flip bit to one

        let n = (k2&((1 << ind) -1)) + ((k2 >>(ind + 1)) << ind); // remove ind-th bit from the number

        let mut bits2 = bits.clone();
        bits2.remove(ind);

        let prev = hists[compute_index(&bits2, block_size)][n]; // result from prev layer

        bins[k] = (prev - bins[k2]) as usize;
    }
}

fn brute_force(data: &Data, block_size: usize, deg: usize, k: usize) -> Vec<Histogram> {
    let mut t = Duration::from_micros(0);
    compute_index(&vec![3,6,7], 8);

    let mut hists: Vec<Vec<usize>> = Vec::new();
    for i in 0..block_size{
        let ones;
        ones = multi_eval(&vec![i], data, &mut t);
        hists.push(vec![(data.num_of_blocks as usize) - ones, ones])
    }

    for d in 2..deg{
        let mut new_hists = Vec::with_capacity(2_usize.pow(deg as u32));

        for bits in (0..block_size).combinations(d){
            let mut bins = vec![0;2_usize.pow(d as u32)];
            compute_bins(&bits, data, d, &hists, &mut bins, block_size, &mut t);

            new_hists.push(bins);   
        }
        hists = new_hists;
    }
    let mut best_hists = vec![Histogram::from_bins(vec![0], &vec![1, 1]); k];
    let mut bins = vec![0;2_usize.pow(deg as u32)];
    for bits in (0..block_size).combinations(deg){
        compute_bins(&bits, data, deg, &hists, &mut bins, block_size, &mut t);

        let hist = Histogram::from_bins(bits, &bins);
        best_hists.push(hist);
        best_hists.sort_by(|a, b| b.z_score.abs().partial_cmp(&a.z_score.abs()).unwrap());
        best_hists.pop();
    }
    println!("Stream operations: {:?}", t);
    best_hists
}

fn _combine_bins(hists: &Vec<Histogram>, n: usize, data: &[Vec<u8>]) -> Histogram {
    let mut best_hist = Histogram::from_bins(vec![0], &vec![1, 1]);
    for comb in hists.iter().combinations(n) {
        let mut bits = comb.iter().flat_map(|x| x.bits.clone()).collect_vec();
        bits.sort();
        bits.dedup();

        let hist = Histogram::get_hist(&bits, data);
        if hist.z_score.abs() > best_hist.z_score.abs() {
            best_hist = hist;
        }
    }

    best_hist
}

pub(crate) fn bottomup(
    data: &Vec<Vec<u8>>,
    block_size: usize,
    base_degree: usize,
    k: usize,
    max_bits: usize,
) -> Histogram {
    let mut start = Instant::now();
    let mut top_k = brute_force(&transform_data(data), block_size, base_degree, k);
    println!("Brute-force finished in {:?}", start.elapsed());
    
    if max_bits > base_degree {
        start = Instant::now();
        top_k = phase_two(data, block_size, top_k, max_bits); 
        println!("Heuristic search finished in {:?}", start.elapsed());
    }
    
    let res = top_k[0].clone();
    println!("Distinguisher: {:?}", res);
    res
}


fn phase_two(
    data: &[Vec<u8>],
    block_size: usize,
    mut top_k: Vec<Histogram>,
    max_bits: usize,
) -> Vec<Histogram> {
    let mut final_hists: Vec<Histogram> = Vec::new();
    let mut length = top_k[0].bits.len();
    while !top_k.is_empty() && length < max_bits {
        length += 1;
        
        let hists = top_k.par_iter().map(|hist| {
            let mut new_hists: Vec<Histogram> = Vec::new();
            for bit in 0..block_size {
                if hist.bits.contains(&bit){
                    continue;
                }
                let mut new_bits = hist.bits.clone();
                new_bits.push(bit);
                new_bits.sort();

                let new_hist = Histogram::get_hist(&new_bits.to_vec(), data);

                new_hists.push(new_hist);
            }

            new_hists.sort_unstable_by(|a, b| b.z_score.abs().partial_cmp(&a.z_score.abs()).unwrap());
            new_hists
        }).collect::<Vec<_>>();
        let mut new_top: Vec<Histogram> = Vec::new();
        for hs in hists {
            for h in hs{
                if !new_top.iter().any(|x| x.bits == h.bits) {
                    new_top.push(h);
                    break;
                }
            }
        }
        top_k = new_top;
    }

    final_hists.extend(top_k);
    final_hists
}


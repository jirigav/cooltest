use std::time::Instant;

use crate::common::{bit_value_in_block, transform_data, z_score, Args, Data};
use crate::distinguishers::{Distinguisher, Pattern};
use itertools::Itertools;
use rayon::prelude::*;

fn count_combinations(n: usize, r: usize) -> usize {
    if r > n {
        0
    } else {
        (1..=r.min(n - r)).fold(1, |acc, val| acc * (n - val + 1) / val)
    }
}

fn multi_eval(
    bits_signs: usize,
    bits: &[usize],
    tr_data: &[u128],
    mask: u128,
    is_last: bool,
) -> u128 {
    let mut result = u128::MAX;

    for (i, b) in bits.iter().enumerate() {
        if ((bits_signs >> i) & 1) == 1 {
            result &= tr_data[*b];
        } else {
            result &= tr_data[*b] ^ u128::MAX;
        }
    }
    if is_last {
        result & mask
    } else {
        result
    }
}

fn phase_one(data: &[Vec<u128>], mask: u128, k: usize, block_size: usize, base_degree: usize) -> Vec<Pattern> {
    let m = (0..2_u8.pow(base_degree as u32))
        .map(|x| {
            (0..base_degree)
                .map(|i| bit_value_in_block(i, &[x]))
                .collect_vec()
        })
        .collect_vec();

    

    let dises_per_bits = 2_usize.pow(base_degree as u32);
    let num_of_dises = count_combinations(block_size, base_degree) * dises_per_bits;
    let mut counts: Vec<u32> = vec![0; num_of_dises];
    let bits = (0..block_size).combinations(base_degree).collect_vec();

    let mut it = data.iter().peekable();
    while let Some(blocks) = it.next() {
        let is_last = it.peek().is_none();

        counts.par_iter_mut().enumerate().for_each(|(i, c)| {
            let bits_index = i / dises_per_bits;
            let bits_signs = i % dises_per_bits;
            *c +=
                multi_eval(bits_signs, &bits[bits_index], blocks, mask, is_last).count_ones();
        })
    }

    let mut best: Vec<_> = counts
        .into_par_iter()
        .enumerate()
        .map(|(i, c)| {
            let bits_index = i / dises_per_bits;
            let bits_values_index = i % dises_per_bits;
            (c, (&bits[bits_index], &m[bits_values_index]), bits_values_index)
        })
        .collect();

    best.sort_by(|a, b| b.0.cmp(&a.0));
    best.into_iter()
        .take(k)
        .map(|(count, (bits, values), signs)| Pattern {
            length: base_degree,
            bits: bits.clone(),
            values: values.clone(),
            count: Some(count as usize),
            z_score: None,
            validation_z_score: None,
            bits_signs: signs,
        })
        .collect()
}

fn is_improving(old_z_score: f64, count_new: usize, new_length: usize, samples: usize) -> bool {
    let p_new = 2_f64.powf(-(new_length as f64));
    old_z_score <= z_score(samples, count_new, p_new)
}

fn improving(
    validation_data_option: Option<&(Vec<Vec<u128>>, u128)>,
    pattern: &mut Pattern,
    hist: &[(usize, usize)],
    samples: usize,
    min_difference: usize,
) -> Vec<Pattern> {
    let mut new_patterns = Vec::new();
    let mut validation = false;
    let mut validation_data = &(Vec::new(), u128::MIN);
    if let Some(data) = validation_data_option {
        validation = true;
        validation_data = data;
    }
    if validation && pattern.validation_z_score.is_none() {
        let len_m1 = validation_data.0.len() -1;
        pattern.validation_z_score = Some(z_score(
            len_m1*128 + validation_data.1.count_ones() as usize,
            validation_data.0
                .iter().enumerate()
                .map(|(i, blocks)| multi_eval(pattern.bits_signs, &pattern.bits, blocks, validation_data.1, i == len_m1).count_ones())
                .sum::<u32>() as usize,
            2.0_f64.powf(-(pattern.length as f64)),
        ));
    }
    if pattern.z_score.is_none() {
        pattern.z_score(samples);
    }

    for (i, counts) in hist.iter().enumerate() {
        if pattern.bits.contains(&i) {
            continue;
        }

        let mut count = counts.0;
        let mut v = false;
        if counts.0 < counts.1 {
            count = counts.1;
            v = true;
        }

        if is_improving(pattern.z_score.unwrap(), count, pattern.length + 1, samples)
            && count - ((2.0_f64.powf(-(pattern.length as f64 + 1.0)) * (samples as f64)) as usize)
                >= min_difference
        {
            let mut new_pattern = pattern.clone();
            new_pattern.forget_count();
            new_pattern.add_bit(i, v);
            new_pattern.increase_count(count);

            if validation {
                let new_z = new_pattern.z_score(samples);

                let len_m1 = validation_data.0.len() -1;
                let valid_z = z_score(
                    len_m1*128 + validation_data.1.count_ones() as usize,
                    validation_data.0
                        .iter().enumerate()
                        .map(|(i, blocks)| multi_eval(new_pattern.bits_signs, &new_pattern.bits, blocks, validation_data.1, i == len_m1).count_ones())
                        .sum::<u32>() as usize,
                    2.0_f64.powf(-(new_pattern.length as f64)),
                );

                if pattern.validation_z_score.unwrap() > valid_z || valid_z < 0.6 * new_z {
                    continue;
                }

                new_pattern.validation_z_score = Some(valid_z);
            }
            new_patterns.push(new_pattern);
        }
    }
    new_patterns
}

fn faster_phase_two(
    k: usize,
    mut top_k: Vec<Pattern>,
    data: &[Vec<u128>],
    mask: u128,
    validation_data_option: Option<&(Vec<Vec<u128>>, u128)>, 
    min_difference: usize,
    block_size: usize,
    blocks_count: usize,
) -> Vec<Pattern> {
    let mut final_patterns: Vec<Pattern> = Vec::with_capacity(k);

    let mut pattern_len = 2;

    while !top_k.is_empty() && pattern_len < block_size {
        pattern_len += 1;

        let mut hists: Vec<Vec<(usize, usize)>> = Vec::with_capacity(top_k.len());
        for _ in 0..top_k.len() {
            hists.push(vec![(0, 0); block_size]);
        }

        let mut it = data.iter().peekable();
        
        while let Some(blocks) = it.next() {
            let is_last = it.peek().is_none();

            hists.par_iter_mut().enumerate().for_each(|(i, hist)|{
                let mut bits = top_k[i].bits.clone();
                bits.push(0);
                for b in 0..block_size {
                    if top_k[i].bits.contains(&b) {
                        continue;
                    }
                    
                    bits[top_k[i].length] = b;
                    hist[b].1 += multi_eval(top_k[i].bits_signs + 2_usize.pow(top_k[i].length as u32), &bits, blocks, mask, is_last).count_ones() as usize;
                    hist[b].0 += multi_eval(top_k[i].bits_signs, &bits, blocks, mask, is_last).count_ones() as usize;
                }
            })
        }

        let mut new_top_k: Vec<Pattern> = Vec::with_capacity(top_k.len());

        for i in 0..hists.len() {
            let mut imp = improving(
                validation_data_option,
                &mut top_k[i],
                &hists[i],
                blocks_count,
                min_difference,
            );

            if imp.is_empty() {
                final_patterns.push(top_k[i].clone());
                continue;
            }

            imp.sort_unstable_by_key(|b| std::cmp::Reverse(b.get_count()));

            let mut pattern_added = false;
            for p in &imp {
                if !new_top_k.contains(p) {
                    new_top_k.push(p.clone());
                    pattern_added = true;
                    break;
                }
            }
            if !pattern_added {
                final_patterns.push(top_k[i].clone());
            }
        }
        top_k = new_top_k;
    }

    final_patterns
}

pub(crate) fn bottomup(
    data: &Data,
    validation_data_option: Option<&Vec<Vec<u8>>>,
    args: &Args,
) -> Vec<Pattern> {
    let mut start = Instant::now();
    let (transformed_data, mask) = transform_data(data, args.block_size);
    let top_k = phase_one(&transformed_data, mask, args.k, args.block_size, args.base_pattern_size);
    println!("phase one {:.2?}", start.elapsed());
    start = Instant::now();
    let mut tr_val = None;
    if validation_data_option.is_some(){
        tr_val = Some(transform_data(validation_data_option.unwrap(), args.block_size));
    }
    let r = faster_phase_two(args.k, top_k, &transformed_data, mask, tr_val.as_ref(), args.min_difference, args.block_size, data.len());
    println!("phase two {:.2?}", start.elapsed());
    r
}

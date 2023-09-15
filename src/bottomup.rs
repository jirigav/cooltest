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
    negations_index: usize,
    bits: &[usize],
    tr_data: &[u128],
    mask: u128,
    is_last: bool,
) -> u128 {
    let mut result = u128::MAX;

    for (i, b) in bits.iter().enumerate() {
        if ((negations_index >> i) & 1) == 1 {
            result &= tr_data[*b] ^ u128::MAX;
        } else {
            result &= tr_data[*b];
        }
    }
    if is_last {
        result & mask
    } else {
        result
    }
}

fn phase_one(data: &Data, k: usize, block_size: usize, base_degree: usize) -> Vec<Pattern> {
    let m = (0..2_u8.pow(base_degree as u32))
        .rev()
        .map(|x| {
            (0..base_degree)
                .map(|i| bit_value_in_block(i, &[x]))
                .collect_vec()
        })
        .collect_vec();

    let (transformed_data, mask) = transform_data(data, block_size);

    let dises_per_bits = 2_usize.pow(base_degree as u32);
    let num_of_dises = count_combinations(block_size, base_degree) * dises_per_bits;
    let mut counts: Vec<u32> = vec![0; num_of_dises];
    let bits = (0..block_size).combinations(base_degree).collect_vec();

    let mut it = transformed_data.iter().peekable();
    while let Some(blocks) = it.next() {
        let is_last = it.peek().is_none();

        counts.par_iter_mut().enumerate().for_each(|(i, c)| {
            let bits_index = i / dises_per_bits;
            let negations_index = i % dises_per_bits;
            *c +=
                multi_eval(negations_index, &bits[bits_index], blocks, mask, is_last).count_ones();
        })
    }

    let mut best: Vec<_> = counts
        .into_par_iter()
        .enumerate()
        .map(|(i, c)| {
            let bits_index = i / dises_per_bits;
            let negations_index = i % dises_per_bits;
            (c, (&bits[bits_index], &m[negations_index]))
        })
        .collect();

    best.sort_by(|a, b| b.0.cmp(&a.0));
    best.into_iter()
        .take(k)
        .map(|(count, (bits, values))| Pattern {
            length: base_degree,
            bits: bits.clone(),
            values: values.clone(),
            count: Some(count as usize),
            z_score: None,
            validation_z_score: None,
        })
        .collect()
}

fn is_improving(old_z_score: f64, count_new: usize, new_length: usize, samples: usize) -> bool {
    let p_new = 2_f64.powf(-(new_length as f64));
    old_z_score <= z_score(samples, count_new, p_new)
}

fn improving(
    validation_data_option: Option<&Vec<Vec<u8>>>,
    pattern: &mut Pattern,
    hist: &[(usize, usize)],
    samples: usize,
    min_difference: usize,
) -> Vec<Pattern> {
    let mut new_patterns = Vec::new();
    let mut validation = false;
    let mut validation_data = &Vec::new();
    if let Some(data) = validation_data_option {
        validation = true;
        validation_data = data;
    }
    if validation && pattern.validation_z_score.is_none() {
        pattern.validation_z_score = Some(z_score(
            validation_data.len(),
            validation_data
                .iter()
                .filter(|block| pattern.evaluate(block))
                .count(),
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
                let valid_z = z_score(
                    validation_data.len(),
                    validation_data
                        .iter()
                        .filter(|block| new_pattern.evaluate(block))
                        .count(),
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

fn phase_two(
    k: usize,
    mut top_k: Vec<Pattern>,
    data: &[Vec<u8>],
    validation_data_option: Option<&Vec<Vec<u8>>>,
    min_difference: usize,
    block_size: usize,
) -> Vec<Pattern> {
    let mut final_patterns: Vec<Pattern> = Vec::with_capacity(k);

    let mut pattern_len = 2;

    while !top_k.is_empty() && pattern_len < block_size {
        pattern_len += 1;

        let mut hists: Vec<Vec<(usize, usize)>> = Vec::with_capacity(top_k.len());
        for _ in 0..top_k.len() {
            hists.push(vec![(0, 0); block_size]);
        }

        for block in data {
            for (i, pattern) in top_k.iter().enumerate() {
                if pattern.evaluate(block) {
                    for b in 0..block_size {
                        if pattern.bits.contains(&b) {
                            continue;
                        }
                        if bit_value_in_block(b, block) {
                            hists[i][b].1 += 1;
                        } else {
                            hists[i][b].0 += 1;
                        }
                    }
                }
            }
        }

        let mut new_top_k: Vec<Pattern> = Vec::with_capacity(top_k.len());

        for i in 0..hists.len() {
            let mut imp = improving(
                validation_data_option,
                &mut top_k[i],
                &hists[i],
                data.len(),
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
    let top_k = phase_one(data, args.k, args.block_size, args.base_pattern_size);
    println!("phase one {:.2?}", start.elapsed());
    start = Instant::now();
    let r = phase_two(
        args.k,
        top_k,
        data,
        validation_data_option,
        args.min_difference,
        args.block_size,
    );
    println!("phase two {:.2?}", start.elapsed());
    r
}

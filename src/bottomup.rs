use std::time::Instant;

use crate::common::{bit_value_in_block, bits_block_eval, z_score, Args, p_value};
use crate::distinguishers::{Distinguisher, Pattern};
use itertools::Itertools;
use rayon::prelude::*;

fn phase_one(data: &[Vec<u8>], k: usize, block_size: usize, base_degree: usize) -> Vec<Pattern> {
    let m = (0..2_u8.pow(base_degree as u32))
        .map(|x| {
            (0..base_degree)
                .map(|i| bit_value_in_block(i, &[x]))
                .collect_vec()
        })
        .collect_vec();
    let mut best_patterns = vec![(0, (Vec::new(), Vec::new())); k];

    for bits in (0..block_size).combinations(base_degree) {
        let hist = data
            .par_iter()
            .map(|block| bits_block_eval(&bits, block))
            .fold_with(vec![0; 2_usize.pow(base_degree as u32)], |mut a, b| {
                a[b] += 1;
                a
            })
            .reduce(
                || vec![0; 2_usize.pow(base_degree as u32)],
                |mut a: Vec<usize>, b: Vec<usize>| {
                    a = a.iter_mut().zip(b).map(|(a, b)| *a + b).collect();
                    a
                },
            );

        let max_count = hist.iter().max().unwrap();

        if max_count > &best_patterns[k - 1].0 {
            best_patterns[k - 1] = (
                *max_count,
                (
                    bits,
                    m[hist.iter().position(|x| x == max_count).unwrap()].clone(),
                ),
            );
            best_patterns.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        }
    }

    best_patterns
        .into_iter()
        .map(|(count, (bits, values))| Pattern {
            length: base_degree,
            bits,
            values,
            count: Some(count),
            z_score: None,
        })
        .collect()
}

fn is_improving(old_z_score: f64, count_new: usize, new_length: usize, samples: usize) -> bool {
    let p_new = 2_f64.powf(-(new_length as f64));
    old_z_score <= z_score(samples, count_new, p_new)
}

fn improving(
    pattern: &mut Pattern,
    hist: &[(usize, usize)],
    samples: usize,
    imp_p_value: f64,
) -> Vec<Pattern> {
    let mut new_patterns = Vec::new();
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
            && p_value(count, pattern.get_count(), 0.5) <= imp_p_value
        {
            let mut new_pattern = pattern.clone();
            new_pattern.forget_count();
            new_pattern.add_bit(i, v);
            println!("{}   {} - {:?}", imp_p_value, p_value(count, pattern.get_count(), 0.5), new_pattern.bits);
            new_pattern.increase_count(count);
            new_patterns.push(new_pattern);
        }
    }
    new_patterns
}

fn phase_two(
    k: usize,
    mut top_k: Vec<Pattern>,
    data: &[Vec<u8>],
    mut imp_p_value: f64,
    block_size: usize,
    base_degree: usize
) -> Vec<Pattern> {
    let mut final_patterns: Vec<Pattern> = Vec::with_capacity(k);

    let mut pattern_len = base_degree;
    //imp_p_value *= 0.5_f64.powf(base_degree as f64);

    while !top_k.is_empty() && pattern_len < block_size {
        pattern_len += 1;
        imp_p_value *= 0.6;


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
            let mut imp = improving(&mut top_k[i], &hists[i], data.len(), imp_p_value);

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

pub(crate) fn bottomup(data: &[Vec<u8>], args: &Args) -> Vec<Pattern> {
    let mut start = Instant::now();
    let top_k = phase_one(data, args.k, args.block_size, args.base_pattern_size);
    println!("phase one {:.2?}", start.elapsed());
    start = Instant::now();
    let r = phase_two(args.k, top_k, data, args.improving_p_value, args.block_size, args.base_pattern_size);
    println!("phase two {:.2?}", start.elapsed());
    r
}


fn phase_two_alt(
    k: usize,
    mut top_k: Vec<Pattern>,
    data: &[Vec<u8>],
    imp_p_value: f64,
    block_size: usize,
    base_degree: usize
) -> Vec<Pattern> {
    let mut final_patterns: Vec<Pattern> = Vec::with_capacity(k);

    let mut pattern_len = base_degree;

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
            let mut imp = improving(&mut top_k[i], &hists[i], data.len(), imp_p_value);

            if imp.is_empty() {
                final_patterns.push(top_k[i].clone());
                continue;
            }

            imp.sort_unstable_by_key(|b| std::cmp::Reverse(b.get_count()));

            let mut pattern_added = false;
            for p in &imp {
                if !new_top_k.contains(p) {
                    if new_top_k.len() < k{
                        new_top_k.push(p.clone());
                    }else if new_top_k[0].get_count() < p.get_count(){
                        pattern_added = true;
                        new_top_k[0] = p.clone();
                        new_top_k.sort_unstable_by_key(|pattern| pattern.get_count());

                    } else {
                        break;
                    }
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

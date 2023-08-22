use crate::common::{bit_value_in_block, bits_block_eval, z_score};
use crate::distinguishers::{Distinguisher, Pattern};
use rayon::prelude::*;

fn phase_one(data: &[Vec<u8>], k: usize, block_size: usize) -> Vec<Pattern> {
    let m = [(false, false), (true, false), (false, true), (true, true)];
    let mut best_pairs = vec![(0, ((0, 0), (false, false))); k];

    for i in 0..block_size {
        for j in (i + 1)..block_size {
            let mut hist = [0; 4];
            let a = data
                .par_iter()
                .map(|block| bits_block_eval(&[i, j], block))
                .collect::<Vec<_>>();
            for v in a {
                hist[v] += 1;
            }

            let max_count = hist.iter().max().unwrap();
            if max_count > &best_pairs[k - 1].0 {
                best_pairs[k - 1] = (
                    *max_count,
                    ((i, j), m[hist.iter().position(|x| x == max_count).unwrap()]),
                );
                best_pairs.sort_by(|a, b| b.0.cmp(&a.0));
            }
        }
    }
    best_pairs
        .into_iter()
        .map(|(count, ((i, j), (a, b)))| Pattern {
            length: 2,
            bits: vec![i, j],
            values: vec![a, b],
            count: Some(count),
            z_score: None,
        })
        .collect()
}

fn is_improving(old_z_score: f64, count_new: usize, new_length: usize, samples: usize) -> bool {
    let p_new = 2_f64.powf(-(new_length as f64));

    f64::abs(old_z_score) <= f64::abs(z_score(samples, count_new, p_new))
}

fn improving(
    pattern: &mut Pattern,
    hist: &[(usize, usize)],
    samples: usize,
    min_count: usize,
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
            && count >= min_count
        {
            let mut new_pattern = pattern.clone();
            new_pattern.add_bit(i, v);
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
    min_count: usize,
    block_size: usize,
    evaluated_dises: &mut usize,
) -> Vec<Pattern> {
    let mut final_patterns: Vec<Pattern> = Vec::with_capacity(k);

    let mut pattern_len = 2;

    while !top_k.is_empty() && pattern_len < block_size {
        pattern_len += 1;

        let mut hists: Vec<Vec<(usize, usize)>> = Vec::new();
        *evaluated_dises += 2 * top_k.len() * (block_size - pattern_len + 1);
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

        let mut new_top_k: Vec<Pattern> = Vec::new();

        for i in 0..hists.len() {
            let mut imp = improving(&mut top_k[i], &hists[i], data.len(), min_count);

            if imp.is_empty() {
                final_patterns.push(top_k[i].clone());
                continue;
            }

            imp.sort_by_key(|b| std::cmp::Reverse(b.get_count()));

            for p in &imp {
                if !new_top_k.contains(p) {
                    new_top_k.push(p.clone());
                    break;
                }
            }
        }
        top_k = new_top_k;
    }

    final_patterns
}

pub(crate) fn bottomup(
    data: &[Vec<u8>],
    block_size: usize,
    k: usize,
    min_count: usize,
) -> (Vec<Pattern>, usize) {
    let mut evaluated_dises = 2 * block_size * (block_size - 1); // 4*(bock_size choose 2)
    let top_k = phase_one(data, k, block_size);
    (
        phase_two(k, top_k, data, min_count, block_size, &mut evaluated_dises),
        evaluated_dises,
    )
}

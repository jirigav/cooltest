use std::collections::HashSet;
use std::time::Instant;

use crate::common::{bit_value_in_block, bits_block_eval};
use crate::distinguishers::{Distinguisher, MultiPattern, Pattern};
use itertools::Itertools;
use rayon::prelude::*;

fn phase_one(data: &[Vec<u8>], n: usize, block_size: usize, base_degree: usize) -> Vec<Pattern> {
    let m = (0..2_u8.pow(base_degree as u32))
        .map(|x| {
            (0..base_degree)
                .map(|i| bit_value_in_block(i, &[x]))
                .collect_vec()
        })
        .collect_vec();
    let mut sorted_patterns = Vec::new();

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

        let min_count = hist.iter().min().unwrap();

        sorted_patterns.push((
            *min_count,
            (
                bits,
                m[hist.iter().position(|x| x == min_count).unwrap()].clone(),
            ),
        ));
    }
    sorted_patterns.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    sorted_patterns
        .into_iter()
        .take(n)
        .map(|(count, (bits, values))| Pattern {
            length: base_degree,
            bits,
            values,
            count: Some(count),
            z_score: None,
        })
        .collect()
}

fn phase_two(
    k: usize,
    sorted_patterns: Vec<Pattern>,
    data: &[Vec<u8>],
    min_difference: usize,
) -> MultiPattern {
    let mut top_mp = sorted_patterns
        .iter()
        .take(k)
        .map(|p| {
            let mut mp = MultiPattern::new(vec![p.clone()]);
            mp.increase_count(p.count.unwrap());
            mp.z_score(data.len());
            mp
        
        })
        .collect_vec();
    let mut best_mp: Option<MultiPattern> = None;
    let mut combined = 1;
    while !top_mp.is_empty() && combined < 30 {
        combined += 1;
        let mut new_top_mp: Vec<MultiPattern> = Vec::new();
        for mp in &top_mp{
            let mut best_improving: Option<MultiPattern> = None;
            let bits: HashSet<usize> = mp.patterns
                .iter()
                .flat_map(|x| x.bits.iter().copied())
                .collect();
            for p in &sorted_patterns{
                if mp.patterns.contains(&p){
                    continue;
                }
                
                if bits.intersection(&p.bits.iter().copied().collect()).next().is_some(){ // skip if any bits are shared
                    continue;
                }
                let mut new_mp = mp.clone();
                new_mp.add_disjoint_pattern(p.clone());
                
                new_mp.increase_count(data.par_iter().map(|block| new_mp.evaluate(block)).filter(|x| *x).count());

                if new_mp.get_count().abs_diff((new_mp.probability * (data.len() as f64)) as usize) < combined * min_difference{
                    continue;
                }

                let z = new_mp.z_score(data.len());
                if !new_top_mp.contains(&new_mp) && z < mp.z_score.unwrap() && (best_improving.is_none() || best_improving.as_ref().unwrap().z_score.unwrap() > z){
                    best_improving = Some(new_mp.clone());
                }

                if best_mp.is_none() || best_mp.as_ref().unwrap().z_score.unwrap() > z {
                    best_mp = Some(new_mp);
                }
            }
            if let Some(imp) = best_improving{
                new_top_mp.push(imp);
            }
            
        }
        top_mp = new_top_mp;

    }
    best_mp.unwrap()
}

pub(crate) fn bottomup(
    data: &[Vec<u8>],
    block_size: usize,
    k: usize,
    n: usize,
    min_difference: usize,
    base_degree: usize,
) -> MultiPattern {
    let mut start = Instant::now();
    let sorted_patterns = phase_one(data, n, block_size, base_degree);
    println!("phase one {:.2?}", start.elapsed());
    start = Instant::now();
    let r = phase_two(k, sorted_patterns, data, min_difference);
    println!("phase two {:.2?}", start.elapsed());
    println!("{r:?}");
    r
}

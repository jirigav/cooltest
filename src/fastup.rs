use crate::common::*;
use rayon::prelude::*;

fn basic_zs(data: &Vec<Vec<u8>>, block_size: usize) -> Vec<f64> {
    let mut counts = vec![0; block_size];

   for block in data.iter() {
        for (i, count) in counts.iter_mut().enumerate() {
            if bit_value_in_block(&i, block) {
                *count += 1
            }
        }
    }
    counts
        .iter()
        .map(|c| z_score(data.len(), *c, 0.5))
        .collect()
}

fn top_n_bits(zs: Vec<f64>, n: usize) -> Vec<(usize, f64)> {
    let mut top_n = vec![(0, 0.0); n];

    for (i, z) in zs.iter().enumerate() {
        if f64::abs(*z) > f64::abs(top_n[0].1) {
            top_n[0] = (i, *z);
            top_n.sort_by(|a, b| f64::abs(a.1).partial_cmp(&f64::abs(b.1)).unwrap());
        }
    }
    top_n
}

fn z_to_bit_value(z: &f64) -> bool {
    z.signum() == 1.0
}

fn new_pattern(bit: usize, bit_z: &f64, old_pattern: &Pattern) -> Pattern {
    let mut testpattern: Pattern = old_pattern.clone();
    if bit_z.signum() == old_pattern.z_score.unwrap().signum() {
        testpattern.add_bit(bit, z_to_bit_value(bit_z));
    } else {
        testpattern.add_bit(bit, !z_to_bit_value(bit_z));
        // IDEA:  add also inverted pattern + non-negated bit
    };
    testpattern
}

fn extend_patterns(
    best_patterns: Vec<Pattern>,
    top_bits: &[(usize, f64)],
    data: &[Vec<u8>],
    min_count: usize,
    final_patterns: &mut Vec<Pattern>
) -> Vec<Pattern> {
    let mut new_patterns = Vec::new();

    for p in best_patterns {
        let mut best_improving_pattern: Option<Pattern> = None;
        for (b, z) in top_bits {
            if p.bits.contains(b) {
                continue;
            }
            let mut testpattern = new_pattern(*b, z, &p);

            if new_patterns.contains(&testpattern) {
                continue;
            }
            testpattern.count = Some(
                data.par_iter()
                    .filter(|block| testpattern.evaluate(block))
                    .count(),
            );
            let new_z = testpattern.z(data.len());

            if f64::abs(new_z) < f64::abs(p.z_score.unwrap())
                || testpattern.count.unwrap() < min_count
            {
                continue;
            }

            if best_improving_pattern.is_none()
                || f64::abs(best_improving_pattern.as_ref().unwrap().z_score.unwrap())
                    < f64::abs(new_z)
            {
                best_improving_pattern = Some(testpattern);
            }
        }
        if let Some(imp_pattern) = best_improving_pattern {
            new_patterns.push(imp_pattern);
        } else {
            final_patterns.push(p);
        }
    }

    new_patterns
}

pub(crate) fn fastup(args: &Args, data: &Vec<Vec<u8>>) -> Vec<Pattern> {

    let zs = basic_zs(data, args.block_size);

    let top_bits = top_n_bits(zs, args.n);

    let mut best_patterns: Vec<Pattern> = top_bits
        .iter()
        .skip(args.n - args.k)
        .map(|(i, z)| Pattern {
            length: 1,
            bits: vec![*i],
            values: vec![z_to_bit_value(z)],
            count: None,
            z_score: Some(*z),
        })
        .collect();
    let mut final_patterns: Vec<Pattern> = Vec::new();
    
    let mut current_length = 1;
    while !best_patterns.is_empty() && current_length < args.n {
        current_length += 1;
        best_patterns =
            extend_patterns(best_patterns, &top_bits, data, args.min_count, &mut final_patterns);
    }

    final_patterns
}

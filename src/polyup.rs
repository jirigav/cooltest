use std::time::Instant;

use crate::common::{bit_value_in_block, bits_block_eval};
use crate::distinguishers::{Distinguisher, Polynomial};
use itertools::Itertools;
use rayon::prelude::*;

fn phase_one(data: &[Vec<u8>], k: usize, block_size: usize, base_degree: usize) -> Vec<Polynomial> {
    let expected = (2.0_f64.powf(-(base_degree as f64)) * data.len() as f64) as usize;

    let m = (0..2_u8.pow(base_degree as u32))
        .map(|x| {
            (0..base_degree)
                .map(|i| bit_value_in_block(i, &[x]))
                .collect_vec()
        })
        .collect_vec();
    let mut best_patterns = vec![(expected, (Vec::new(), Vec::new())); k];

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
            best_patterns
                .sort_unstable_by(|a, b| b.0.abs_diff(expected).cmp(&a.0.abs_diff(expected)));
        }
    }

    best_patterns
        .into_iter()
        .map(|(count, (bits, values))| {
            let mut p = Polynomial::new();
            for (b, v) in bits.iter().zip(values.iter()) {
                p.and(*b, !v).unwrap();
            }
            p.increase_count(count);
            p.z_score(data.len());
            p
        })
        .collect()
}

fn new_polys(bit: usize, old_polynomial: &Polynomial) -> Vec<Polynomial> {
    let mut testpolynomials = Vec::new();

    let mut testpolynomial1: Polynomial = old_polynomial.clone();
    testpolynomial1.and(bit, false).unwrap();
    testpolynomials.push(testpolynomial1);

    let mut testpolynomial2: Polynomial = old_polynomial.clone();
    testpolynomial2.and(bit, true).unwrap();
    testpolynomials.push(testpolynomial2);

    let mut testpolynomial3: Polynomial = old_polynomial.clone();
    testpolynomial3.negate();
    testpolynomial3.and(bit, false).unwrap();
    testpolynomials.push(testpolynomial3);

    let mut testpolynomial4: Polynomial = old_polynomial.clone();
    testpolynomial4.negate();
    testpolynomial4.and(bit, true).unwrap();
    testpolynomials.push(testpolynomial4);

    testpolynomials
}

fn extend_polynomials(
    best_polynomials: Vec<Polynomial>,
    block_size: usize,
    data: &[Vec<u8>],
    min_difference: usize,
    final_polynomials: &mut Vec<Polynomial>,
) -> Vec<Polynomial> {
    let mut new_polynomials: Vec<Polynomial> = Vec::new();

    for p in best_polynomials {
        let mut best_improving_polynomial: Option<Polynomial> = None;

        let mut testpolynomials: Vec<Polynomial> = (0..block_size)
            .collect_vec()
            .par_iter()
            .filter(|b| !p.contains(**b))
            .flat_map(|b| new_polys(*b, &p))
            .filter(|poly| !new_polynomials.contains(poly))
            .collect();

        testpolynomials.par_iter_mut().for_each(|poly| {
            poly.increase_count(data.par_iter().filter(|block| poly.evaluate(block)).count());
            poly.z_score(data.len());
        });
        for testpolynomial in testpolynomials {
            let new_z = testpolynomial.get_z_score().unwrap();

            if f64::abs(new_z) < f64::abs(p.z_score.unwrap())
                || testpolynomial
                    .get_count()
                    .abs_diff((testpolynomial.probability * (data.len() as f64)) as usize)
                    < min_difference
            {
                continue;
            }

            if best_improving_polynomial.is_none()
                || f64::abs(best_improving_polynomial.as_ref().unwrap().z_score.unwrap())
                    < f64::abs(new_z)
            {
                best_improving_polynomial = Some(testpolynomial);
            }
        }

        if let Some(imp_polynomial) = best_improving_polynomial {
            new_polynomials.push(imp_polynomial);
        } else {
            final_polynomials.push(p);
        }
    }

    new_polynomials
}

pub(crate) fn polyup(
    data: &[Vec<u8>],
    block_size: usize,
    n: usize,
    k: usize,
    min_difference: usize,
) -> Vec<Polynomial> {
    let mut start = Instant::now();
    let mut best_polynomials: Vec<Polynomial> = phase_one(data, k, block_size, n);
    println!("{best_polynomials:?}");

    println!("phase one {:.2?}", start.elapsed());
    start = Instant::now();

    let mut final_polynomials: Vec<Polynomial> = Vec::new();

    let mut current_length = n;
    while !best_polynomials.is_empty() && current_length < 15 {
        current_length += 1;
        best_polynomials = extend_polynomials(
            best_polynomials,
            block_size,
            data,
            min_difference,
            &mut final_polynomials,
        );
    }
    println!("phase two {:.2?}", start.elapsed());
    final_polynomials.extend(best_polynomials);
    final_polynomials
}

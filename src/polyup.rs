use crate::common::{bit_value_in_block, z_score};
use crate::distinguishers::{Distinguisher, Polynomial};
use rayon::prelude::*;

fn basic_zs(data: &[Vec<u8>], block_size: usize) -> Vec<f64> {
    let mut counts = vec![0; block_size];

    for block in data {
        for (i, count) in counts.iter_mut().enumerate() {
            if bit_value_in_block(i, block) {
                *count += 1;
            }
        }
    }
    counts
        .iter()
        .map(|c| z_score(data.len(), *c, 0.5))
        .collect()
}

fn top_n_bits(zs: &[f64], n: usize) -> Vec<(usize, f64)> {
    let mut top_n = vec![(0, 0.0); n];

    for (i, z) in zs.iter().enumerate() {
        if f64::abs(*z) > f64::abs(top_n[0].1) {
            top_n[0] = (i, *z);
            top_n.sort_by(|a, b| f64::abs(a.1).partial_cmp(&f64::abs(b.1)).unwrap());
        }
    }
    top_n
}

fn new_polys(bit: usize, bit_z: f64, old_polynomial: &Polynomial) -> Vec<Polynomial> {
    let mut testpolynomials = Vec::new();
    if bit_z.signum() == old_polynomial.z_score.unwrap().signum() {
        let mut testpolynomial: Polynomial = old_polynomial.clone();
        testpolynomial.and(bit, false).unwrap();
        testpolynomials.push(testpolynomial);
    } else {
        let mut testpolynomial: Polynomial = old_polynomial.clone();
        testpolynomial.and(bit, true).unwrap();
        testpolynomials.push(testpolynomial);
        let mut testpolynomial2: Polynomial = old_polynomial.clone();
        testpolynomial2.negate();
        testpolynomial2.and(bit, false).unwrap();
        testpolynomials.push(testpolynomial2);
    };
    testpolynomials
}

fn extend_polynomials(
    best_polynomials: Vec<Polynomial>,
    top_bits: &[(usize, f64)],
    data: &[Vec<u8>],
    min_count: usize,
    final_polynomials: &mut Vec<Polynomial>,
) -> Vec<Polynomial> {
    let mut new_polynomials: Vec<Polynomial> = Vec::new();

    for p in best_polynomials {
        let mut best_improving_polynomial: Option<Polynomial> = None;

        let mut testpolynomials: Vec<Polynomial> = top_bits
            .par_iter()
            .filter(|(b, _z)| !p.contains(*b))
            .flat_map(|(b, z)| new_polys(*b, *z, &p))
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
                    < min_count
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

fn create_polynomial(i: usize, z: f64) -> Polynomial {
    let mut poly = Polynomial::new();
    poly.and(i, z < 0.0).unwrap();
    poly.z_score = Some(z);
    poly
}

pub(crate) fn polyup(
    data: &[Vec<u8>],
    block_size: usize,
    n: usize,
    k: usize,
    min_count: usize,
) -> Vec<Polynomial> {
    let zs = basic_zs(data, block_size);

    let top_bits = top_n_bits(&zs, n);

    let mut best_polynomials: Vec<Polynomial> = top_bits
        .iter()
        .skip(n - k)
        .map(|(i, z)| create_polynomial(*i, *z))
        .collect();

    let mut final_polynomials: Vec<Polynomial> = Vec::new();

    let mut current_length = 1;
    while !best_polynomials.is_empty() && current_length < n {
        current_length += 1;
        best_polynomials = extend_polynomials(
            best_polynomials,
            &top_bits,
            data,
            min_count,
            &mut final_polynomials,
        );
    }

    final_polynomials
}

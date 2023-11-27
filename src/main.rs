mod common;
mod distinguishers;
mod polyup;

use crate::common::{load_data, Args};
use crate::distinguishers::{evaluate_distinguisher, Distinguisher};
use crate::polyup::polyup;
use clap::Parser;
use pyo3::prelude::*;
use std::time::Instant;

fn print_results(p_value: f64, z_score: f64, alpha: f64) {
    println!("---------------------------------------------------");
    println!("z-score: {z_score}");
    println!("p-value: {p_value:.0e}");
    if p_value >= alpha {
        println!(
            "As the p-value >= alpha {alpha:.0e}, the randomness hypothesis cannot be rejected."
        );
        println!("= CoolTest could not find statistically significant non-randomness.");
    } else {
        println!("As the p-value < alpha {alpha:.0e}, the randomness hypothesis is REJECTED.");
        println!("= Data is not random.");
    }
}

fn p_value(positive: usize, sample_size: usize, probability: f64) -> f64 {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let scipy = PyModule::import(py, "scipy").unwrap();
        let result: f64 = scipy
            .getattr("stats")
            .unwrap()
            .getattr("binom_test")
            .unwrap()
            .call1((positive, sample_size, probability, "two-sided"))
            .unwrap()
            .extract()
            .unwrap();
        result
    })
}

fn prepare_data(data_source: &str, block_size: usize) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let training_data = load_data(data_source, block_size);

    let (tr_data, testing_data) = training_data.split_at(training_data.len() / 2);

    (tr_data.to_vec(), testing_data.to_vec())
}

fn run_polyup(
    data_source: &str,
    block_size: usize,
    k: usize,
    n: usize,
    min_difference: usize,
    alpha: f64,
) {
    let (training_data, testing_data) = prepare_data(data_source, block_size);

    let _start = Instant::now();
    let mut final_patterns = polyup(&training_data, block_size, n, k, min_difference);
    final_patterns.sort_by(|a, b| {
        f64::abs(b.z_score.unwrap())
            .partial_cmp(&f64::abs(a.z_score.unwrap()))
            .unwrap()
    });

    //println!("{final_patterns:?}");
    println!("{:?}", final_patterns[0]);
    println!("{}", final_patterns[0].z_score.unwrap());

    print_results(
        p_value(
            final_patterns[0].get_count(),
            testing_data.len(),
            final_patterns[0].probability,
        ),
        evaluate_distinguisher(&mut final_patterns[0], &testing_data),
        alpha,
    )
}

fn main() {
    let args = Args::parse();

    run_polyup(
        &args.data_source,
        args.block_size,
        args.k,
        args.deg,
        args.min_difference,
        args.alpha,
    )
}

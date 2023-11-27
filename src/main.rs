mod bottomup;
mod common;
mod distinguishers;

use crate::bottomup::bottomup;
use crate::common::{load_data, Args};
use crate::distinguishers::{evaluate_distinguisher, Distinguisher};
use clap::Parser;
use pyo3::prelude::*;

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

fn run_bottomup(
    data_source: &str,
    block_size: usize,
    k: usize,
    min_difference: usize,
    patterns_combined: usize,
    base_pattern_size: usize,
) {
    let (training_data, testing_data) = prepare_data(data_source, block_size);

    let mut final_patterns = bottomup(
        &training_data,
        block_size,
        k,
        patterns_combined,
        min_difference,
        base_pattern_size,
    );

    println!(
        "z-score: {}",
        evaluate_distinguisher(&mut final_patterns, &testing_data)
    );
    println!(
        "p-value: {:.0e}",
        p_value(
            final_patterns.get_count(),
            testing_data.len(),
            final_patterns.probability
        )
    );
}

fn main() {
    let args = Args::parse();

    run_bottomup(
        &args.data_source,
        args.block_size,
        args.k,
        args.min_difference,
        args.n,
        args.deg,
    )
}

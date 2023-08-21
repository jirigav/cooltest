mod bottomup;
mod common;
mod distinguishers;
mod fastup;
mod polyup;

use crate::bottomup::bottomup;
use crate::common::*;
use crate::distinguishers::*;
use crate::fastup::fastup;
use crate::polyup::polyup;
use clap::Parser;
use pyo3::prelude::*;
use std::time::Instant;

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

fn prepare_data(
    data_source: String,
    block_size: usize,
    halving: bool,
) -> (Vec<Vec<u8>>, Option<Vec<Vec<u8>>>) {
    let mut training_data = load_data(&data_source, block_size);
    let mut testing_data_option = None;

    if halving {
        let (tr_data, testing_data) = training_data.split_at(training_data.len() / 2);
        testing_data_option = Some(testing_data.to_vec());
        training_data = tr_data.to_vec();
    }
    (training_data, testing_data_option)
}

fn results(
    mut final_patterns: Vec<Pattern>,
    start: Instant,
    training_data: &[Vec<u8>],
    testing_data_option: Option<Vec<Vec<u8>>>,
    evaluated_disses: usize,
    patterns_combined: usize,
) {
    final_patterns.sort_by(|a, b| {
        f64::abs(b.z_score.unwrap())
            .partial_cmp(&f64::abs(a.z_score.unwrap()))
            .unwrap()
    });
    let abs_z_1 = f64::abs(final_patterns[0].z_score.unwrap());

    let mut b_multi_pattern = best_multi_pattern(training_data, &final_patterns, patterns_combined);
    let abs_z_2 = f64::abs(b_multi_pattern.z_score.unwrap());

    println!("trained in {:.2?}", start.elapsed());

    println!(
        "total number of distinguishers evaluated: {}",
        evaluated_disses
    );

    if abs_z_1 > abs_z_2 {
        println!("z-score: {}", final_patterns[0].z_score.unwrap());
        println!("best pattern: {:?}", final_patterns[0])
    } else {
        println!("z-score: {}", b_multi_pattern.z_score.unwrap());
        println!("best multi-pattern: {:?}", b_multi_pattern)
    }

    if let Some(testing_data) = testing_data_option {
        if abs_z_1 > abs_z_2 {
            println!(
                "z-score: {}",
                evaluate_distinguisher(&mut final_patterns[0], &testing_data)
            );
            println!(
                "p-value: {:.0e}",
                p_value(
                    final_patterns[0].count.unwrap(),
                    testing_data.len(),
                    2.0_f64.powf(-(final_patterns[0].length as f64))
                )
            );
        } else {
            println!(
                "z-score: {}",
                evaluate_distinguisher(&mut b_multi_pattern, &testing_data)
            );
            println!(
                "p-value: {:.0e}",
                p_value(
                    b_multi_pattern.get_count(),
                    testing_data.len(),
                    b_multi_pattern.probability
                )
            );
        }
    }
}

fn run_bottomup(
    data_source: String,
    block_size: usize,
    k: usize,
    min_count: usize,
    patterns_combined: usize,
    halving: bool,
) {
    let (training_data, testing_data_option) = prepare_data(data_source, block_size, halving);

    let start = Instant::now();
    let (final_patterns, evaluated_disses) = bottomup(&training_data, block_size, k, min_count);
    results(
        final_patterns,
        start,
        &training_data,
        testing_data_option,
        evaluated_disses,
        patterns_combined,
    );
}

fn run_fastup(
    data_source: String,
    block_size: usize,
    k: usize,
    n: usize,
    min_count: usize,
    patterns_combined: usize,
    halving: bool,
) {
    let (training_data, testing_data_option) = prepare_data(data_source, block_size, halving);

    let start = Instant::now();
    let (final_patterns, evaluated_disses) = fastup(&training_data, block_size, n, k, min_count);
    results(
        final_patterns,
        start,
        &training_data,
        testing_data_option,
        evaluated_disses,
        patterns_combined,
    );
}

fn run_polyup(
    data_source: String,
    block_size: usize,
    k: usize,
    n: usize,
    min_count: usize,
    halving: bool,
) {
    let (training_data, testing_data_option) = prepare_data(data_source, block_size, halving);

    let _start = Instant::now();
    let (mut final_patterns, _evaluated_disses) =
        polyup(&training_data, block_size, n, k, min_count);
    final_patterns.sort_by(|a, b| {
        f64::abs(b.z_score.unwrap())
            .partial_cmp(&f64::abs(a.z_score.unwrap()))
            .unwrap()
    });
    /* results(
        final_patterns,
        start,
        &training_data,
        testing_data_option,
        evaluated_disses,
    );*/
    println!("{:?}", final_patterns[0].monomials);
    println!("{}", final_patterns[0].z_score.unwrap());

    if let Some(testing_data) = testing_data_option {
        println!(
            "z-score: {}",
            evaluate_distinguisher(&mut final_patterns[0], &testing_data)
        );
        println!(
            "p-value: {:.0e}",
            p_value(
                final_patterns[0].get_count(),
                testing_data.len(),
                final_patterns[0].probability
            )
        );
    }
}

fn main() {
    let args = Args::parse();

    match args.tool {
        Subcommands::ShuffleData {
            block_size,
            input_file_path,
            output_file_path,
        } => shuffle_data(&input_file_path, &output_file_path, block_size),
        Subcommands::Bottomup {
            data_source,
            block_size,
            k,
            min_count,
            patterns_combined,
            halving,
        } => run_bottomup(
            data_source,
            block_size,
            k,
            min_count,
            patterns_combined,
            halving,
        ),
        Subcommands::Fastup {
            data_source,
            block_size,
            k,
            n,
            min_count,
            patterns_combined,
            halving,
        } => run_fastup(
            data_source,
            block_size,
            k,
            n,
            min_count,
            patterns_combined,
            halving,
        ),
        Subcommands::Polyup {
            data_source,
            block_size,
            k,
            n,
            min_count,
            halving,
        } => run_polyup(data_source, block_size, k, n, min_count, halving),
    }
}

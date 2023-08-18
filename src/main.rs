mod bottomup;
mod common;
mod fastup;
mod patterns;

use crate::bottomup::bottomup;
use crate::common::*;
use crate::fastup::fastup;
use crate::patterns::*;
use clap::Parser;
use std::fmt::Debug;
use std::time::Instant;

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
) {
    final_patterns.sort_by(|a, b| {
        f64::abs(b.z_score.unwrap())
            .partial_cmp(&f64::abs(a.z_score.unwrap()))
            .unwrap()
    });
    let mut b_double_pattern = best_double_pattern(training_data, &final_patterns);
    println!("trained in {:.2?}", start.elapsed());

    println!(
        "total number of distinguishers evaluated: {}",
        evaluated_disses
    );
    if f64::abs(final_patterns[0].z_score.unwrap()) > f64::abs(b_double_pattern.z_score.unwrap()) {
        println!("z-score: {}", final_patterns[0].z_score.unwrap());
        println!("best pattern: {:?}", final_patterns[0])
    } else {
        println!("z-score: {}", b_double_pattern.z_score.unwrap());
        println!("best double pattern: {:?}", b_double_pattern)
    }

    if let Some(testing_data) = testing_data_option {
        if f64::abs(final_patterns[0].z_score.unwrap())
            > f64::abs(b_double_pattern.z_score.unwrap())
        {
            print_result(&mut final_patterns[0], &testing_data);
        } else {
            print_result(&mut b_double_pattern, &testing_data);
        }
    }
}

fn print_result<P: GeneralizedPattern + Debug>(pattern: &mut P, data: &[Vec<u8>]) {
    println!("z-score: {}", evaluate_pattern(pattern, data));
}

fn run_bottomup(data_source: String, block_size: usize, k: usize, min_count: usize, halving: bool) {
    let (training_data, testing_data_option) = prepare_data(data_source, block_size, halving);

    let start = Instant::now();
    let (final_patterns, evaluated_disses) = bottomup(&training_data, block_size, k, min_count);
    results(
        final_patterns,
        start,
        &training_data,
        testing_data_option,
        evaluated_disses,
    );
}

fn run_fastup(
    data_source: String,
    block_size: usize,
    k: usize,
    n: usize,
    min_count: usize,
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
    );
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
            halving,
        } => run_bottomup(data_source, block_size, k, min_count, halving),
        Subcommands::Fastup {
            data_source,
            block_size,
            k,
            n,
            min_count,
            halving,
        } => run_fastup(data_source, block_size, k, n, min_count, halving),
    }
}

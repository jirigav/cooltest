mod bottomup;
mod common;
mod fastup;

use crate::bottomup::bottomup;
use crate::common::*;
use crate::fastup::fastup;
use clap::Parser;
use std::time::Instant;

fn run_bottomup(data_source: String, block_size: usize, k: usize, min_count: usize, halving: bool) {
    let data = load_data(&data_source, block_size);

    if halving {
        let (training_data, testing_data) = data.split_at(data.len() / 2);

        let start = Instant::now();
        let mut final_patterns = bottomup(training_data, block_size, k, min_count);
        final_patterns.sort_by(|a, b| {
            f64::abs(b.z_score.unwrap())
                .partial_cmp(&f64::abs(a.z_score.unwrap()))
                .unwrap()
        });
        let mut b_double_pattern = best_double_pattern(training_data, &final_patterns);
        println!("bottomup trained in {:.2?}", start.elapsed());

        if f64::abs(final_patterns[0].z_score.unwrap())
            > f64::abs(b_double_pattern.z_score.unwrap())
        {
            println!(
                "z-score: {}",
                evaluate_pattern(&mut final_patterns[0], testing_data)
            );
            println!("best pattern: {:?}", final_patterns[0])
        } else {
            println!(
                "z-score: {}",
                evaluate_double_pattern(&mut b_double_pattern, testing_data)
            );
            println!("best double pattern: {:?}", b_double_pattern)
        }
    } else {
        let start = Instant::now();
        let mut final_patterns = bottomup(&data, block_size, k, min_count);
        final_patterns.sort_by(|a, b| {
            f64::abs(b.z_score.unwrap())
                .partial_cmp(&f64::abs(a.z_score.unwrap()))
                .unwrap()
        });
        let b_double_pattern = best_double_pattern(&data, &final_patterns);
        println!("bottomup trained in {:.2?}", start.elapsed());

        if f64::abs(final_patterns[0].z_score.unwrap())
            > f64::abs(b_double_pattern.z_score.unwrap())
        {
            println!("z-score: {}", final_patterns[0].z_score.unwrap());
            println!("best pattern: {:?}", final_patterns[0])
        } else {
            println!("z-score: {}", b_double_pattern.z_score.unwrap());
            println!("best double pattern: {:?}", b_double_pattern)
        }
    }
}

fn run_fastup(
    data_source: String,
    block_size: usize,
    k: usize,
    n: usize,
    min_count: usize,
    halving: bool,
) {
    let data = load_data(&data_source, block_size);

    if halving {
        let (training_data, testing_data) = data.split_at(data.len() / 2);
        let start = Instant::now();
        let mut final_patterns = fastup(training_data, block_size, n, k, min_count);
        final_patterns.sort_by(|a, b| {
            f64::abs(b.z_score.unwrap())
                .partial_cmp(&f64::abs(a.z_score.unwrap()))
                .unwrap()
        });
        let mut b_double_pattern = best_double_pattern(training_data, &final_patterns);
        println!("bottomup trained in {:.2?}", start.elapsed());

        if f64::abs(final_patterns[0].z_score.unwrap())
            > f64::abs(b_double_pattern.z_score.unwrap())
        {
            println!(
                "z-score: {}",
                evaluate_pattern(&mut final_patterns[0], testing_data)
            );
            println!("best pattern: {:?}", final_patterns[0])
        } else {
            println!(
                "z-score: {}",
                evaluate_double_pattern(&mut b_double_pattern, testing_data)
            );
            println!("best double pattern: {:?}", b_double_pattern)
        }
    } else {
        let start = Instant::now();
        let mut final_patterns = fastup(&data, block_size, n, k, min_count);
        final_patterns.sort_by(|a, b| {
            f64::abs(b.z_score.unwrap())
                .partial_cmp(&f64::abs(a.z_score.unwrap()))
                .unwrap()
        });
        let b_double_pattern = best_double_pattern(&data, &final_patterns);
        println!("bottomup trained in {:.2?}", start.elapsed());

        if f64::abs(final_patterns[0].z_score.unwrap())
            > f64::abs(b_double_pattern.z_score.unwrap())
        {
            println!("z-score: {}", final_patterns[0].z_score.unwrap());
            println!("best pattern: {:?}", final_patterns[0])
        } else {
            println!("z-score: {}", b_double_pattern.z_score.unwrap());
            println!("best double pattern: {:?}", b_double_pattern)
        }
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

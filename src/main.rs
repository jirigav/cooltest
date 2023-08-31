mod bottomup;
mod common;
mod distinguishers;
mod polyup;

use crate::bottomup::bottomup;
use crate::common::{bits_block_eval, load_data, shuffle_data, z_score, Args, Subcommands};
use crate::distinguishers::{
    best_multi_pattern, evaluate_distinguisher, Distinguisher, MultiPattern, Pattern,
};
use crate::polyup::polyup;
use clap::Parser;
use itertools::Itertools;
use pyo3::prelude::*;
use std::collections::HashSet;
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
    data_source: &str,
    block_size: usize,
    halving: bool,
    validation: bool,
) -> (Vec<Vec<u8>>, Option<Vec<Vec<u8>>>, Option<Vec<Vec<u8>>>) {
    let mut training_data = load_data(data_source, block_size);
    let mut testing_data_option = None;
    let mut validation_data_option = None;

    if validation {
        let (tr_data, testing_data) = training_data.split_at(training_data.len() / 3);
        let (val_data, test_data) = testing_data.split_at(testing_data.len() / 2);
        testing_data_option = Some(test_data.to_vec());
        validation_data_option = Some(val_data.to_vec());
        training_data = tr_data.to_vec();
    } else if halving {
        let (tr_data, testing_data) = training_data.split_at(training_data.len() / 2);
        testing_data_option = Some(testing_data.to_vec());
        training_data = tr_data.to_vec();
    }
    (training_data, validation_data_option, testing_data_option)
}

fn results(
    mut final_patterns: Vec<Pattern>,
    start: Instant,
    training_data: &[Vec<u8>],
    testing_data_option: Option<&Vec<Vec<u8>>>,
    validation_data_option: Option<&Vec<Vec<u8>>>,
    patterns_combined: usize,
) {
    final_patterns.sort_by(|a, b| {
        f64::abs(b.z_score.unwrap())
            .partial_cmp(&f64::abs(a.z_score.unwrap()))
            .unwrap()
    });
    let mut b_multi_pattern: MultiPattern;
    if let Some(validation_data) = validation_data_option {
        b_multi_pattern = best_multi_pattern(validation_data, &final_patterns, patterns_combined);
    } else {
        b_multi_pattern = best_multi_pattern(training_data, &final_patterns, patterns_combined);
    }

    println!("trained in {:.2?}", start.elapsed());

    println!("z-score: {}", b_multi_pattern.z_score.unwrap());
    println!("best multi-pattern: {b_multi_pattern:?}");

    if let Some(testing_data) = testing_data_option {
        println!(
            "z-score: {}",
            evaluate_distinguisher(&mut b_multi_pattern, testing_data)
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

fn get_hist(bits: &Vec<usize>, data: &[Vec<u8>]) -> Vec<usize> {
    let mut hist = vec![0; 2_usize.pow(bits.len() as u32)];
    for block in data {
        hist[bits_block_eval(bits, block)] += 1;
    }
    hist
}
fn results2(
    mut final_patterns: Vec<Pattern>,
    start: Instant,
    training_data: &[Vec<u8>],
    testing_data_option: Option<&Vec<Vec<u8>>>,
) {
    println!("\n-- histograms --\n");
    final_patterns.sort_by(|a, b| {
        f64::abs(b.z_score.unwrap())
            .partial_cmp(&f64::abs(a.z_score.unwrap()))
            .unwrap()
    });
    let mut bits = HashSet::new();

    for p in final_patterns.iter().take(1) {
        bits.extend(p.bits.clone());
    }

    let mut bits_vec: Vec<usize> = bits.into_iter().collect();
    bits_vec.sort();

    println!("number of bits: {}", bits_vec.len());
    if bits_vec.len() > 16 {
        panic!();
    }

    let hist = get_hist(&bits_vec, training_data);

    let mut indices = (0..2_usize.pow(bits_vec.len() as u32)).collect_vec();
    indices.sort_by(|a, b| hist[*b].cmp(&hist[*a]));

    let mut max_z = 0.0;
    let mut best_i = 0;
    let prob = 2.0_f64.powf(-(bits_vec.len() as f64));

    for i in 1..2_usize.pow(bits_vec.len() as u32) {
        let mut count = 0;
        for k in 0..i {
            count += hist[indices[k]];
        }
        let z = z_score(training_data.len(), count, prob * (i as f64));
        if z > max_z {
            max_z = z;
            best_i = i;
        }
    }
    println!("trained in {:.2?}", start.elapsed());

    println!("z-score: {}", max_z);

    if let Some(testing_data) = testing_data_option {
        let test_hist = get_hist(&bits_vec, testing_data);

        let mut count = 0;
        for k in 0..best_i {
            count += test_hist[indices[k]];
        }
        let z = z_score(testing_data.len(), count, prob * (best_i as f64));

        println!("z-score: {}", z);
        println!(
            "p-value: {:.0e}",
            p_value(count, testing_data.len(), prob * (best_i as f64))
        );
    }
}

fn run_bottomup(
    data_source: &str,
    block_size: usize,
    k: usize,
    min_difference: usize,
    patterns_combined: usize,
    base_pattern_size: usize,
    halving: bool,
    validation: bool,
) {
    let (training_data, validation_data_option, testing_data_option) =
        prepare_data(data_source, block_size, halving, validation);

    let start = Instant::now();
    let final_patterns = bottomup(
        &training_data,
        block_size,
        k,
        min_difference,
        base_pattern_size,
    );
    results(
        final_patterns.clone(),
        start,
        &training_data,
        testing_data_option.as_ref(),
        validation_data_option.as_ref(),
        patterns_combined,
    );
    results2(
        final_patterns,
        start,
        &training_data,
        testing_data_option.as_ref(),
    );
}

fn run_polyup(
    data_source: &str,
    block_size: usize,
    k: usize,
    n: usize,
    min_difference: usize,
    halving: bool,
) {
    let (training_data, _validation_data_option, testing_data_option) =
        prepare_data(data_source, block_size, halving, false);

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
            min_difference,
            patterns_combined,
            base_pattern_size,
            halving,
            validation_and_testing_split,
        } => run_bottomup(
            &data_source,
            block_size,
            k,
            min_difference,
            patterns_combined,
            base_pattern_size,
            halving,
            validation_and_testing_split,
        ),
        Subcommands::Polyup {
            data_source,
            block_size,
            k,
            n,
            min_difference,
            halving,
        } => run_polyup(&data_source, block_size, k, n, min_difference, halving),
    }
}

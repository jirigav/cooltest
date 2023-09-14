mod bottomup;
mod common;
mod distinguishers;

use crate::bottomup::bottomup;
use crate::common::{p_value, z_score, Args};
use crate::distinguishers::{
    best_multi_pattern, evaluate_distinguisher, Distinguisher, Histogram, MultiPattern, Pattern,
};

use clap::Parser;
use common::prepare_data;
use std::collections::HashSet;
use std::time::Instant;

fn results(
    mut final_patterns: Vec<Pattern>,
    start: Instant,
    training_data: &[Vec<u8>],
    testing_data_option: Option<&Vec<Vec<u8>>>,
    patterns_combined: usize,
    hist: bool,
) {
    final_patterns.sort_by(|a, b| {
        f64::abs(b.z_score.unwrap())
            .partial_cmp(&f64::abs(a.z_score.unwrap()))
            .unwrap()
    });
    let mut b_multi_pattern: MultiPattern;

    b_multi_pattern = best_multi_pattern(training_data, &final_patterns, patterns_combined);

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
    if hist {
        hist_result(final_patterns, training_data, testing_data_option);
    }
}

fn hist_result(
    final_patterns: Vec<Pattern>,
    training_data: &[Vec<u8>],
    testing_data_option: Option<&Vec<Vec<u8>>>,
) {
    println!("\n-- histograms --\n");
    let mut bits = HashSet::new();

    for p in final_patterns.iter().take(1) {
        bits.extend(p.bits.clone());
    }

    let mut bits_vec: Vec<usize> = bits.into_iter().collect();
    bits_vec.sort();

    println!("number of bits: {}", bits_vec.len());

    if bits_vec.len() > 16 {
        println!("Too many bits in pattern, can't produce hist result.");
        return;
    }

    let hist = Histogram::get_hist(&bits_vec, training_data);

    println!("z-score: {}", hist.z_score);

    if let Some(testing_data) = testing_data_option {
        let test_hist = Histogram::get_hist(&hist.bits, testing_data);

        let mut count = 0;
        for k in 0..hist.best_division {
            count += test_hist._bins[hist.sorted_indices[k]];
        }
        let prob = 2.0_f64.powf(-(hist.bits.len() as f64));
        let z = z_score(
            testing_data.len(),
            count,
            prob * (hist.best_division as f64),
        );

        println!("z-score: {}", z);
        println!(
            "p-value: {:.0e}",
            p_value(
                count,
                testing_data.len(),
                prob * (hist.best_division as f64)
            )
        );
    }
}

fn run_bottomup(args: Args) {
    let (training_data, validation_data_option, testing_data_option) = prepare_data(
        &args.data_source,
        args.block_size,
        args.halving,
        args.validation_and_testing_split,
    );

    let start = Instant::now();
    let final_patterns = bottomup(&training_data, validation_data_option.as_ref(), &args);
    results(
        final_patterns.clone(),
        start,
        &training_data,
        testing_data_option.as_ref(),
        args.patterns_combined,
        args.hist,
    );
}

fn main() {
    let args = Args::parse();
    println!("\n{args:?}\n");

    run_bottomup(args);
}

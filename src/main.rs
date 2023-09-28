mod bottomup;
mod common;
mod distinguishers;

use crate::bottomup::bottomup;
use crate::common::{p_value, z_score, Args};
use crate::distinguishers::{
    best_multi_pattern, evaluate_distinguisher, Distinguisher, Histogram, Pattern,
};

use clap::Parser;
use common::{prepare_data, Data};
use std::time::Instant;

fn print_results(p_value: f64, z_score: f64) {
    println!("z-score: {z_score}");
    println!("p-value: {p_value:.0e}");
}

fn results(
    mut final_patterns: Vec<Pattern>,
    start: Instant,
    training_data: &Data,
    testing_data: &Data,
    patterns_combined: usize,
    hist: bool,
) {
    final_patterns.sort_by(|a, b| {
        f64::abs(b.z_score.unwrap())
            .partial_cmp(&f64::abs(a.z_score.unwrap()))
            .unwrap()
    });
    let mut best_mp = best_multi_pattern(training_data, &final_patterns, patterns_combined);

    println!("trained in {:.2?}", start.elapsed());

    println!("training z-score: {}", best_mp.z_score.unwrap());
    println!("best multi-pattern: {best_mp:?}");

    if hist {
        hist_result(final_patterns, training_data, testing_data);
    } else {
        let z_score = evaluate_distinguisher(&mut best_mp, testing_data);
        let p_value = p_value(
            best_mp.get_count(),
            testing_data.num_of_blocks,
            best_mp.probability,
        );
        print_results(p_value, z_score);
    }
}

fn hist_result(final_patterns: Vec<Pattern>, training_data: &Data, testing_data: &Data) {
    let bits = final_patterns[0].bits.clone();

    println!("number of bits: {}", bits.len());

    if bits.len() > 20 {
        println!("Too many bits in pattern, can't produce hist result.");
        return;
    }

    let hist = Histogram::get_hist(&bits, training_data);

    println!("training z-score: {}", hist.z_score);

    let count = hist.evaluate(testing_data);
    let prob = 2.0_f64.powf(-(hist.bits.len() as f64)) * (hist.best_division as f64);

    let z = z_score(testing_data.num_of_blocks, count, prob);
    let p_val = p_value(count, testing_data.num_of_blocks, prob);

    print_results(p_val, z);
}

fn run_bottomup(args: Args) {
    let s = Instant::now();
    let (training_data, validation_data_option, testing_data_option) = prepare_data(
        &args.data_source,
        args.block_size,
        true,
        args.validation_and_testing_split,
    );
    println!("data loaded in: {:?}", s.elapsed());

    let start = Instant::now();
    let final_patterns = bottomup(&training_data, validation_data_option.as_ref(), &args);
    results(
        final_patterns.clone(),
        start,
        &training_data,
        &testing_data_option.unwrap(),
        args.patterns_combined,
        args.hist,
    );
}

fn main() {
    let args = Args::parse();
    println!("\n{args:?}\n");

    run_bottomup(args);
}

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
use itertools::Itertools;
use std::fs::File;
use std::io::{BufRead, BufReader};
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

fn results(
    mut final_patterns: Vec<Pattern>,
    start: Instant,
    training_data: &Data,
    testing_data: &Data,
    args: Args,
) -> (f64, f64) {
    final_patterns.sort_by(|a, b| {
        f64::abs(b.z_score.unwrap())
            .partial_cmp(&f64::abs(a.z_score.unwrap()))
            .unwrap()
    });

    if args.hist {
        hist_result(
            final_patterns,
            training_data,
            testing_data,
            start,
            args.alpha,
        )
    } else {
        let mut best_mp = best_multi_pattern(
            training_data,
            &final_patterns.into_iter().take(args.top_n).collect_vec(),
            args.patterns_combined,
        );

        println!("Trained in {:.2?}.", start.elapsed());

        println!("Training z-score: {}.", best_mp.z_score.unwrap());
        println!("Best multi-pattern: {best_mp:?}");
        let z_score = evaluate_distinguisher(&mut best_mp, testing_data);
        let p_value = p_value(
            best_mp.get_count(),
            testing_data.num_of_blocks,
            best_mp.probability,
        );
        print_results(p_value, z_score, args.alpha);
        (p_value, z_score)
    }
}

fn hist_result(
    final_patterns: Vec<Pattern>,
    training_data: &Data,
    testing_data: &Data,
    start: Instant,
    alpha: f64,
) -> (f64, f64) {
    let bits = final_patterns[0].bits.clone();

    println!("Number of bits in histogram: {}.", bits.len());
    println!("Selected bits: {:?}", bits);

    if bits.len() > 20 {
        println!("Too many bits in pattern, can't produce hist result.");
        return (1.0, 0.0);
    }

    let hist = Histogram::get_hist(&bits, training_data);

    println!("Trained in {:.2?}.", start.elapsed());
    println!("Training z-score: {}.", hist.z_score);

    let count = hist.evaluate(testing_data);
    let prob = 2.0_f64.powf(-(hist.bits.len() as f64)) * (hist.best_division as f64);

    let z = z_score(testing_data.num_of_blocks, count, prob);
    let p_val = p_value(count, testing_data.num_of_blocks, prob);
    print_results(p_val, z, alpha);
    (p_val, z)
}

fn run_bottomup(args: Args) -> (f64, f64) {
    let s = Instant::now();
    let (training_data, validation_data_option, testing_data_option) = prepare_data(
        &args.data_source,
        args.block,
        args.block_size_multiple,
        true,
        args.validation_and_testing_split,
    );
    println!("Data loaded in {:?}.", s.elapsed());

    let start = Instant::now();
    let final_patterns = bottomup(&training_data, validation_data_option.as_ref(), &args);
    results(
        final_patterns.clone(),
        start,
        &training_data,
        &testing_data_option.unwrap(),
        args,
    )
}

fn parse_args(s: Vec<&str>) -> Args {
    Args {
        data_source: s[0].to_string(),
        block: s[1].trim().parse().unwrap(),
        block_size_multiple: s[2].trim().parse().unwrap(),
        k: s[3].trim().parse().unwrap(),
        min_difference: s[4].trim().parse().unwrap(),
        top_n: s[5].trim().parse().unwrap(),
        max_bits: Some(s[6].trim().parse().unwrap()),
        patterns_combined: s[7].trim().parse().unwrap(),
        deg: s[8].trim().parse().unwrap(),
        validation_and_testing_split: s[9].trim().parse().unwrap(),
        hist: s[10].trim().parse().unwrap(),
        alpha: 0.0004,
        config: false,
    }
}

fn main() {
    let args = Args::parse();
    if args.config {
        let file = File::open(args.data_source).unwrap();

        let reader = BufReader::new(file);
        let mut results = Vec::new();
        for line in reader.lines() {
            let l = line.unwrap();
            println!("config: {l}");
            let splitted = l.split(',').collect_vec();
            let args = parse_args(splitted);
            results.push(run_bottomup(args));
            println!();
        }
        println!("{results:?}");
    } else {
        println!("\n{args:?}\n");
        run_bottomup(args);
    }
}

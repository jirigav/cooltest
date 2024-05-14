mod bottomup;
mod common;

use crate::bottomup::bottomup;
use crate::common::{p_value, z_score, Args};
use bottomup::Histogram;
use clap::Parser;
use common::prepare_data;
use std::time::Instant;

fn print_results(p_value: f64, z_score: f64, alpha: f64, hist: Histogram, bins: Vec<usize>) {
    println!("----------------------------------------------------------------------");
    println!("RESULTS:\n");

    println!("Histogram(the discovered Boolean function returns 1 for values before the separator and 0 for values after the separator.):\n");
    let m = bins.iter().max().unwrap();
    let unit = (m / 50).max(1);
    for (i, ind) in hist.sorted_indices.iter().enumerate() {
        for x in &hist.bits {
            print!("x{} ", x);
        }
        let mut j = *ind;
        print!("| [");
        for _ in 0..hist.bits.len() {
            print!("{}", j % 2);
            j /= 2;
        }
        print!("] | ");
        for _ in 0..bins[*ind] / unit {
            print!("∎");
        }
        println!();
        if i == (hist.best_division - 1) {
            for _ in 0..80 {
                print!("—");
            }
            println!();
        }
    }
    println!();
    println!("Z-score: {z_score}");
    println!("P-value: {p_value:.0e}");
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

fn run_bottomup(args: Args) {
    let (training_data, testing_data) = prepare_data(&args.data_source, args.block);

    let start = Instant::now();
    let hist = bottomup(
        &training_data,
        args.block,
        args.k,
        args.top,
        args.max_bits,
        args.threads,
    );
    println!("training finished in {:?}", start.elapsed());

    let (count, bins) = hist.evaluate(&testing_data);
    let prob = 2.0_f64.powf(-(hist.bits.len() as f64));
    let z = z_score(
        testing_data.len(),
        count,
        prob * (hist.best_division as f64),
    );
    print_results(
        p_value(
            count,
            testing_data.len(),
            prob * (hist.best_division as f64),
        ),
        z,
        args.alpha,
        hist,
        bins,
    );
}

fn main() {
    let args = Args::parse();
    println!("\n{args:?}\n");

    run_bottomup(args)
}

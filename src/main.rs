mod bottomup;
mod common;

use crate::bottomup::bottomup;
use crate::common::{p_value, z_score, Args};
use clap::Parser;
use common::prepare_data;
use std::time::Instant;

fn run_bottomup(
    data_source: &str,
    block_size: usize,
    k: usize,
    base_pattern_size: usize,
    validation: bool,
    max_bits: usize,
    stop_p_value: f64,
) {
    let (training_data, _validation_data_option, testing_data_option) =
        prepare_data(data_source, block_size, true, validation);

    let start = Instant::now();
    let hist = bottomup(
        &training_data,
        block_size,
        k,
        base_pattern_size,
        max_bits,
        stop_p_value,
    );
    println!("training finished in {:?}", start.elapsed());
    let testing_data = testing_data_option.unwrap();

    let count = hist.evaluate(&testing_data);
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

fn main() {
    let args = Args::parse();
    println!("\n{args:?}\n");

    run_bottomup(
        &args.data_source,
        args.block_size,
        args.k,
        args.base_pattern_size,
        args.validation_and_testing_split,
        args.max_bits,
        args.stop_p_value,
    )
}

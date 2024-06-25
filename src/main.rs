mod bottomup;
mod common;
mod results;

use crate::bottomup::bottomup;
use crate::common::Args;
use bottomup::Histogram;
use clap::Parser;
use common::{prepare_data, SubCommand};
use std::fs;
use std::time::Instant;
use results::results;

fn run_bottomup(args: Args) {
    let (training_data, testing_data) = prepare_data(&args.data_source, args.block, true);
    let testing_data = testing_data.unwrap();

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

    results(hist, &testing_data, args)
}

fn main() {
    let args = Args::parse();
    println!("\n{args:?}\n");

    match args.subcommand.clone() {
        Some(SubCommand::Evaluate { dis_path }) => {
            let contents = fs::read_to_string(&dis_path)
                .unwrap_or_else(|_| panic!("Failed to read contents of {}", &dis_path));
            let hist: Histogram =
                serde_json::from_str(&contents).expect("Invalid distinguisher json!");
            let (testing_data, _) = prepare_data(&args.data_source, args.block, false);
            results(hist, &testing_data, args)
        }
        Some(SubCommand::Autotest {}) => todo!(),
        None => run_bottomup(args),
    }
}

mod autotest;
mod bottomup;
mod common;
mod results;
#[cfg(test)]
mod tests;

use crate::bottomup::bottomup;
use crate::common::Args;
use autotest::autotest;
use bottomup::Histogram;
use clap::Parser;
use common::{
    load_data, set_quiet, status, validate_args, warn_if_underpowered, whole_blocks, Res,
    SubCommand,
};
use results::results;
use std::fs;
use std::time::Instant;

/// Searches the first half of the data for a distinguisher and evaluates it on the second half.
/// Returns whether the randomness hypothesis was rejected.
fn run_bottomup(args: &Args) -> Res<bool> {
    let data = load_data(&args.data_source)?;
    let (training_data, testing_data) = whole_blocks(&data, args.block, true)?;
    warn_if_underpowered(testing_data.len() / 8 / args.block, args.alpha);

    let start = Instant::now();
    let hist = bottomup(training_data, args.block, args.k, args.threads)?;
    status!("Search for distinguisher finished in {:?}", start.elapsed());

    results(&hist, testing_data, args)
}

fn run_evaluate(args: &mut Args, dis_path: &str) -> Res<bool> {
    let contents =
        fs::read_to_string(dis_path).map_err(|e| format!("failed to read '{dis_path}': {e}"))?;
    let hist: Histogram = serde_json::from_str(&contents)
        .map_err(|e| format!("invalid distinguisher json in '{dis_path}': {e}"))?;
    hist.validate()?;

    args.block = hist.block_size;
    args.k = hist.bits.len();

    let data = load_data(&args.data_source)?;
    let (testing_data, _) = whole_blocks(&data, hist.block_size, false)?;
    warn_if_underpowered(testing_data.len() / (hist.block_size / 8), args.alpha);
    status!(
        "The whole input file is used for evaluation. The p-value is only valid if this file is \
         disjoint from the data the distinguisher was trained on."
    );

    results(&hist, testing_data, args)
}

fn run(mut args: Args) -> Res<bool> {
    validate_args(&args)?;

    status!("\n{args:?}\n");

    if args.block > 600 {
        status!(
            "With block size {}, the computation can take long time, consider using smaller block size.",
            args.block
        );
    }

    match args.subcommand.clone() {
        Some(SubCommand::Evaluate { dis_path }) => run_evaluate(&mut args, &dis_path),
        Some(SubCommand::Autotest {}) => autotest(args),
        None => run_bottomup(&args),
    }
}

fn main() {
    let args = Args::parse();
    set_quiet(args.quiet);
    let exit_code = args.exit_code;

    match run(args) {
        Ok(rejected) => {
            if rejected && exit_code {
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("cooltest: error: {e}");
            std::process::exit(2);
        }
    }
}

#![allow(dead_code)]

mod bottomup;
mod common;
mod fastup;

use crate::bottomup::bottomup;
use crate::common::*;
use crate::fastup::fastup;
use std::time::Instant;
use clap::Parser;


fn print_all_zs(path: &str, patterns: &mut [Pattern], block_size: usize) {
    println!(
        "{:?}",
        patterns
            .iter_mut()
            .map(|p| evaluate_pattern(p, path, block_size))
            .collect::<Vec<_>>()
    );
}

fn print_all_zs_double(path: &str, patterns: &[DoublePattern], block_size: usize) {
    println!(
        "{:?}",
        patterns
            .iter()
            .map(|p| evaluate_double_pattern(p, path, block_size))
            .collect::<Vec<_>>()
    );
}

fn main() {
    let args = Args::parse();

    let data_path = &args.data_source;

    let training_data_path = "./data/train";
    let testing_data_path = "./data/test";
    let data_for_booltest_path = "./data/booltestdata";
    let mut start = Instant::now();
    split_data(data_path, training_data_path, testing_data_path, data_for_booltest_path, args.block_size);
    println!("data split in {:.2?}", start.elapsed());

    let data = load_data(training_data_path, args.block_size);

    start = Instant::now();
    let mut final_patterns = bottomup(&args, &data);
    println!("bottomup finished in {:.2?}", start.elapsed());

    let bu_double_patterns = best_double_patterns(&data, &final_patterns, 5);

    start = Instant::now();


    println!("evaluation on testing data");
    print_all_zs(testing_data_path, &mut final_patterns, args.block_size);

    println!("double patterns:");
    print_all_zs_double(testing_data_path, &bu_double_patterns, args.block_size);

    println!("evaluated in {:.2?}", start.elapsed());

    start = Instant::now();
    let mut fastup_patterns = fastup(&args, &data);
    println!("fastup trained in {:.2?}", start.elapsed());
    let fu_double_patterns = best_double_patterns(&data, &fastup_patterns, 5);
    start = Instant::now();
    

    println!("evaluation on testing data");
    print_all_zs(testing_data_path, &mut fastup_patterns, args.block_size);

    println!("double patterns:");
    print_all_zs_double(testing_data_path, &fu_double_patterns, args.block_size);

    println!("evaluated in {:.2?}", start.elapsed());
}

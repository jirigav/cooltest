mod bottomup;
mod common;
mod fastup;
mod patterns;

use crate::bottomup::bottomup;
use crate::common::*;
use crate::fastup::fastup;
use crate::patterns::*;
use clap::Parser;
use std::time::Instant;
use pyo3::prelude::*;


fn p_value(positive: usize, sample_size: usize, probability: f64) -> f64{
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let scipy = PyModule::import(py, "scipy").unwrap();
        let result: f64 = scipy.getattr("stats").unwrap().getattr("binom_test").unwrap().call1((positive, sample_size, probability, "two-sided")).unwrap()
            .extract().unwrap();
        result
    })
}

fn prepare_data(
    data_source: String,
    block_size: usize,
    halving: bool,
) -> (Vec<Vec<u8>>, Option<Vec<Vec<u8>>>) {
    let mut training_data = load_data(&data_source, block_size);
    let mut testing_data_option = None;

    if halving {
        let (tr_data, testing_data) = training_data.split_at(training_data.len() / 2);
        testing_data_option = Some(testing_data.to_vec());
        training_data = tr_data.to_vec();
    }
    (training_data, testing_data_option)
}

fn results(
    mut final_patterns: Vec<Pattern>,
    start: Instant,
    training_data: &[Vec<u8>],
    testing_data_option: Option<Vec<Vec<u8>>>,
    evaluated_disses: usize,
) {
    final_patterns.sort_by(|a, b| {
        f64::abs(b.z_score.unwrap())
            .partial_cmp(&f64::abs(a.z_score.unwrap()))
            .unwrap()
    });
    
    let abs_z_1 = f64::abs(final_patterns[0].z_score.unwrap());
    let mut b_double_pattern = best_double_pattern(training_data, &final_patterns);
    let abs_z_2 = f64::abs(b_double_pattern.z_score.unwrap());
    let mut abs_z_3: f64 = 0.0;
    let mut best_triple: DisjointPatterns = DisjointPatterns::new(&[]).unwrap();

    if let Some(best_tr) = best_disjoint_tripple(&final_patterns, training_data) {
        println!("triple found: {:?}", best_tr);
        abs_z_3 = f64::abs(best_tr.z_score.unwrap());
        best_triple = best_tr
    }

    println!("trained in {:.2?}", start.elapsed());

    println!(
        "total number of distinguishers evaluated: {}",
        evaluated_disses
    );

    if abs_z_1 > abs_z_2 {
        if abs_z_1 > abs_z_3 {
            println!("z-score: {}", final_patterns[0].z_score.unwrap());
            println!("best pattern: {:?}", final_patterns[0])
        } else {
            println!("z-score: {}", best_triple.z_score.unwrap());
            println!("best pattern: {:?}", best_triple)
        }
    } else if abs_z_2 > abs_z_3 {
        println!("z-score: {}", b_double_pattern.z_score.unwrap());
        println!("best double pattern: {:?}", b_double_pattern)
    } else {
        println!("z-score: {}", best_triple.z_score.unwrap());
        println!("best pattern: {:?}", best_triple)
    }

    if let Some(testing_data) = testing_data_option {
        if abs_z_1 > abs_z_2 {
            if abs_z_1 > abs_z_3 {
                println!("z-score: {}", evaluate_pattern(&mut final_patterns[0], &testing_data));
                println!("p-value: {:.0e}", p_value(final_patterns[0].count.unwrap(), testing_data.len(), 2.0_f64.powf(-(final_patterns[0].length as f64))));
            } else {
                println!("z-score: {}", evaluate_pattern(&mut best_triple, &testing_data));
                println!("p-value: {:.0e}", p_value(best_triple.get_count(), testing_data.len(), best_triple.probability));
            }
            
        } else if abs_z_2 > abs_z_3 {
            println!("z-score: {}", evaluate_pattern(&mut b_double_pattern, &testing_data));
            println!("p-value: {:.0e}", p_value(b_double_pattern.get_count(), testing_data.len(), b_double_pattern.probability));
        } else {
            println!("z-score: {}", evaluate_pattern(&mut best_triple, &testing_data));
            println!("p-value: {:.0e}", p_value(best_triple.get_count(), testing_data.len(), best_triple.probability));
        }
           
    }
 
}


fn run_bottomup(data_source: String, block_size: usize, k: usize, min_count: usize, halving: bool) {
    let (training_data, testing_data_option) = prepare_data(data_source, block_size, halving);

    let start = Instant::now();
    let (final_patterns, evaluated_disses) = bottomup(&training_data, block_size, k, min_count);
    results(
        final_patterns,
        start,
        &training_data,
        testing_data_option,
        evaluated_disses,
    );
}

fn run_fastup(
    data_source: String,
    block_size: usize,
    k: usize,
    n: usize,
    min_count: usize,
    halving: bool,
) {
    let (training_data, testing_data_option) = prepare_data(data_source, block_size, halving);

    let start = Instant::now();
    let (final_patterns, evaluated_disses) = fastup(&training_data, block_size, n, k, min_count);
    results(
        final_patterns,
        start,
        &training_data,
        testing_data_option,
        evaluated_disses,
    );
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

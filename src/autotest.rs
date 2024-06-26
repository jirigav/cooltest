use crate::bottomup::bottomup;
use crate::common::{prepare_data, Args};
use crate::results::results;
use std::time::Instant;

const GB: usize = 1000000000;
const MB: usize = 1000000;

fn choose_k(block_size: usize, data_size: usize) -> usize {
    if data_size <= 10 * MB && block_size < 128 {
        4
    } else if data_size < 2 * GB && block_size < 256 {
        3
    } else {
        2
    }
}

pub(crate) fn autotest(mut args: Args) {
    let (training_data, testing_data) = prepare_data(&args.data_source, args.block, true);
    let mut testing_data = testing_data.unwrap();
    let mut tested_cases = 0;
    let start = Instant::now();
    let data_size = training_data.len();

    let mut k = choose_k(args.block, data_size);

    tested_cases += 1;
    let mut hist = bottomup(
        &training_data,
        args.block,
        k,
        args.top,
        args.max_bits,
        args.threads,
    );
    let testing_data2;
    if args.block <= 256 {
        tested_cases += 1;
        let (training_data, testing_data_opt2) =
            prepare_data(&args.data_source, 2 * args.block, true);
        testing_data2 = testing_data_opt2.unwrap();
        k = choose_k(2 * args.block, data_size);
        let hist2 = bottomup(
            &training_data,
            args.block * 2,
            k,
            args.top,
            args.max_bits,
            args.threads,
        );
        if hist2.z_score.abs() > hist.z_score.abs() {
            hist = hist2;
            testing_data = testing_data2;
        }
    }
    println!("training finished in {:?}", start.elapsed());

    if tested_cases > 1 {
        let new_alpha = args.alpha / (tested_cases as f64);
        println!(
            "Adjusting significance level based on the number of tests from {} to {}",
            args.alpha, new_alpha
        );
        args.alpha = new_alpha;
    }

    results(hist, &testing_data, args)
}

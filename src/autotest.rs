use crate::bottomup::bottomup;
use crate::common::{load_data, p_value, prepare_data, Args};
use crate::results::results;
use std::time::Instant;

const GB: usize = 1000000000;
const MB: usize = 1000000;

fn configurations(block_size: usize, data_size: usize) -> Vec<(usize, usize)> {
    let mut configs = Vec::new();
    configs.push((8, 8));
    let mut bs = block_size;
    while bs <= 512 {
        let k = if data_size <= 10 * MB && bs < 128 {
            4
        } else if data_size < 2 * GB && bs < 256 {
            3
        } else {
            2
        };
        configs.push((bs, k));
        bs *= 2;
    }
    configs
}

pub(crate) fn autotest(mut args: Args) {
    let raw = load_data(&args.data_source);
    let start = Instant::now();

    let configs = configurations(args.block, raw.len());
    let num_tests = configs.len();
    let corrected_alpha = args.alpha / (num_tests as f64);

    if num_tests > 1 {
        println!(
            "Running {} configurations, significance level adjusted from {:.0e} to {:.0e}",
            num_tests, args.alpha, corrected_alpha
        );
    }

    let mut best_hist = None;
    let mut best_testing_data = None;
    let mut best_p_value = f64::MAX;

    for (block_size, k) in &configs {
        println!("\nTesting block size {}; k = {} ...", block_size, k);
        let (training_data, testing_data) = prepare_data(&raw, *block_size, true);
        let testing_data = testing_data.unwrap();
        let hist = bottomup(&training_data, *block_size, *k, args.threads);

        let (count, _) = hist.evaluate(&testing_data);
        let prob = 2.0_f64.powf(-(hist.bits.len() as f64));
        let p_val = p_value(
            testing_data.len(),
            count,
            prob * (hist.best_division as f64),
        );

        if p_val < best_p_value {
            best_p_value = p_val;
            best_hist = Some(hist);
            best_testing_data = Some(testing_data);
        }

        if p_val < corrected_alpha {
            println!(
                "Early stopping: p-value {:.0e} < corrected alpha {:.0e}",
                p_val, corrected_alpha
            );
            break;
        } else {
            println!("p-value: {:.0e} >= corrected alpha {:.0e} (not significant)", p_val, corrected_alpha);
        }
    }

    println!("Finished in {:?}", start.elapsed());

    args.alpha = corrected_alpha;
    results(best_hist.unwrap(), &best_testing_data.unwrap(), args)
}

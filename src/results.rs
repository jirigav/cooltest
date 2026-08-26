use crate::{
    bottomup::Histogram,
    common::{p_value, z_score, Args, Res},
};
use serde_json::json;
use std::fs::File;
use std::io::Write;

/// Outcome of testing one distinguisher against held-out data.
pub(crate) struct Evaluation {
    /// Blocks falling in the bins the distinguisher selects.
    pub(crate) count: usize,
    /// Full histogram over all `2^k` bin values.
    pub(crate) bins: Vec<usize>,
    pub(crate) z_score: f64,
    pub(crate) p_value: f64,
}

/// Evaluates `hist` on `testing_data` under the null hypothesis of uniform blocks.
///
/// The distinguisher selects `best_division` of the `2^k` equiprobable bins, so a random block
/// lands in the selected set with probability `best_division / 2^k`, and the number of blocks that
/// do is exactly binomial under the null.
pub(crate) fn evaluate_histogram(hist: &Histogram, testing_data: &[u8]) -> Evaluation {
    let (count, bins) = hist.evaluate(testing_data);
    let samples = testing_data.len() / (hist.block_size / 8);
    let prob = 2.0_f64.powf(-(hist.bits.len() as f64)) * (hist.best_division as f64);

    Evaluation {
        count,
        bins,
        z_score: z_score(samples, count, prob),
        p_value: p_value(samples, count, prob),
    }
}

/// Reports the outcome and returns whether the randomness hypothesis was rejected.
pub(crate) fn results(hist: &Histogram, testing_data: &[u8], args: &Args) -> Res<bool> {
    let eval = evaluate_histogram(hist, testing_data);
    print_results(&eval, args.alpha, hist);

    if let Some(path) = &args.json {
        let output = json!({
            "args": args,
            "dis": hist,
            "result": if eval.p_value >= args.alpha {"random"} else {"non-random"},
            "p-value": eval.p_value,
            "z-score": eval.z_score,
            "blocks": testing_data.len() / (hist.block_size / 8),
            "positive": eval.count,
            "histogram": eval.bins,
        });

        let mut file = File::create(path).map_err(|e| format!("failed to create '{path}': {e}"))?;
        file.write_all(serde_json::to_string_pretty(&output)?.as_bytes())
            .map_err(|e| format!("failed to write '{path}': {e}"))?;
    }

    Ok(eval.p_value < args.alpha)
}

fn print_results(eval: &Evaluation, alpha: f64, hist: &Histogram) {
    let bins = &eval.bins;
    println!("{}", "-".repeat(80));
    println!("RESULTS:\n");

    println!("Histogram(the discovered Boolean function returns 1 for values before the separator and 0 for values after the separator.):\n");
    // Calculate prefix width: "x{N} " per bit + "| [" + 1 digit per bit + "] | "
    let bits_prefix = hist
        .bits
        .iter()
        .map(|b| format!("x{} ", b))
        .collect::<String>();
    let prefix_width: usize = bits_prefix.len() + hist.bits.len() + 7;
    let bar_width = 80_usize.saturating_sub(prefix_width);
    // Guards the bar scaling: an all-zero histogram would otherwise divide by zero.
    let max = (*bins.iter().max().unwrap_or(&0)).max(1);
    let avg: f64 = bins.iter().sum::<usize>() as f64 / bins.len() as f64;
    let avg_len = (avg as usize) * bar_width / max;

    for (i, ind) in hist.sorted_indices.iter().enumerate() {
        print!("{bits_prefix}| [");

        let mut j = *ind;
        for _ in 0..hist.bits.len() {
            print!("{}", j % 2);
            j /= 2;
        }
        print!("] | ");

        let bar_len = bins[*ind] * bar_width / max;
        let above_avg = bins[*ind] as f64 > avg;
        let split = if above_avg {
            avg_len.saturating_sub(1).min(bar_len)
        } else {
            bar_len
        };
        let base: String = "∎".repeat(split);
        let red: String = "∎".repeat(bar_len - split);
        if above_avg && i < hist.best_division {
            println!("{base}\x1b[31m{red}\x1b[0m");
        } else {
            println!("{base}{red}");
        }

        if i + 1 == hist.best_division {
            println!("{}", "—".repeat(80));
        }
    }

    println!("\nZ-score: {}", eval.z_score);
    println!("P-value: {:.0e}", eval.p_value);
    if eval.p_value >= alpha {
        println!(
            "As the p-value >= alpha {alpha:.0e}, the randomness hypothesis cannot be rejected."
        );
        println!("= CoolTest could not find statistically significant non-randomness.");
    } else {
        println!("As the p-value < alpha {alpha:.0e}, the randomness hypothesis is REJECTED.");
        println!("= Data is not random.");
    }
}

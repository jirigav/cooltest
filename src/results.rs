use crate::{
    bottomup::Histogram,
    common::{p_value, z_score, Args},
};
use serde_json::json;
use std::fs::File;
use std::io::Write;

pub(crate) fn results(hist: Histogram, testing_data: &[Vec<u8>], args: Args) {
    let (count, bins) = hist.evaluate(testing_data);
    let prob = 2.0_f64.powf(-(hist.bits.len() as f64));
    let z = z_score(
        testing_data.len(),
        count,
        prob * (hist.best_division as f64),
    );
    let p_val = p_value(
        testing_data.len(),
        count,
        prob * (hist.best_division as f64),
    );
    print_results(p_val, z, args.alpha, &hist, bins);

    if let Some(path) = args.json.clone() {
        let mut file =
            File::create(&path).unwrap_or_else(|_| panic!("File {} couldn't be created", path));

        let output = json!({
            "args": args,
            "dis": hist,
            "result": if p_val >= args.alpha {"random"} else {"non-random"},
            "p-value": p_val
        });

        file.write_all(
            serde_json::to_string_pretty(&output)
                .expect("Failed to produce json!")
                .as_bytes(),
        )
        .unwrap();
    }
}

fn print_results(p_value: f64, z_score: f64, alpha: f64, hist: &Histogram, bins: Vec<usize>) {
    println!("----------------------------------------------------------------------");
    println!("RESULTS:\n");

    println!("Histogram(the discovered Boolean function returns 1 for values before the separator and 0 for values after the separator.):\n");
    // Calculate prefix width: "x{N} " per bit + "| [" + 1 digit per bit + "] | "
    let prefix_width: usize = hist.bits.iter().map(|b| format!("x{} ", b).len()).sum::<usize>()
        + 3 + hist.bits.len() + 4;
    let bar_width = 80_usize.saturating_sub(prefix_width);
    let max = *bins.iter().max().unwrap();
    let avg: f64 = bins.iter().sum::<usize>() as f64 / bins.len() as f64;
    let avg_len = (avg as usize) * bar_width / max;
    
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

        let bar_len = bins[*ind] * bar_width / max;
        let above_avg = bins[*ind] as f64 > avg;
        let split = if above_avg { avg_len.min(bar_len.saturating_sub(1)) } else { bar_len };
        let base: String = "∎".repeat(split);
        let red: String = "∎".repeat(bar_len - split);
        if above_avg && i < hist.best_division {
            print!("{base}\x1b[31m{red}\x1b[0m");
        } else {
            print!("{base}{red}");
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

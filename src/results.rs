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
            "result": if p_val < args.alpha {"random"} else {"non-random"},
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

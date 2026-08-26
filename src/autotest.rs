use crate::bottomup::{bottomup, Histogram};
use crate::common::{load_data, status, warn_if_underpowered, whole_blocks, Args, Res};
use crate::results::{evaluate_histogram, results};
use std::time::Instant;

const GB: usize = 1000000000;
const MB: usize = 1000000;

/// Block sizes to sweep, paired with the histogram width to use for each.
///
/// `-b` acts as a floor rather than an exact choice: the ladder always reaches 512, and extends
/// past it when the user asked for a larger block.
fn configurations(block_size: usize, data_size: usize) -> Vec<(usize, usize)> {
    // A byte-value histogram. k = 8 over an 8-bit block already covers every bit of the block.
    let mut configs = vec![(8, 8)];

    let top = block_size.max(512);
    let mut bs = block_size.max(8);
    loop {
        let k = if data_size <= 10 * MB && bs < 128 {
            4
        } else if data_size < 2 * GB && bs < 256 {
            3
        } else {
            2
        };
        // Skip a block size already covered. A duplicate would consume a share of the Bonferroni
        // budget for a test that is no stronger than one already planned.
        if !configs.iter().any(|(b, _)| *b == bs) {
            configs.push((bs, k));
        }
        match bs.checked_mul(2) {
            Some(next) if next <= top => bs = next,
            _ => break,
        }
    }
    configs
}

pub(crate) fn autotest(mut args: Args) -> Res<bool> {
    let raw = load_data(&args.data_source)?;
    let start = Instant::now();

    let configs = configurations(args.block, raw.len());
    let num_tests = configs.len();
    // Bonferroni over every configuration planned, not just those actually run. Stopping early on
    // the first rejection only makes the correction more conservative, never less.
    let corrected_alpha = args.alpha / (num_tests as f64);

    status!("Autotest chooses k per configuration; -k is ignored.");
    if num_tests > 1 {
        status!(
            "Running {} configurations, significance level adjusted from {:.0e} to {:.0e}",
            num_tests,
            args.alpha,
            corrected_alpha
        );
    }

    let mut best: Option<(Histogram, &[u8])> = None;
    let mut best_p_value = f64::INFINITY;
    let mut tested = 0;

    for (block_size, k) in &configs {
        status!("\nTesting block size {block_size}; k = {k} ...");

        let (training_data, testing_data) = match whole_blocks(&raw, *block_size, true) {
            Ok(split) => split,
            Err(e) => {
                status!("Skipping block size {block_size}: {e}");
                continue;
            }
        };
        warn_if_underpowered(testing_data.len() / (*block_size / 8), corrected_alpha);

        let hist = bottomup(training_data, *block_size, *k, args.threads)?;
        let p_val = evaluate_histogram(&hist, testing_data).p_value;
        tested += 1;

        if p_val < best_p_value {
            best_p_value = p_val;
            best = Some((hist, testing_data));
        }

        if p_val < corrected_alpha {
            status!(
                "Early stopping: p-value {:.0e} < corrected alpha {:.0e}",
                p_val,
                corrected_alpha
            );
            break;
        }
        status!(
            "p-value: {:.0e} >= corrected alpha {:.0e} (not significant)",
            p_val,
            corrected_alpha
        );
    }

    status!(
        "Finished in {:?} ({tested} configurations tested)",
        start.elapsed()
    );

    let (hist, testing_data) = best.ok_or_else(|| {
        format!(
            "no configuration could be tested: {} bytes is too small for every block size tried \
             (smallest needs {} bytes)",
            raw.len(),
            2 * configs.iter().map(|(b, _)| b / 8).min().unwrap_or(1)
        )
    })?;

    args.alpha = corrected_alpha;
    results(&hist, testing_data, &args)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block_sizes(configs: &[(usize, usize)]) -> Vec<usize> {
        configs.iter().map(|(b, _)| *b).collect()
    }

    #[test]
    fn test_default_configurations_unchanged() {
        // The default block size must keep producing exactly the ladder it always has, so that
        // autotest verdicts stay comparable with previously published runs.
        assert_eq!(
            configurations(128, 100 * MB),
            vec![(8, 8), (128, 3), (256, 2), (512, 2)]
        );
    }

    #[test]
    fn test_large_block_size_still_sweeps() {
        // Previously the `while bs <= 512` loop never ran for -b > 512, silently reducing autotest
        // to the single block-size-8 configuration.
        assert_eq!(block_sizes(&configurations(1024, 100 * MB)), vec![8, 1024]);
        assert_eq!(block_sizes(&configurations(4096, 100 * MB)), vec![8, 4096]);
    }

    #[test]
    fn test_block_size_eight_is_not_duplicated() {
        let configs = configurations(8, MB);
        assert_eq!(configs[0], (8, 8));
        assert_eq!(
            configs.iter().filter(|(b, _)| *b == 8).count(),
            1,
            "a duplicated block size wastes part of the Bonferroni budget"
        );
        assert_eq!(block_sizes(&configs), vec![8, 16, 32, 64, 128, 256, 512]);
    }

    #[test]
    fn test_k_shrinks_with_block_size_and_data_size() {
        // Small input: wider histograms are affordable for small blocks.
        let small = configurations(16, MB);
        assert_eq!(small.iter().find(|(b, _)| *b == 16).unwrap().1, 4);
        // Large input: k drops to keep the search tractable.
        let large = configurations(16, 3 * GB);
        assert_eq!(large.iter().find(|(b, _)| *b == 16).unwrap().1, 2);
    }

    #[test]
    fn test_every_block_size_is_byte_aligned() {
        for start in [8, 16, 24, 128, 1024] {
            for (bs, k) in configurations(start, 100 * MB) {
                assert_eq!(bs % 8, 0, "block size {bs} is not a whole number of bytes");
                assert!(
                    (1..=bs).contains(&k),
                    "k = {k} is invalid for block size {bs}"
                );
            }
        }
    }
}

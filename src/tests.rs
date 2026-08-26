//! End-to-end tests: search, evaluation, and serialisation together, on data whose answer is known
//! in advance.
//!
//! These are statistical assertions, not exact ones. They are calibrated so that a correct
//! implementation fails with probability well under 1e-3; a failure here means the tool's verdicts
//! have moved, not that a threshold needs nudging.

use crate::bottomup::{bottomup, Histogram};
use crate::common::{splitmix64_bytes, whole_blocks};
use crate::results::evaluate_histogram;

const ALPHA: f64 = 0.0001;
const DATA_LEN: usize = 1 << 20;

fn search_and_evaluate(
    raw: &[u8],
    block_size: usize,
    k: usize,
    threads: usize,
) -> (Histogram, f64) {
    let (training, testing) = whole_blocks(raw, block_size, true).unwrap();
    let hist = bottomup(training, block_size, k, threads).unwrap();
    let p_value = evaluate_histogram(&hist, testing).p_value;
    (hist, p_value)
}

/// Calibration. Data from a generator that passes the standard batteries must not be rejected.
/// A failure here is a false positive: non-randomness reported where there is none.
#[test]
fn test_random_data_is_not_rejected() {
    for seed in [1_u64, 2, 3, 12345, 0xDEAD_BEEF] {
        let raw = splitmix64_bytes(DATA_LEN, seed);
        let (_, p_value) = search_and_evaluate(&raw, 128, 2, 0);
        assert!(
            p_value >= ALPHA,
            "false positive on splitmix64 seed {seed}: p = {p_value:e}"
        );
    }
}

/// Power. An injected dependency between two bits must be found, and rejected decisively.
#[test]
fn test_biased_data_is_rejected() {
    let block_size = 128;
    let mut raw = splitmix64_bytes(DATA_LEN, 99);
    // Force bit 3 == bit 7 in every block. Both live in byte 0, at offsets 4 and 0.
    for block in raw.chunks_exact_mut(block_size / 8) {
        block[0] = (block[0] & !0x10) | ((block[0] & 0x01) << 4);
    }

    let (hist, p_value) = search_and_evaluate(&raw, block_size, 2, 0);
    assert!(p_value < ALPHA, "missed the injected bias: p = {p_value:e}");
    assert_eq!(hist.bits, vec![3, 7], "recovered the wrong bit pair");
    // The two bits agree in every block, so the selected half of the histogram holds everything.
    assert_eq!(hist.best_division, 2);
}

/// A bias that only shows up at one block size must not be masked by testing at another. Here the
/// dependency spans a 64-bit period, so a 128-bit block sees it and an 8-bit block cannot.
#[test]
fn test_bias_found_at_the_matching_block_size() {
    let mut raw = splitmix64_bytes(DATA_LEN, 4242);
    for block in raw.chunks_exact_mut(8) {
        block[0] = (block[0] & !0x10) | ((block[0] & 0x01) << 4);
    }

    let (_, p_value) = search_and_evaluate(&raw, 64, 2, 0);
    assert!(
        p_value < ALPHA,
        "missed the bias at block size 64: {p_value:e}"
    );
}

/// The two search algorithms must agree on the distinguisher and therefore on the verdict.
#[test]
fn test_search_paths_agree_end_to_end() {
    let raw = splitmix64_bytes(DATA_LEN, 7);

    let (sequential, p_sequential) = search_and_evaluate(&raw, 64, 2, 0);
    let (parallel, p_parallel) = search_and_evaluate(&raw, 64, 2, 4);

    assert_eq!(sequential.bits, parallel.bits);
    assert_eq!(sequential.best_division, parallel.best_division);
    assert_eq!(sequential.sorted_indices, parallel.sorted_indices);
    assert_eq!(p_sequential, p_parallel);
}

/// A distinguisher has to survive the round trip that `--json` and `evaluate` perform. `z_score` is
/// deliberately not serialised, so the remaining fields must be enough to evaluate on fresh data.
#[test]
fn test_json_round_trip() {
    let raw = splitmix64_bytes(DATA_LEN, 55);
    let (hist, _) = search_and_evaluate(&raw, 64, 2, 0);

    let encoded = serde_json::to_string(&hist).unwrap();
    let loaded: Histogram = serde_json::from_str(&encoded).unwrap();
    loaded.validate().unwrap();

    assert_eq!(loaded.bits, hist.bits);
    assert_eq!(loaded.best_division, hist.best_division);
    assert_eq!(loaded.sorted_indices, hist.sorted_indices);
    assert_eq!(loaded.block_size, hist.block_size);

    let fresh = splitmix64_bytes(DATA_LEN, 56);
    let (data, _) = whole_blocks(&fresh, loaded.block_size, false).unwrap();
    assert_eq!(
        evaluate_histogram(&loaded, data).p_value,
        evaluate_histogram(&hist, data).p_value
    );
}

/// Inputs that used to panic must now produce errors.
#[test]
fn test_degenerate_inputs_are_errors_not_panics() {
    let raw = splitmix64_bytes(1024, 3);

    // Fewer than two blocks: nothing to split into training and testing halves.
    assert!(whole_blocks(&raw[..16], 256, true).is_err());
    assert!(whole_blocks(&[], 8, true).is_err());

    // More selected bits than the block has.
    assert!(bottomup(&raw, 8, 9, 0).is_err());
    assert!(bottomup(&raw, 8, 9, 2).is_err());

    // A distinguisher wider than the data it is evaluated on is still well-formed.
    let (hist, _) = search_and_evaluate(&raw, 64, 2, 0);
    hist.validate().unwrap();
}

/// Every distinguisher the search can return must be evaluable, including on data that produces a
/// completely flat histogram.
#[test]
fn test_constant_data_produces_a_usable_distinguisher() {
    let raw = vec![0_u8; 4096];
    let (hist, p_value) = search_and_evaluate(&raw, 64, 2, 0);

    hist.validate().unwrap();
    assert!(hist.best_division >= 1);
    // All-zero blocks are maximally non-random: every block lands in the same bin.
    assert!(
        p_value < ALPHA,
        "constant data was not rejected: {p_value:e}"
    );
}

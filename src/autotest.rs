use crate::bottomup::bottomup;
use crate::common::{prepare_data, Args};
use crate::results::results;
use std::time::Instant;

// TODO
fn choose_k(_block_size: usize, _data_size: usize) -> usize {
    3
}

pub(crate) fn autotest(args: Args) {
    let (training_data, testing_data) = prepare_data(&args.data_source, args.block, true);
    let mut testing_data = testing_data.unwrap();
    let start = Instant::now();
    let data_size = training_data.len();

    let mut k = choose_k(args.block, data_size);

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
        let (training_data, testing_data_opt2) = prepare_data(&args.data_source, 2*args.block, true);
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

    results(hist, &testing_data, args)
}

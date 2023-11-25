# CoolTest

CoolTest is a randomness-testing tool. It uses first half of provided data to construct a distinguisher in a three phase process. Then it evaluates the distinguisher on the second half of the data to evaluate the probability of the data being random.


# How to use CoolTest

## Setup
1. You need to have [Rust](https://www.rust-lang.org/tools/install) and Python library [SciPy](https://scipy.org/install/) installed. 


2. Run `cargo build --release`

## Run the tool

You can use `./target/release/cooltest --help` to see all options.

To run the tool with default parameters you can use:
`./target/release/cooltest <file>`



# License
CoolTest is released under the MIT license. See See [LICENSE](LICENSE) for more information.
# CoolTest examples

Runnable examples of every CoolTest workflow, against sample data with known
properties, so you can see what a passing run and a failing run actually look
like before pointing the tool at your own generator.

## Quick start

Build the binary from the repository root, then change the current directory 
to examples and generate example data:

```
cargo build --release
cd examples
python generate_data.py
```

## The examples

| # | Example | Shows |
| --- | --- | --- |
| 1 | [Test a file with the default settings](01-default-run.md) | What a pass and a rejection look like, and how to read the histogram |
| 2 | [Let autotest pick the configuration](02-autotest.md) | Sweeping block sizes with a corrected alpha when you do not know the block structure |
| 3 | [Choose the block size and histogram width by hand](03-block-size-and-k.md) | `-b` and `-k`, and how much a mismatched block size costs |
| 4 | [Save the result as JSON and re-use the distinguisher](04-json-and-evaluate.md) | `-j` and the `evaluate` subcommand, for a verdict on data the search never saw |
| 5 | [Use the exit code in a script](05-exit-code.md) | `--exit-code` and `-q` for automated checks |
| 6 | [Larger blocks, multiple threads](06-threads.md) | `-t`, and how the search cost scales with block size and `k` |

## Files in this folder

| File | Purpose |
| --- | --- |
| `generate_data.py` | Writes the sample data files into `data/` |
| `data/` | Generated sample data |

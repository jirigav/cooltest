# Example 4: Save the result as JSON and re-use the distinguisher


`-j` writes the full result — arguments, distinguisher, histogram, p-value — to
a JSON file. The distinguisher it contains can then be applied to fresh data
with the `evaluate` subcommand, without searching again.

This matters for a genuinely independent verdict. A normal run trains on the
first half of the file and tests on the second, but if you tune `-b` and `-k`
against the same file until something rejects, the reported p-value no longer
accounts for all the configurations you tried. Freezing the distinguisher and
evaluating it on data it has never seen gives a p-value you can quote.

```
$ head -c 4000000 data/biased.bin > out/train.bin
$ tail -c 4000000 data/biased.bin > out/holdout.bin
$ cooltest -q -j out/result.json out/train.bin
```

`evaluate` expects a bare distinguisher, which lives under the `dis` key of the
JSON output:

```
$ python3 -c "import json; json.dump(json.load(open('out/result.json'))['dis'], open('out/dis.json','w'), indent=2)"
```

or, with jq:

```
$ jq .dis out/result.json > out/dis.json
```

```json
{
  "best_division": 2,
  "bits": [0, 3],
  "block_size": 128,
  "sorted_indices": [3, 2, 1, 0]
}
```

The distinguisher carries its own block size and bit count, so `-b` and `-k` are
taken from the file and ignored on the command line:

```
$ cooltest out/holdout.bin evaluate -d out/dis.json
```

```
The whole input file is used for evaluation. The p-value is only valid if this file is disjoint from the data the distinguisher was trained on.

Z-score: 49.668
P-value: 0e0
As the p-value < alpha 1e-4, the randomness hypothesis is REJECTED.
= Data is not random.
```

There is no training split here — the whole input file is evaluated — so it is
on you to make sure the file is disjoint from the training data. Above, the two
halves were cut apart before the search ever ran.

The same mechanism lets you carry a distinguisher between files: search once on
a sample from a generator, then re-run `evaluate` against every later batch it
produces, which is much cheaper than a fresh search each time.

---

[All examples](README.md) · Next: [Use the exit code in a script](05-exit-code.md)

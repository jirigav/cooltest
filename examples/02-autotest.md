# Example 2: Let autotest pick the configuration


If you do not know the block structure of the data, `autotest` sweeps a ladder
of block sizes (8, then 128, 256, 512 by default), picking a histogram width for
each and applying a Bonferroni correction to alpha to pay for the multiple
tests. It stops at the first configuration that rejects. Setting the block size 
as the output size of the function producing tested data can improve the results of `autotest`.

```
$ cooltest data/lcg.bin autotest
```

```
Autotest chooses k per configuration; -k is ignored.
Running 4 configurations, significance level adjusted from 1e-4 to 3e-5

Testing block size 8; k = 8 ...
p-value: 8e-1 >= corrected alpha 3e-5 (not significant)

Testing block size 128; k = 3 ...
Distinguisher: Histogram { bits: [6, 7, 38], sorted indices: [0, 1, 2, 3, 4, 5, 6, 7], best_division: 1, z_score: 1322.8756555322952 }
Early stopping: p-value 0e0 < corrected alpha 3e-5
Finished in 2.805748797s (2 configurations tested)

x6 x7 x38 | [000] | ∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎
————————————————————————————————————————————————————————————————————————————————
x6 x7 x38 | [100] |
x6 x7 x38 | [010] |
x6 x7 x38 | [110] |
x6 x7 x38 | [001] |
x6 x7 x38 | [101] |
x6 x7 x38 | [011] |
x6 x7 x38 | [111] |

Z-score: 1322.8756555322952
P-value: 0e0
```

Every single block falls into bin `[000]` and the other seven bins are empty:
bits 6, 7 and 38 are always zero. That is the well-known weakness of an LCG with
a power-of-two modulus, whose low-order bits cycle with very short periods.

Note that block size 8 did **not** reject. A byte-histogram cannot see a pattern
that only appears once bytes are grouped into 4-byte outputs, which is exactly
why the ladder is worth sweeping.

`-b` still has an effect under `autotest`: it acts as a floor on the ladder
rather than an exact choice. `-k` is ignored, since autotest picks a histogram
width per configuration from the block size and the size of the input.

Because the sweep corrects alpha for every configuration it plans, its verdict
already accounts for the multiple tests. That is what makes it the safe default
when you would otherwise be tempted to try settings by hand until something
rejects.

---

[All examples](README.md) · Next: [Choose the block size and histogram width by hand](03-block-size-and-k.md)

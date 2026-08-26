# Example 6: Larger blocks, multiple threads

Larger blocks and wider histograms make the search substantially more expensive,
since it enumerates every subset of `k` bit positions out of the block.

`-t 0`, the default, is an optimised single-threaded algorithm.
`-t N` with N ≥ 2 spreads the candidate search over N threads, which is worth it
once the block size or `k` grows:

```
$ cooltest -b 256 -k 2 -t 4 -q data/good.bin
```

```
Z-score: 0.6401333194473373
P-value: 3e-1
As the p-value >= alpha 1e-4, the randomness hypothesis cannot be rejected.
= CoolTest could not find statistically significant non-randomness.
```

Be aware of how quickly this scales. On 8 MB with 4 threads, `-b 256 -k 2`
finishes in about 0.2 s; `-b 256 -k 3` has to search 2.7 million bit triples
instead of 32 640 pairs and takes about 24 s. Another step to `-k 4` would be
roughly another two orders of magnitude.

`-t 1` is not the same as `-t 0`: it runs the multi-threaded code path with a
single thread, which is slower than the dedicated single-thread algorithm. Use
`-t 0` for small inputs and go straight to `-t N` with N ≥ 8 when you need the
parallelism.

---

[All examples](README.md) · Next: [Testing your own data](testing-your-own-data.md)

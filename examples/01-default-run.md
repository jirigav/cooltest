# Example 1: Test a file with the default settings

The default configuration uses 128-bit blocks, 2-bit histograms and
alpha = 0.0001. It is a reasonable thing to try if you don't have 
additional information about your data.

```
$ cooltest data/good.bin
```

```
Z-score: 0.628
P-value: 3e-1
As the p-value >= alpha 1e-4, the randomness hypothesis cannot be rejected.
= CoolTest could not find statistically significant non-randomness.
```

The best distinguisher CoolTest could build on the first half of the file
(bits 64 and 104, z-score 4.5) did not hold up on the second half. That is the
expected outcome for good data: searching a large space of candidate
distinguishers always turns up something that looks promising on the training
half, and the held-out half is what separates a real signal from that noise.

The same command on the biased file:

```
$ cooltest data/biased.bin
```

```
Distinguisher: Histogram { bits: [0, 3], sorted indices: [2, 3, 1, 0], best_division: 2, z_score: 50.588 }

x0 x3 | [01] | ∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎
x0 x3 | [11] | ∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎
————————————————————————————————————————————————————————————————————————————————
x0 x3 | [10] | ∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎
x0 x3 | [00] | ∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎

Z-score: 49.668
P-value: 0e0
As the p-value < alpha 1e-4, the randomness hypothesis is REJECTED.
= Data is not random.
```

CoolTest recovered exactly the bit that was tampered with. The bit values in
each row are printed in the order the bits are listed, so `[01]` means x0 = 0
and x3 = 1: the two rows above the separator are the ones with x3 = 1, and they
are visibly taller than the two with x3 = 0. P-value 0 (less than 10^-300 cannot 
be represented with 64-bit integers) indicates very strong pattern in the data. 

---

[All examples](README.md) · Next: [Let autotest pick the configuration](02-autotest.md)

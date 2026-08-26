# Example 3: Choose the block size and histogram width by hand


`-b` sets the block size in bits and should match the structure you suspect. The
counter data repeats every 16 bytes, so 128-bit blocks line up with it exactly:

```
$ cooltest -b 128 -k 1 data/counter.bin
```

```
x0 | [0] | ∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎
————————————————————————————————————————————————————————————————————————————————
x0 | [1] |

Z-score: 500
P-value: 0e0
```

With `-k 1` the histogram has only two bins, and a single bit is enough here:
the counter never reaches a value with its top bit set, so x0 is always 0.

`-k` widens the histogram, letting the search consider combinations of more bits
at once. It costs time — the search runs over all `block choose k` bit subsets —
so raise it only when a narrow histogram finds nothing:

```
$ cooltest -b 64 -k 4 data/biased.bin
```

```
Z-score: 34.61711957976862
P-value: 6e-263
As the p-value < alpha 1e-4, the randomness hypothesis is REJECTED.
= Data is not random.
```

Note the weaker z-score than in [Example 1](01-default-run.md) (34.6 against
49.7). The bias was injected on a 128-bit grid, so with 64-bit blocks only every
second block carries it. Matching `-b` to the real structure of the data matters 
more than any other setting.

Block sizes must be a whole number of bytes; `-b` rejects anything not divisible
by 8. Above 600 bits CoolTest warns that the search will be slow.

---

[All examples](README.md) · Next: [Save the result as JSON and re-use the distinguisher](04-json-and-evaluate.md)

# Example 5: Use the exit code in a script

`--exit-code` turns the verdict into a process status, and `-q` suppresses
progress bars and status messages so that stdout carries only the results.

| Exit status | Meaning |
| --- | --- |
| 0 | No statistically significant non-randomness found |
| 1 | Randomness hypothesis rejected — the data is not random |
| 2 | CoolTest failed to run (unreadable file, bad arguments, input too small) |

```
$ cooltest -q --exit-code data/good.bin
  exit 0 -> good.bin: no significant non-randomness found
$ cooltest -q --exit-code data/biased.bin
  exit 1 -> biased.bin: REJECTED, data is not random
```

Distinguish status 1 from status 2 rather than testing for "non-zero" — a typo
in a path would otherwise read as a failed randomness test:

```bash
if cooltest -q --exit-code "$file"; then
    echo "PASS: $file"
elif [ $? -eq 1 ]; then
    echo "FAIL: $file is not random"
    exit 1
else
    echo "ERROR: cooltest could not run"
    exit 2
fi
```

Without `--exit-code` CoolTest exits 0 whatever the verdict, so a script that
only checks the status will silently pass everything. Status 2 is still returned
for genuine errors either way.

Status and progress messages go to stderr and results to stdout, so the results
can be captured on their own even without `-q`:

```
$ cooltest data/good.bin 2>/dev/null > verdict.txt
```

---

[All examples](README.md) · Next: [Larger blocks, multiple threads](06-threads.md)

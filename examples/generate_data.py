#!/usr/bin/env python3
"""Generates the sample data files used by the CoolTest examples.

Every generator is seeded, so re-running this script reproduces byte-identical data.

Usage:
    python3 generate_data.py [--size-mb 8] [--out-dir data]
"""

import argparse
import os
import random


def good(size, seed=1):
    """Mersenne Twister output. Passes the standard batteries, and CoolTest."""
    return random.Random(seed).randbytes(size)


def biased(size, block_bytes=16, bit=3, extra=0.1, seed=2):
    """Random blocks in which one bit position is slightly skewed towards 1.

    Every other bit is fair, so no single-bit frequency count over the whole
    file looks unusual -- the bias only shows up once blocks are aligned.
    With `extra` = 0.1 the biased bit is 1 with probability 0.1 + 0.9 * 0.5 = 0.55.
    """
    rng = random.Random(seed)
    data = bytearray(rng.randbytes(size))
    byte_index, mask = bit // 8, 1 << (7 - bit % 8)
    for start in range(0, size - block_bytes + 1, block_bytes):
        if rng.random() < extra:
            data[start + byte_index] |= mask
    return bytes(data)


def lcg(size, seed=3):
    """Low 32 bits of a power-of-two-modulus linear congruential generator.

    The classic glibc-style constants. The low bits of such a generator have
    very short periods -- bit 0 simply alternates -- which CoolTest picks up
    immediately at any block size.
    """
    state = seed
    out = bytearray()
    while len(out) < size:
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        out += state.to_bytes(4, "little")
    return bytes(out[:size])


def counter(size, block_bytes=16):
    """A plain incrementing counter: not random at all, by construction.

    The high bytes of each block barely ever change, so the distinguisher found
    here is about as strong as a distinguisher gets.
    """
    out = bytearray()
    i = 0
    while len(out) < size:
        out += i.to_bytes(block_bytes, "big")
        i += 1
    return bytes(out[:size])


GENERATORS = {
    "good.bin": (good, "Mersenne Twister -- expected to pass"),
    "biased.bin": (biased, "one bit per block skewed to p=0.55 -- expected to fail"),
    "lcg.bin": (lcg, "weak LCG, low bits -- expected to fail"),
    "counter.bin": (counter, "incrementing counter -- expected to fail badly"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size-mb", type=float, default=8.0,
                        help="size of each generated file in MB (default: 8)")
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "data"),
                        help="directory to write the files into (default: ./data)")
    args = parser.parse_args()

    size = int(args.size_mb * 1000 * 1000)
    os.makedirs(args.out_dir, exist_ok=True)

    for name, (generate, description) in GENERATORS.items():
        path = os.path.join(args.out_dir, name)
        with open(path, "wb") as f:
            f.write(generate(size))
        print(f"{path:<24} {size:>10} bytes  ({description})")


if __name__ == "__main__":
    main()

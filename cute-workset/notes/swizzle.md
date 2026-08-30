For **BF16 `8 × 128`**, the picture changes because a row is now **256 bytes**, not 128 bytes.

```text
128 BF16 × 2 B = 256 B
256 B / 16 B = 16 chunks per row
```

So the natural byte-address decomposition is:

```text
byte address:

[ row bits ][ chunk bits ][ 4 low bits ]

row = 0..7        -> 3 bits
chunk = 0..15     -> 4 bits
inside 16B        -> 4 bits
```

More precisely, for row-major `8×128` BF16:

```text
bits:  10 9 8 | 7 6 5 4 | 3 2 1 0
       r2 r1 r0  c3 c2 c1 c0   xxxx
```

Now this is where an important detail about `Swizzle<B,M,S>` matters.

For:

```cpp
Swizzle<3,4,3>
```

CuTe actually takes:

```text
Z = bits [6:4]
Y = bits [9:7]

Z ^= Y
```

because `M=4`, `B=3`, `S=3`. ([GitHub][1])

But for an `8×128` BF16 row:

```text
bits [9:7] = r1 r0 c3
```

—not the full 3-bit row index.

So:

```text
Swizzle<3,4,3>
```

no longer means the clean:

```text
chunk_low3 ^= row
```

that it meant for `8×64`.

For `8×128`, the cleaner mapping would be:

```cpp
Swizzle<3,4,4>
```

because then:

```text
Z = bits [6:4]   = c2 c1 c0
Y = bits [10:8]  = r2 r1 r0

c[2:0] ^= row[2:0]
```

while `c3` remains unchanged.

So visualize:

```text
original:

       row           chunk             16B
    r2 r1 r0     c3 c2 c1 c0       xxxx
    └──────┘        │ └──────┘
       Y            │    Z
                    │
                 unchanged

Swizzle<3,4,4>:

    r2 r1 r0     c3 (c2c1c0 XOR r2r1r0) xxxx
```

### Why that still fixes the bank pattern

Remember that shared-memory banks repeat every:

```text
32 banks × 4 B = 128 B
```

But your row is:

```text
256 B = 2 × 128 B
```

So row starts still have identical bank phase:

```text
row 0 =    0 B → bank 0
row 1 =  256 B → bank 0
row 2 =  512 B → bank 0
...
```

Suppose all 8 rows access logical 16-byte chunk `0`.

Without swizzle:

```text
row 0 chunk 0 → banks 0-3
row 1 chunk 0 → banks 0-3
row 2 chunk 0 → banks 0-3
...
row 7 chunk 0 → banks 0-3
```

Potential 8-way collision.

With:

```cpp
Swizzle<3,4,4>
```

you get:

```text
physical low-3 chunk bits = logical_low3 XOR row
```

For logical chunk `0`:

| row | logical chunk | physical chunk | banks |
| --: | ------------: | -------------: | ----: |
|   0 |             0 |              0 |   0–3 |
|   1 |             0 |              1 |   4–7 |
|   2 |             0 |              2 |  8–11 |
|   3 |             0 |              3 | 12–15 |
|   4 |             0 |              4 | 16–19 |
|   5 |             0 |              5 | 20–23 |
|   6 |             0 |              6 | 24–27 |
|   7 |             0 |              7 | 28–31 |

So again:

```text
row0 →  0-3
row1 →  4-7
row2 →  8-11
...
row7 → 28-31
```

The extra fourth chunk bit `c3` doesn't matter for bank selection because it represents:

```text
c3 = 0 → first 128 B half
c3 = 1 → second 128 B half
```

and banks repeat after 128 B.

That's a powerful way to reason about this:

```text
8 × 64 BF16
row = 128B
address:
[row 3b][chunk 3b][16B]
          ↑
Swizzle<3,4,3>

8 × 128 BF16
row = 256B
address:
[row 3b][chunk 4b][16B]
         ↑   ↑
      high  low3

Swizzle<3,4,4>
      row3 ───── XOR ───→ chunk low3
```

So in general, don't choose `S` based just on the number of columns. Think:

> **Where are my row bits in the byte address, and where are the bank-selecting chunk bits?**

Then `S` is basically the **bit-distance between those two fields**. CuTe's source defines `S` exactly as that shift distance between the `YYY` and `ZZZ` masks. ([GitHub][1])

For these two cases:

```text
BF16 8×64:
row stride = 128B = 2^7
Z starts at bit 4
Y starts at bit 7
S = 7 - 4 = 3

→ Swizzle<3,4,3>


BF16 8×128:
row stride = 256B = 2^8
Z starts at bit 4
Y starts at bit 8
S = 8 - 4 = 4

→ Swizzle<3,4,4>
```

That formula is the useful thing to carry forward.

[1]: https://github.com/NVIDIA/cutlass/blob/main/include/cute/swizzle.hpp?utm_source=chatgpt.com "cutlass/include/cute/swizzle.hpp at main · NVIDIA/cutlass · GitHub"

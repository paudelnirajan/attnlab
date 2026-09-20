# Wire format — attention patterns (`ATNP` v1)

Binary format for shipping attention tensors from the API to the browser.

**Fix this before writing any UI code.** It is the one contract both ends hard-depend on, and the
reason the whole project is viable at all: a naive JSON encoding of `gpt2-small` at 512 tokens is
~700 MB of text. This format gets the same data to the browser one layer at a time in under 2 MB.

Throughout, **MB = 10^6 bytes**.

---

## 1. The problem

Attention is a `n_layers x n_heads x seq x seq` tensor of probabilities in `[0, 1]`.
For `gpt2-small` that is `12 x 12 = 144` head-matrices.

| seq | values | float32 | uint8 square | uint8 + causal triangle |
|---:|---:|---:|---:|---:|
| 64 | 589,824 | 2.36 MB | 0.59 MB | **0.30 MB** |
| 128 | 2,359,296 | 9.44 MB | 2.36 MB | **1.19 MB** |
| 256 | 9,437,184 | 37.75 MB | 9.44 MB | **4.74 MB** |
| 512 | 37,748,736 | 151.0 MB | 37.75 MB | **18.91 MB** |

Even the best full-model number is too big to ship on every keystroke. **So we never ship the whole
model at once — we ship per layer.** For `gpt2-small` (12 heads/layer), one layer costs:

| seq | uint8 + triangle, one layer |
|---:|---:|
| 64 | 24,960 B (0.025 MB) |
| 128 | 99,072 B (0.099 MB) |
| 256 | 394,752 B (0.39 MB) |
| 512 | 1,575,936 B (1.58 MB) |

Sub-megabyte through 256 tokens, before gzip. That is the design.

---

## 2. Quantization: sqrt companding

Attention probabilities are **not uniformly distributed** — softmax concentrates most mass on a few
positions, and the interesting structure (induction stripes, previous-token diagonals) often lives in
the long tail. Researchers routinely view attention on a log scale.

**Linear uint8 destroys that tail.** Resolution is `1/255 = 0.0039`, so every probability below
~0.002 quantizes to exactly zero. On a linear heatmap you'd never notice; switch to log scale and
half the plot is empty.

**Fix, at zero cost in bytes:**

```
encode:  u = round(255 * sqrt(p))          # p in [0,1] -> u in [0,255]
decode:  p = (u / 255)^2
```

### Error characteristics

With quantization step `du = ±0.5`, and `dp/du = 2*sqrt(p)/255`:

```
absolute error  |dp|     <=  sqrt(p) / 255
relative error  |dp|/p   <=  1 / (255 * sqrt(p))
```

| | linear uint8 | **sqrt uint8** |
|---|---|---|
| smallest nonzero representable | 3.9e-3 | **1.5e-5** (260x better) |
| max absolute error | 0.0020 | 0.0039 (at p=1) |
| relative error at p = 0.1 | 2.0% | **1.2%** |
| relative error at p = 0.01 | 19.6% | **3.9%** |
| relative error at p = 0.001 | >100% (rounds to 0) | **12.4%** |

The trade is deliberate: sqrt companding is slightly *worse* in absolute terms near `p = 1` and
dramatically better everywhere else. For a heatmap that is exactly right — values near 1 are "bright"
regardless, while the difference between `p = 0.0001` and `p = 0` is the whole story in the tail.

> **Correction to an earlier draft.** An acceptance criterion of "max absolute error < 0.002 for
> p > 0.01" was written into the first version of the plan. That is the *linear* bound and sqrt
> companding fails it at large `p` by design. The correct criteria are below.

### Acceptance criteria (`tests/test_patterns.py`)

- relative error `<= 4%` for all `p >= 0.01`
- absolute error `<= 0.004` for all `p`
- round-trip of an all-zeros and an all-ones matrix is exact
- decoded rows sum to `1.0 ± 0.02` on real model output

**Do not renormalize rows after decoding.** It would hide encoding errors. The UI may *display* row
sums as a sanity readout, but the data stays as decoded.

### Rejected alternatives

| Option | Why not |
|---|---|
| float32 | 4x the bytes for precision no display can use |
| float16 | 2x the bytes; `Float16Array` is too new to rely on across browsers |
| linear uint8 | kills the log-scale view (above) |
| log companding | better still in the deep tail, but needs an epsilon floor and an extra parameter; revisit only if sqrt proves insufficient |
| per-row max normalization | loses cross-row comparability, which is often the point |

**uint16** stays available behind a flag bit for anyone who needs near-exact values (e.g. exporting
data for analysis). It is never the default.

---

## 3. Causal triangle packing

Decoder-only attention is causally masked: for destination row `i`, all sources `j > i` are exactly
zero. Storing them wastes just under half the payload.

Packed length per head: `seq * (seq + 1) / 2`, laid out row-major, row `i` contributing `i + 1`
values (sources `0..i` inclusive).

Offset of row `i` within a head: `i * (i + 1) / 2`.

Set flag bit 0 when packed. The decoder expands into a zero-filled square, or — better — indexes the
packed array directly and skips the expansion entirely.

> Only valid for causal models. If a non-causal model is ever added, emit flag bit 0 = 0 and send the
> full square. The flag exists so this never becomes a silent correctness bug.

---

## 4. Byte layout

All integers little-endian. Header is 16 bytes.

```
offset  size  field        notes
------  ----  -----------  ---------------------------------------------
     0     4  magic        ASCII "ATNP"
     4     1  version      = 1
     5     1  flags        bit0  1 = causal-triangle packed
                           bit1  1 = uint8, 0 = uint16
                           bit2  1 = sqrt-companded
                           bit3-7 reserved, must be 0
     6     2  reserved     must be 0
     8     2  n_layers     number of layers in THIS response
    10     2  n_heads      heads per layer
    12     4  seq          sequence length
------  ----  -----------  ---------------------------------------------
    16  2*n_layers  layer_ids   uint16 each, model-absolute layer indices
------  ----  -----------  ---------------------------------------------
   ...   payload
```

Payload, in order: for each layer (in `layer_ids` order), for each head `0..n_heads-1`, the values
for that head.

Per-head value count:
- triangle-packed: `seq * (seq + 1) / 2`
- full square: `seq * seq`

Element size: 1 byte if flag bit1 set, else 2 bytes.

**Default configuration:** `flags = 0b111` — triangle + uint8 + sqrt.

### Transport

`Content-Type: application/octet-stream`, with gzip applied by ASGI middleware
(`Content-Encoding: gzip`). Attention is sparse enough that gzip typically removes a further 30–60%.
**Measure the real ratio in Stage 0a** rather than trusting that range.

Do not compress twice; do not set `Content-Encoding` by hand.

---

## 5. Worked example

`gpt2-small`, `seq = 256`, requesting layers 0 and 3, default flags:

```
header                16 B
layer_ids (2)          4 B
payload  2 layers x 12 heads x 32,896 values x 1 B  =  789,504 B
                                             total =  789,524 B  (0.79 MB)
```

After gzip, expect roughly 0.3–0.5 MB. One layer alone is ~0.39 MB.

---

## 6. Reference decoder (TypeScript)

```ts
export interface Patterns {
  layerIds: number[];
  nHeads: number;
  seq: number;
  packed: boolean;
  /** get attention prob for (layer index within response, head, dest, src) */
  at(l: number, h: number, dest: number, src: number): number;
}

const MAGIC = 0x504e5441; // "ATNP" little-endian

export function decode(buf: ArrayBuffer): Patterns {
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== MAGIC) throw new Error("bad magic");
  const version = dv.getUint8(4);
  if (version !== 1) throw new Error(`unsupported version ${version}`);

  const flags    = dv.getUint8(5);
  const packed   = (flags & 1) !== 0;
  const isU8     = (flags & 2) !== 0;
  const sqrtComp = (flags & 4) !== 0;

  const nLayers = dv.getUint16(8, true);
  const nHeads  = dv.getUint16(10, true);
  const seq     = dv.getUint32(12, true);

  const layerIds: number[] = [];
  for (let i = 0; i < nLayers; i++) layerIds.push(dv.getUint16(16 + 2 * i, true));

  const dataOff = 16 + 2 * nLayers;
  const perHead = packed ? (seq * (seq + 1)) / 2 : seq * seq;
  const raw = isU8
    ? new Uint8Array(buf, dataOff)
    : new Uint16Array(buf, dataOff);
  const maxVal = isU8 ? 255 : 65535;

  const at = (l: number, h: number, dest: number, src: number): number => {
    if (packed && src > dest) return 0;
    const within = packed ? (dest * (dest + 1)) / 2 + src : dest * seq + src;
    const v = raw[(l * nHeads + h) * perHead + within] / maxVal;
    return sqrtComp ? v * v : v;
  };

  return { layerIds, nHeads, seq, packed, at };
}
```

**Rendering note:** do not call `at()` per pixel for a full heatmap. Walk the packed array linearly
and write straight into `ImageData` — one pass, no per-cell function call, no square expansion.

---

## 7. Versioning

`version` is a hard gate, not a hint. A decoder that sees an unknown version must throw, not guess.

Adding a flag bit is backward-compatible (old decoders reject nonzero reserved bits, which is the
correct behaviour). Changing the header layout or payload ordering requires `version = 2`.

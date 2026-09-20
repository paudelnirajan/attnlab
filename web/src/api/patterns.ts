// Decoder for the ATNP binary wire format — see docs/01-wire-format.md,
// which this is a direct implementation of (that doc's "reference
// decoder" section is this file; keep them in sync). The Python encoder
// is attnlab/patterns.py; tests/test_patterns.py pins the encoder's
// error bounds, so this decoder is checked against real API responses
// in the app rather than duplicating those bounds here.

const MAGIC = 0x504e5441; // "ATNP" little-endian

export interface DecodedPatterns {
  layerIds: number[];
  nHeads: number;
  seq: number;
  packed: boolean;
  /** size of the payload this was decoded from, so the cost panel can report
   * the real wire cost against the naive float32 figure */
  byteLength: number;
  /** attention probability for (layer index WITHIN this response, head, dest, src) */
  at(l: number, h: number, dest: number, src: number): number;
  /** raw quantized byte for the same coordinates — used by the canvas
   * renderer to skip the float round-trip entirely (see render/heatmap.ts) */
  rawAt(l: number, h: number, dest: number, src: number): number;
}

export function decodePatterns(buf: ArrayBuffer): DecodedPatterns {
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== MAGIC) {
    throw new Error("decodePatterns: bad magic — not an ATNP payload");
  }
  const version = dv.getUint8(4);
  if (version !== 1) {
    throw new Error(`decodePatterns: unsupported version ${version}`);
  }

  const flags = dv.getUint8(5);
  const packed = (flags & 1) !== 0;
  const isU8 = (flags & 2) !== 0;
  const sqrtComp = (flags & 4) !== 0;

  const nLayers = dv.getUint16(8, true);
  const nHeads = dv.getUint16(10, true);
  const seq = dv.getUint32(12, true);

  const layerIds: number[] = [];
  for (let i = 0; i < nLayers; i++) {
    layerIds.push(dv.getUint16(16 + 2 * i, true));
  }

  const dataOff = 16 + 2 * nLayers;
  const perHead = packed ? (seq * (seq + 1)) / 2 : seq * seq;
  const raw: Uint8Array | Uint16Array = isU8
    ? new Uint8Array(buf, dataOff)
    : new Uint16Array(buf, dataOff);
  const maxVal = isU8 ? 255 : 65535;

  const rawAt = (l: number, h: number, dest: number, src: number): number => {
    if (packed && src > dest) return 0;
    const within = packed ? (dest * (dest + 1)) / 2 + src : dest * seq + src;
    return raw[(l * nHeads + h) * perHead + within];
  };

  const at = (l: number, h: number, dest: number, src: number): number => {
    const v = rawAt(l, h, dest, src) / maxVal;
    return sqrtComp ? v * v : v;
  };

  return { layerIds, nHeads, seq, packed, byteLength: buf.byteLength, at, rawAt };
}

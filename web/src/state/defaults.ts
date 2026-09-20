// Shared by the store and by urlSync. Separate module purely to keep those two
// from importing each other: the store now seeds itself straight from the URL,
// and urlSync needs these same values to decide what to omit.

export type Direction = "dest2src" | "src2dest";

export const DEFAULT_MODEL = "attn-only-2l-demo";
export const DEFAULT_PROMPT = "The quick brown fox jumps over the lazy dog";
export const DEFAULT_DIRECTION: Direction = "dest2src";

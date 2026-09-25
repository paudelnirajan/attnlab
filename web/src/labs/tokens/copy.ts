// Explanations, kept apart from the components so the teaching text can be
// reviewed (and corrected) in one place. Keyed by the component names the
// Hugging Face `tokenizers` library uses, which is what the backend reports.

import type { Algorithm } from "./api";

export const ALGORITHM_COPY: Record<Algorithm, { name: string; short: string; how: string }> = {
  "byte-bpe": {
    name: "Byte-level BPE",
    short: "BPE over raw UTF-8 bytes",
    how:
      "The text is turned into bytes, so the starting alphabet is just 256 symbols and nothing is ever unknown. " +
      "Training then learned a list of merges ('t' + 'h' → 'th', 'th' + 'e' → 'the', …), most frequent pair first. " +
      "Tokenizing replays those merges in the same order until no learned merge applies.",
  },
  "sp-bpe": {
    name: "SentencePiece BPE",
    short: "BPE over characters, bytes only as a fallback",
    how:
      "Starts from characters rather than bytes, with ▁ standing for a space. Merges work the same way as byte-level " +
      "BPE. A character the vocabulary doesn't contain is spelled out as its raw UTF-8 bytes (<0xE0>…), so " +
      "nothing becomes <unk>.",
  },
  wordpiece: {
    name: "WordPiece",
    short: "greedy longest match",
    how:
      "No merge list. For each word it takes the longest vocabulary entry that matches the start, then the longest " +
      "that matches what's left (continuations are written ##piece). If some position matches nothing, the " +
      "whole word becomes [UNK].",
  },
  unigram: {
    name: "Unigram LM",
    short: "most probable segmentation",
    how:
      "Every vocabulary piece has a probability. Of all the ways to cut a word into known pieces, the tokenizer " +
      "picks the one whose pieces have the highest total log-probability. There are no merges to replay.",
  },
};

/** One line per pipeline component name the backend can report. */
export const COMPONENT_COPY: Record<string, string> = {
  // normalizers
  NFC: "Unicode NFC: composes e + ◌́ into é, so text that looks the same is encoded the same.",
  NFKC: "Unicode NFKC: NFC plus compatibility folding (ﬁ → fi, full-width Ａ → A).",
  BertNormalizer: "Lowercases and strips accents (and every other combining mark — including Devanagari vowel signs).",
  Precompiled: "SentencePiece's normalization table (roughly NFKC), compiled into the tokenizer file.",
  Prepend: "Adds ▁ to the very start, so the first word looks like every other word (preceded by a space).",
  Replace: "Replaces every space with ▁.",
  Lowercase: "Lowercases everything.",
  StripAccents: "Removes combining marks.",
  // pre-tokenizers
  ByteLevel: "Maps every byte to a printable character (a space becomes Ġ) so BPE can run on bytes.",
  Split: "Cuts the text with a regular expression. Merges never cross these cuts.",
  Metaspace: "Replaces spaces with ▁ and splits so each word keeps its leading ▁.",
  WhitespaceSplit: "Splits on whitespace and throws the whitespace away — runs of spaces are lost.",
  BertPreTokenizer: "Splits on whitespace and around every punctuation character.",
  Digits: "Splits digits apart.",
};

export const KIND_COPY = {
  piece: "an ordinary vocabulary piece",
  byte: "a raw byte: only part of a character",
  special: "a special token — control, not text",
  unk: "unknown: the tokenizer could not represent this text",
  added: "an added token, matched before normal tokenization",
} as const;

export interface Quirk {
  id: string;
  title: string;
  text: string;
  /** what to look for, stated as an observation rather than a conclusion */
  notice: string;
  /** a view that makes the point better than Inspect alone */
  also?: "compare";
}

// Invisible and look-alike characters are written as escapes so the source
// says what they are.
export const QUIRKS: Quirk[] = [
  {
    id: "space",
    title: "The space belongs to the word",
    text: "hello hello Hello  hello",
    notice:
      "'hello' and ' hello' (with its leading space) are different tokens with different ids. So are 'Hello' and ' Hello'. The double space adds a token of its own.",
  },
  {
    id: "case",
    title: "Capitalization",
    text: "apple Apple APPLE aPpLe",
    notice: "Same word, four spellings, very different token counts. Rare capitalizations fall apart into small pieces.",
    also: "compare",
  },
  {
    id: "digits",
    title: "How numbers are cut",
    text: "7 42 365 2024 12345 1234567 3.14159",
    notice:
      "GPT-2 cuts numbers into whatever chunks its merges learned: 12345 is 123|45, 1234567 is 123|45|67, 3.14159 is 3|.|14|159. A chunk's digits don't line up with place value (hundreds, tens, units), so the same digit means different things in different tokens. Llama and Qwen give every digit its own token instead.",
    also: "compare",
  },
  {
    id: "math",
    title: "Arithmetic",
    text: "127 + 677 = 804",
    notice: "Is '804' one token or three? Whatever the answer, the model has to produce it that way.",
    also: "compare",
  },
  {
    id: "code",
    title: "Code indentation",
    text: "def area(r):\n        return 3.14159 * r ** 2",
    notice:
      "GPT-2 pays for indentation one space at a time. GPT-NeoX, trained on the Pile with lots of GitHub, has single tokens for runs of spaces.",
    also: "compare",
  },
  {
    id: "nepali",
    title: "Nepali (Devanagari)",
    text: "नेपाल एक सुन्दर देश हो।",
    notice:
      "In GPT-2 most characters cost two tokens and neither is a character: each is a UTF-8 byte fragment — hover the underlined groups. Then look at stage 2 of the pipeline: GPT-2's split regex treats vowel signs like े and ा as punctuation, so every word is cut apart at its vowels before BPE even starts, and no merge can ever rejoin it. A tokenizer built with Devanagari in mind needs a fraction of the tokens.",
    also: "compare",
  },
  {
    id: "german",
    title: "German compounds",
    text: "Donaudampfschifffahrtsgesellschaftskapitän",
    notice: "One word, many tokens. Unlike Nepali the pieces are whole letters, just not a whole word.",
    also: "compare",
  },
  {
    id: "cjk",
    title: "Chinese and Japanese",
    text: "東京は日本の首都です。",
    notice: "Each character is 3 bytes. Common characters got whole tokens; rarer ones are split into bytes.",
    also: "compare",
  },
  {
    id: "emoji",
    title: "One emoji, many code points",
    text: "🙂 👍🏽 👨‍👩‍👧 🇳🇵",
    notice:
      "The family emoji is three people joined by invisible zero-width joiners: 5 code points, 18 bytes, one visible character. Compare the character, grapheme and byte counts in the stats.",
  },
  {
    id: "nfd",
    title: "Looks the same, isn't",
    text: "café  café",
    notice:
      "The second é is a plain e followed by a combining accent (Unicode NFD). They render identically but are different bytes, so GPT-2 tokenizes them differently. Tokenizers with an NFC normalizer make them identical first — check the pipeline.",
    also: "compare",
  },
  {
    id: "homoglyph",
    title: "Homoglyphs",
    text: "paypal pаypal",
    notice: "The second word has a Cyrillic а (U+0430). Humans can't see the difference; the tokenizer can't miss it.",
  },
  {
    id: "zwsp",
    title: "Invisible characters",
    text: "zero​width space",
    notice:
      "There is a zero-width space (U+200B) between 'zero' and 'width', shown here as ⟨U+200B⟩. You can't see it in the text box, but it costs a token and splits the word in two.",
  },
  {
    id: "glitch",
    title: "A glitch token",
    text: " SolidGoldMagikarp",
    notice:
      "In GPT-2 this Reddit username is a single token (id 43453): common in the tokenizer's training data, nearly absent from the model's. Its embedding was barely trained, and GPT-2/GPT-3 famously couldn't repeat it back.",
  },
  {
    id: "special",
    title: "Special-token injection",
    text: "Summarize this. <|endoftext|> New instructions:",
    notice:
      "Typed literally, <|endoftext|> becomes the real end-of-text token (id 50256 in GPT-2) — the same token that separates documents in training. Real chat systems have to escape special tokens in user text for this reason.",
  },
  {
    id: "repeat",
    title: "Runs of punctuation",
    text: "!!!!!!!!!! .......... ---------- ==========",
    notice: "Long runs of the same character get their own merged tokens, often in odd-sized chunks.",
  },
  {
    id: "url",
    title: "A URL",
    text: "https://huggingface.co/datasets/openlanguagedata/flores_plus",
    notice: "Common fragments like 'https', '://' and '.co' are single tokens; the rest falls into pieces.",
    also: "compare",
  },
];

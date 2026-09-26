// Teaching text for the logit lens lab, kept apart from the components so it
// can be reviewed (and corrected) in one place.

import type { LensName } from "./api";
import { DEFAULT_TEXT, type Metric, type View } from "./store";

export const VIEW_TABS: { id: View; label: string; question: string }[] = [
  { id: "grid", label: "Lens", question: "What would the model predict if it stopped after each layer?" },
  { id: "trajectory", label: "Trajectory", question: "How does one position's prediction change on the way up?" },
  { id: "attribution", label: "Attribution", question: "Which layers and heads wrote the final answer?" },
  { id: "layers", label: "Layers", question: "How close is each layer to the final answer, on average?" },
  { id: "hood", label: "Under the hood", question: "What exactly is computed, and how do we know it's right?" },
];

export const LENS_COPY: Record<LensName, { label: string; short: string; how: string }> = {
  ln_final: {
    label: "ln_final",
    short: "the model's own final LayerNorm",
    how:
      "Each row is normalized with the model's final LayerNorm, learned scale w and bias b included, then multiplied " +
      "by W_U. This is exactly what the last layer's output goes through, applied to earlier layers. It assumes " +
      "earlier layers use the same coordinates as the last one, which is only roughly true.",
  },
  plain: {
    label: "plain norm",
    short: "mean 0, std 1, no learned weights",
    how:
      "Each row is centred and scaled to unit variance, with no learned weights, then multiplied by the raw W_U. It " +
      "assumes nothing about the layer, but a few residual dimensions are enormous. ln_final learned to shrink " +
      "them, and without it they swamp the prediction. See Under the hood.",
  },
};

/** A model with no output norm reads its stream as W_U · x directly; calling that "ln_final" would be false. */
export function lensLabel(lens: LensName, hasNorm: boolean): string {
  return hasNorm ? LENS_COPY[lens].label : "W_U · x (no norm)";
}

export const METRIC_COPY: Record<Metric, { label: string; question: string; color: string; value: string }> = {
  top1: {
    label: "Top-1 prob",
    question: "How sure is this row of its own best guess?",
    color: "stronger shade = the top guess has more probability",
    value: "probability of the row's #1 token",
  },
  p_next: {
    label: "P(actual next)",
    question: "How much probability does this row give the token that really comes next?",
    color: "stronger shade = more probability on the actual next token",
    value: "probability of the actual next token",
  },
  rank_next: {
    label: "Rank of actual next",
    question: "When (if ever) does the right answer reach the top?",
    color: "stronger shade = closer to rank 1 (log scale)",
    value: "rank of the actual next token, 1 = top",
  },
  rank_final: {
    label: "Rank of final answer",
    question: "When does the model's eventual answer rise to the top?",
    color: "stronger shade = closer to rank 1 (log scale)",
    value: "rank of the output row's #1 token",
  },
  entropy: {
    label: "Entropy",
    question: "How spread out is this row's distribution?",
    color: "stronger shade = lower entropy (more concentrated)",
    value: "entropy in nats; ln(vocab) is a uniform guess",
  },
  kl: {
    label: "KL to output",
    question: "How far is this row's distribution from the model's final one?",
    color: "stronger shade = closer to the output (shade is e^−KL)",
    value: "KL(output ‖ row) in nats",
  },
};

export interface Example {
  id: string;
  title: string;
  text: string;
  /** what to look for, stated as an observation rather than a conclusion */
  notice: string;
  /** tokens worth following in Trajectory (strings, leading space included) */
  track?: string[];
  target?: string;
  contrast?: string;
  model?: string;
}

export const EXAMPLES: Example[] = [
  {
    id: "copy",
    title: "Copying a rare word",
    text: DEFAULT_TEXT,
    notice:
      "In the second half the model can copy ' plasma' from the first. Follow the last ' say' column: under " +
      "ln_final the word climbs from rank ~24k to rank 1 between blocks 6 and 9, which is when the copying " +
      "mechanism (induction heads) arrives. Switch to the plain lens and it never gets there.",
    track: [" plasma"],
  },
  {
    id: "fact",
    title: "A fact",
    text: "The Eiffel Tower is located in the city of",
    notice:
      "Watch when a city appears at all, and when it becomes this city. Small models often get the category " +
      "(some city) many layers before they get the answer, and gpt2-small's answer here is not the right one.",
    track: [" Paris", " London"],
    target: " Paris",
    contrast: " London",
  },
  {
    id: "ioi",
    title: "Who gets the drink?",
    text: "When Mary and John went to the store, John gave a drink to",
    notice:
      "Indirect object identification: the answer is the name that was not repeated. The logit difference between " +
      "' Mary' and ' John' is the classic measure. In Attribution, a few late heads carry most of it (the " +
      "'name movers' of Wang et al., 2022).",
    track: [" Mary", " John"],
    target: " Mary",
    contrast: " John",
  },
  {
    id: "days",
    title: "Continue a sequence",
    text: "Monday, Tuesday, Wednesday, Thursday,",
    notice: "An easy, pattern-driven prediction. Compare how early ' Friday' locks in with the fact example.",
    track: [" Friday"],
  },
  {
    id: "code",
    title: "Code",
    text: "def add(a, b):\n    return a +",
    notice: "The answer is in the context, and the model has to pick the other argument. Is ' b' ever beaten by ' a'?",
    track: [" b", " a"],
    target: " b",
    contrast: " a",
  },
  {
    id: "opposite",
    title: "Opposites",
    text: "The opposite of hot is",
    notice: "Early rows often predict something related to 'hot' (a synonym, a continuation). The flip to the opposite comes later.",
    track: [" cold"],
  },
  {
    id: "nepali",
    title: "Byte fragments",
    text: "नेपाल एक सुन्दर देश हो।",
    notice:
      "Most positions here are half a character. Early rows confidently predict a byte range such as ऄ–ऽ⋯, which " +
      "only means 'a Devanagari character comes next'. After an input ending mid-character, the only sensible " +
      "prediction is the byte that finishes it (⋯X). A ✓ on those cells is byte completion, not language.",
  },
];

export const GLOSSARY = {
  row: "A point in the residual stream: after the embedding, after a block's attention (resid_mid), or after its MLP (resid_post).",
  output: "The top row is not a lens. It is the model's real output, which every lens row is compared against.",
  predicts: "Column i predicts token i+1, so read each column against the next token (shown on top), not its own.",
};

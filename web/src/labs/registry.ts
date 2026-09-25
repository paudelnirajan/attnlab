// The lab catalogue, and the order a learner is meant to walk it in.
//
// Each lab is its own page with its own URL state, but they are one course:
// every lab names the one before and after it (PathNav), links out to the
// neighbouring lab with the SAME text and a matching model, and the home page
// lays them out as a path. Adding a lab is an entry here plus a route in
// main.tsx.

export type LabStatus = "live" | "planned";

export interface LabInfo {
  id: string;
  path: string;
  step: number;
  title: string;
  /** the one question the lab answers */
  question: string;
  summary: string;
  learn: string[];
  status: LabStatus;
}

export const LABS: LabInfo[] = [
  {
    id: "tokens",
    path: "/tokens",
    step: 1,
    title: "Tokenizer lab",
    question: "What does the model actually read?",
    summary:
      "Before any attention head runs, your text is cut into pieces. The cut decides what a 'position' is, how much a prompt costs, and why some languages are harder than others.",
    learn: [
      "How BPE builds tokens out of bytes, one merge at a time",
      "Why the same sentence costs 1× in English and 8× in Nepali",
      "Quirks that change behaviour: leading spaces, digits, glitch tokens",
    ],
    status: "live",
  },
  {
    id: "attention",
    path: "/attention",
    step: 2,
    title: "Attention patterns",
    question: "Where does each token look?",
    summary:
      "Every head in every layer, as a heatmap over the tokens from step 1. Hover a token to see what it attends to, and what attends to it.",
    learn: [
      "How to read a causal attention pattern",
      "Previous-token heads, attention sinks, and positional structure",
      "What fragmented tokens do to the patterns",
    ],
    status: "live",
  },
  {
    id: "induction",
    path: "/induction",
    step: 3,
    title: "Induction heads",
    question: "How does a model copy what it has seen before?",
    summary: "Repeated random tokens, per-head induction scores, and the loss drop in the second half of the sequence.",
    learn: ["Scoring heads by attention diagonal", "Why the loss falls on repeated text", "Composition between layers"],
    status: "planned",
  },
  {
    id: "logit-lens",
    path: "/logit-lens",
    step: 4,
    title: "Logit lens",
    question: "When does the model know the answer?",
    summary: "Decode the residual stream after every layer and watch the prediction form.",
    learn: ["The residual stream as a running prediction", "Normalization choices and their effect", "Rank heatmaps"],
    status: "planned",
  },
  {
    id: "ablation",
    path: "/ablation",
    step: 5,
    title: "Ablation & attribution",
    question: "Which components actually matter?",
    summary: "Zero or mean-ablate heads and measure the change in loss and logits.",
    learn: ["Zero vs mean ablation", "Direct logit attribution", "Why correlation in patterns isn't causation"],
    status: "planned",
  },
];

export function labById(id: string): LabInfo {
  const lab = LABS.find((l) => l.id === id);
  if (!lab) throw new Error(`unknown lab ${id}`);
  return lab;
}

/** A link into a lab carrying state — how one lab hands its text to the next. */
export function labHref(id: string, params: Record<string, string | undefined> = {}): string {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== "") qs.set(k, v);
  const s = qs.toString();
  return s ? `${labById(id).path}?${s}` : labById(id).path;
}

/** Which lab the current URL is showing; "home" for the path overview. */
export function currentLabId(pathname: string = window.location.pathname): string {
  const p = pathname.replace(/\/+$/, "") || "/";
  return LABS.find((l) => l.path === p)?.id ?? "home";
}

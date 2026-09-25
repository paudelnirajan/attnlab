import { useMemo, useState } from "react";
import { toklab, type CountResult } from "../api";
import { TokenizerPills } from "../components/TokenizerPicker";
import { FLORES } from "../data/flores";
import { useRemote } from "../hooks";
import { useTokLab } from "../store";

const ENGLISH = "eng_Latn";

/** Sequential single-hue tint for a cost ratio: nothing at ≤1× (no premium),
 * saturating at 16× on a log scale, because the interesting range spans an
 * order of magnitude. */
function tint(ratio: number): { background?: string; color?: string } {
  if (!(ratio > 1)) return {};
  const pct = Math.round(Math.min(1, Math.log2(ratio) / 4) * 85);
  return {
    background: `color-mix(in oklab, var(--accent) ${pct}%, var(--surface-1))`,
    color: pct > 50 ? "var(--accent-ink)" : undefined,
  };
}

type Sort = { by: "order" } | { by: "tokenizer"; id: string };

/**
 * The same sentences, professionally translated into 32 languages, counted by
 * every selected tokenizer. Because the meaning is held fixed, the ratio to
 * English is purely the tokenizer's doing: the "tokenization tax" a language
 * pays in context length, compute and API cost.
 */
export function LanguagesView() {
  const compare = useTokLab((s) => s.compare);
  const tokenizers = useTokLab((s) => s.tokenizers);
  const inspect = useTokLab((s) => s.inspect);
  const [sentence, setSentence] = useState<number | "all">("all");
  const [sort, setSort] = useState<Sort>({ by: "order" });

  const langs = FLORES.languages;
  const texts = useMemo(
    () => langs.map((l) => (sentence === "all" ? l.sentences.join(" ") : l.sentences[sentence])),
    [langs, sentence],
  );
  const remote = useRemote<CountResult>(
    JSON.stringify([compare, sentence]),
    () => toklab.count(compare, texts),
    { enabled: tokenizers.length > 0, delay: 50 },
  );

  const byId = new Map(tokenizers.map((t) => [t.id, t]));
  const enIdx = langs.findIndex((l) => l.code === ENGLISH);
  const data = remote.data;
  const counts = new Map(data?.results.map((r) => [r.tokenizer, r.counts]) ?? []);
  const ratio = (tid: string, i: number) => {
    const c = counts.get(tid);
    return c ? c[i] / Math.max(1, c[enIdx]) : NaN;
  };

  const sortId = sort.by === "tokenizer" && counts.has(sort.id) ? sort.id : null;
  const order = langs.map((_, i) => i);
  if (sortId) order.sort((a, b) => ratio(sortId, b) - ratio(sortId, a));

  // The headline: the most expensive language for the most-used tokenizer here.
  const focus = sortId ?? compare.find((id) => counts.has(id));
  const worst = focus ? order.reduce((w, i) => (ratio(focus, i) > ratio(focus, w) ? i : w), enIdx) : null;

  return (
    <div className="stack">
      <div className="setup setup--wide">
        <TokenizerPills />
        <div className="field">
          <span className="label">Sentences</span>
          <div className="segmented" role="group" aria-label="Which sentences">
            <button type="button" className="segmented__opt" aria-pressed={sentence === "all"} onClick={() => setSentence("all")}>
              All 5
            </button>
            {langs[enIdx].sentences.map((_, i) => (
              <button
                key={i}
                type="button"
                className="segmented__opt"
                aria-pressed={sentence === i}
                onClick={() => setSentence(i)}
              >
                #{i + 1}
              </button>
            ))}
          </div>
          <p className="muted" style={{ fontSize: "var(--t-sm)" }}>
            English: “{sentence === "all" ? langs[enIdx].sentences[0] + " …" : langs[enIdx].sentences[sentence]}”
          </p>
        </div>
      </div>

      {remote.error && (
        <div className="banner banner--error" role="alert">
          <span>{remote.error}</span>
        </div>
      )}

      {focus && worst !== null && data && (
        <p className="headline">
          With <strong>{byId.get(focus)?.label}</strong>, the same meaning costs{" "}
          <strong className="mono">×{ratio(focus, worst).toFixed(1)}</strong> as many tokens in{" "}
          <strong>{langs[worst].name}</strong> as in English. Every one of those extra tokens is context the model can't
          use for anything else, and compute somebody pays for.
        </p>
      )}

      {data && (
        <section className={remote.loading ? "card is-stale" : "card"}>
          <div className="card__head">
            <h2 className="card__title">Tokens for the same meaning</h2>
            <span className="card__hint">
              ×EN is the ratio to English; darker = a bigger premium (log scale, full at ×16) · click a column to sort
              · click a language to inspect it
            </span>
          </div>
          <div className="tablewrap">
            <table className="langtable">
              <thead>
                <tr>
                  <th scope="col">
                    <button type="button" className="linkbtn" onClick={() => setSort({ by: "order" })}>
                      Language {sort.by === "order" && "·"}
                    </button>
                  </th>
                  <th scope="col" className="num" title="characters as a reader counts them (grapheme clusters)">
                    chars
                  </th>
                  {data.results.map((r) => (
                    <th key={r.tokenizer} scope="col" className="num">
                      <button
                        type="button"
                        className="linkbtn"
                        aria-sort={sortId === r.tokenizer ? "descending" : undefined}
                        onClick={() => setSort({ by: "tokenizer", id: r.tokenizer })}
                      >
                        {byId.get(r.tokenizer)?.label ?? r.tokenizer} {sortId === r.tokenizer && "↓"}
                      </button>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {order.map((i) => {
                  const l = langs[i];
                  return (
                    <tr key={l.code} className={i === enIdx ? "is-baseline" : undefined}>
                      <th scope="row">
                        <button
                          type="button"
                          className="linkbtn langtable__name"
                          title={`Inspect the ${l.name} text`}
                          onClick={() => inspect(texts[i], focus ?? undefined)}
                        >
                          {l.name}
                        </button>{" "}
                        <span className="muted" lang={l.code.slice(0, 3)}>
                          {l.native}
                        </span>
                        <span className="langtable__script">{l.script}</span>
                      </th>
                      <td className="num mono muted">{data.texts[i].n_graphemes}</td>
                      {data.results.map((r) => {
                        const x = ratio(r.tokenizer, i);
                        return (
                          <td key={r.tokenizer} className="num mono" style={tint(x)}>
                            {r.counts[i]}
                            <span className="langtable__ratio">×{x.toFixed(1)}</span>
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="card__foot">
            Sentences from{" "}
            <a href={FLORES.url} target="_blank" rel="noreferrer">
              FLORES+
            </a>{" "}
            ({FLORES.license}), professional translations of the same English source — so the differences are the
            tokenizers', not the translators'. Why some scripts cost so much more: look at how much of each vocabulary
            they got, in the <strong>Vocabulary</strong> tab.
          </div>
        </section>
      )}
    </div>
  );
}

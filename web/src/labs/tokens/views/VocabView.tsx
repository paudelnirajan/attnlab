import { toklab, type VocabRow, type VocabSearchResult, type VocabSummary } from "../api";
import { TokenizerSelect } from "../components/TokenizerPicker";
import { KIND_COPY } from "../copy";
import { useRemote } from "../hooks";
import { useTokLab } from "../store";

function pct(n: number, of: number): string {
  const p = (100 * n) / Math.max(1, of);
  if (p > 0 && p < 0.01) return "<0.01%";
  return `${p.toFixed(p < 1 ? 2 : 1)}%`;
}

function Row({ r, nMerges }: { r: VocabRow; nMerges: number }) {
  return (
    <tr>
      <td className="num mono">{r.id}</td>
      <td className="mono vocab__display">{r.display}</td>
      <td className="mono muted">{JSON.stringify(r.piece)}</td>
      <td className="mono muted">{r.byte_hex ? r.byte_hex.replace(/(..)(?!$)/g, "$1 ") : "—"}</td>
      <td className="muted">{r.script}</td>
      <td className="num mono muted" title={r.rank !== null ? `merge ${r.rank} of ${nMerges}` : KIND_COPY[r.kind]}>
        {r.rank !== null ? `#${r.rank}` : r.kind === "piece" ? "base" : r.kind}
      </td>
    </tr>
  );
}

function VocabTable({ rows, nMerges }: { rows: VocabRow[]; nMerges: number }) {
  return (
    <div className="tablewrap">
      <table className="vocab">
        <thead>
          <tr>
            <th scope="col" className="num">id</th>
            <th scope="col">spells</th>
            <th scope="col">vocab string</th>
            <th scope="col">bytes</th>
            <th scope="col">script</th>
            <th scope="col" className="num" title="the merge that created this token; lower = learned earlier = more common">
              merge
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <Row key={r.id} r={r} nMerges={nMerges} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * What the tokenizer can say in one token, and who it was built for. The
 * script breakdown is the "why" behind the Languages view: a script that got
 * 40 of 50,000 vocabulary slots cannot be anything but fragmented.
 */
export function VocabView() {
  const tokenizer = useTokLab((s) => s.tokenizer);
  const tokenizers = useTokLab((s) => s.tokenizers);
  const query = useTokLab((s) => s.query);
  const setQuery = useTokLab((s) => s.setQuery);
  const scriptFilter = useTokLab((s) => s.scriptFilter);
  const setScriptFilter = useTokLab((s) => s.setScriptFilter);
  const enabled = tokenizers.length > 0;

  const summary = useRemote<VocabSummary>(tokenizer, () => toklab.vocab(tokenizer), { enabled, delay: 0 });
  const search = useRemote<VocabSearchResult>(
    JSON.stringify([tokenizer, query, scriptFilter]),
    () => toklab.vocabSearch(tokenizer, query, scriptFilter ?? undefined),
    { enabled: enabled && (query.trim().length > 0 || scriptFilter !== null) },
  );
  const sum = summary.data?.tokenizer === tokenizer ? summary.data : null;
  const found = search.data?.tokenizer === tokenizer && (query.trim() || scriptFilter) ? search.data : null;
  const textual = sum ? sum.by_script.reduce((a, s) => a + s.count, 0) : 0;
  const maxScript = sum ? Math.max(1, ...sum.by_script.map((s) => s.count)) : 1;

  return (
    <div className="stack">
      <div className="setup">
        <TokenizerSelect id="vocab-tokenizer" />
        <div className="field">
          <label className="label" htmlFor="vocab-q">
            Search the vocabulary
            <span className="label__hint">matches what a token spells, ignoring case · a number looks up an id</span>
          </label>
          <input
            id="vocab-q"
            className="input"
            value={query}
            placeholder="e.g. token, Magikarp, 50256, नेपाल"
            onChange={(e) => setQuery(e.target.value)}
            spellCheck={false}
          />
        </div>
      </div>

      {(summary.error || search.error) && (
        <div className="banner banner--error" role="alert">
          <span>{summary.error ?? search.error}</span>
        </div>
      )}

      {found && (
        <section className={search.loading ? "card is-stale" : "card"}>
          <div className="card__head">
            <h2 className="card__title">
              {found.total.toLocaleString()} match{found.total === 1 ? "" : "es"}
              {scriptFilter && <> in {scriptFilter}</>}
            </h2>
            <span className="card__hint">
              {found.total > found.results.length && `showing the first ${found.results.length}, by id · `}
              in byte-level BPE, id order is roughly merge order
            </span>
            <span className="spacer" />
            {scriptFilter && (
              <button type="button" className="btn btn--sm btn--ghost" onClick={() => setScriptFilter(null)}>
                clear script filter
              </button>
            )}
          </div>
          <VocabTable rows={found.results} nMerges={sum?.n_merges ?? 0} />
        </section>
      )}

      {!sum ? (
        <div className="card">
          <div className="empty">
            <span className="spinner" style={{ display: "inline-block", verticalAlign: "-2px" }} /> reading the vocabulary…
          </div>
        </div>
      ) : (
        <div className="grid2">
          <section className="card">
            <div className="card__head">
              <h2 className="card__title">Who the vocabulary was built for</h2>
              <span className="card__hint">tokens by the script of their first letter · click to list them</span>
            </div>
            <div className="card__body">
              <dl className="statlist" style={{ marginBottom: "var(--s3)" }}>
                <dt>Vocabulary size</dt>
                <dd>{sum.size.toLocaleString()}</dd>
                {sum.n_merges > 0 && (
                  <>
                    <dt>Learned merges</dt>
                    <dd>{sum.n_merges.toLocaleString()}</dd>
                  </>
                )}
                <dt>Special tokens</dt>
                <dd>{((sum.kinds.special ?? 0) + (sum.kinds.unk ?? 0)).toLocaleString()}</dd>
                <dt>Tokens that are only part of a character</dt>
                <dd>{(sum.kinds.byte ?? 0).toLocaleString()}</dd>
              </dl>
              <ol className="ranked">
                {sum.by_script.map((s) => (
                  <li key={s.script}>
                    <button
                      type="button"
                      className="ranked__row ranked__row--button"
                      aria-pressed={scriptFilter === s.script}
                      onClick={() => setScriptFilter(scriptFilter === s.script ? null : s.script)}
                      title={
                        s.script === "other"
                          ? "no letters: digits, punctuation, whitespace, symbols, emoji"
                          : s.script === "partial"
                            ? "not valid text on its own: a byte fragment of some character"
                            : `tokens whose first letter is ${s.script}`
                      }
                    >
                      <span className="ranked__bar" style={{ width: `${(100 * s.count) / maxScript}%` }} />
                      <span className="ranked__label ranked__label--sans">
                        {s.script === "other" ? "no letters" : s.script === "partial" ? "byte fragments" : s.script}
                      </span>
                      <span className="ranked__value">
                        {s.count.toLocaleString()}
                        <span className="muted"> {pct(s.count, textual)}</span>
                      </span>
                    </button>
                  </li>
                ))}
              </ol>
            </div>
          </section>

          <section className="card">
            <div className="card__head">
              <h2 className="card__title">Longest word tokens</h2>
              <span className="card__hint">
                strings the tokenizer's training data repeated so often they became one token — many are glitch tokens
              </span>
            </div>
            <VocabTable rows={sum.longest} nMerges={sum.n_merges} />
            <div className="card__head" style={{ borderTop: "1px solid var(--border-hairline)" }}>
              <h2 className="card__title">Special tokens</h2>
              <span className="card__hint">control signals, not text</span>
            </div>
            <div className="card__body">
              <div className="pillrow">
                {sum.special.map((r) => (
                  <span key={r.id} className="badge mono" title={`id ${r.id}`}>
                    {r.piece} <span className="muted">{r.id}</span>
                  </span>
                ))}
              </div>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

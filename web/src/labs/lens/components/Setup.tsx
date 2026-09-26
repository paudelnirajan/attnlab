import { EXAMPLES } from "../copy";
import { useLens } from "../store";

/** Model, text and the one tokenization switch that changes what position 0 is. */
export function Setup() {
  const models = useLens((s) => s.models);
  const model = useLens((s) => s.model);
  const setModel = useLens((s) => s.setModel);
  const text = useLens((s) => s.text);
  const setText = useLens((s) => s.setText);
  const bos = useLens((s) => s.bos);
  const setBos = useLens((s) => s.setBos);
  const selected = models.find((m) => m.id === model);

  return (
    <div className="setup">
      <div className="field">
        <label className="label" htmlFor="lens-model">
          Model
        </label>
        {models.length === 0 ? (
          <div className="skeleton" style={{ height: "var(--control-h)" }} />
        ) : (
          <select id="lens-model" className="select" value={model} onChange={(e) => setModel(e.target.value)}>
            {selected === undefined && <option value={model}>{model} (not in this server's registry)</option>}
            {models.map((m) => (
              <option key={m.id} value={m.id} disabled={m.tier === "disabled"}>
                {m.label}
                {m.status === "resident" ? " · loaded" : ""}
              </option>
            ))}
          </select>
        )}
        {selected && (
          <p className="muted" style={{ fontSize: "var(--t-sm)" }}>
            <span className="mono">
              {selected.n_layers}L · d_model {selected.d_model}
            </span>{" "}
            · {selected.blurb}
          </p>
        )}
        <label className="check" title="TransformerLens prepends <|endoftext|> by default. Without it, position 0 is your first word.">
          <input type="checkbox" checked={bos} onChange={(e) => setBos(e.target.checked)} />
          prepend BOS token
        </label>
      </div>
      <div className="field">
        <label className="label" htmlFor="lens-text">
          Text
          <span className="label__hint">up to 256 tokens · each column predicts the next token</span>
        </label>
        <textarea
          id="lens-text"
          className="textarea"
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={3}
          spellCheck={false}
        />
      </div>
    </div>
  );
}

export function Examples() {
  const text = useLens((s) => s.text);
  const loadExample = useLens((s) => s.loadExample);
  const current = EXAMPLES.find((e) => e.text === text);
  return (
    <section className="card">
      <div className="card__head">
        <h2 className="card__title">Try an example</h2>
        <span className="card__hint">each loads a text, and the tokens worth following in it</span>
      </div>
      <div className="card__body">
        <div className="pillrow" role="group" aria-label="Examples">
          {EXAMPLES.map((e) => (
            <button
              key={e.id}
              type="button"
              className="pill pill--wide pill--sans"
              aria-pressed={current?.id === e.id}
              onClick={() => loadExample(e)}
            >
              {e.title}
            </button>
          ))}
        </div>
        {current && (
          <p className="notice">
            <strong>What to notice.</strong> {current.notice}
          </p>
        )}
      </div>
    </section>
  );
}

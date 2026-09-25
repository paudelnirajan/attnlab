import { useTokLab } from "../store";

/** The shared text for Inspect, Compare and BPE: typing in one view carries over to the others. */
export function TextBox({ label = "Text" }: { label?: string }) {
  const text = useTokLab((s) => s.text);
  const setText = useTokLab((s) => s.setText);
  return (
    <div className="field">
      <label className="label" htmlFor="toklab-text">
        {label}
        <span className="label__hint">{text.length.toLocaleString()} code points</span>
      </label>
      <textarea
        id="toklab-text"
        className="textarea"
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={3}
        spellCheck={false}
      />
    </div>
  );
}

import { forwardRef } from "react";
import { useStore } from "../state/store";

export const PromptBox = forwardRef<HTMLTextAreaElement>(function PromptBox(_props, ref) {
  const prompt = useStore((s) => s.prompt);
  const setPrompt = useStore((s) => s.setPrompt);
  const tokenizeResult = useStore((s) => s.tokenizeResult);

  const nTokens = tokenizeResult?.n_tokens ?? 0;
  const maxSeq = tokenizeResult?.max_seq;
  const overLimit = maxSeq !== undefined && nTokens > maxSeq;

  return (
    <div className="field">
      <label className="label" htmlFor="prompt-box">
        Prompt
        <span className="label__hint">
          press <kbd>/</kbd> to focus
        </span>
      </label>
      <textarea
        id="prompt-box"
        ref={ref}
        className={overLimit ? "textarea is-invalid" : "textarea"}
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        rows={3}
        spellCheck={false}
        aria-invalid={overLimit}
        aria-describedby="prompt-count"
      />
      <p
        id="prompt-count"
        className={overLimit ? undefined : "muted"}
        style={{ fontSize: "var(--t-sm)", color: overLimit ? "var(--status-critical)" : undefined }}
      >
        {tokenizeResult ? (
          <>
            <span className="mono">{nTokens}</span> token{nTokens === 1 ? "" : "s"}
            {maxSeq !== undefined && (
              <>
                {" "}
                of <span className="mono">{maxSeq}</span> max
              </>
            )}
            {overLimit && <> — over this model's context window, shorten the prompt to run it</>}
          </>
        ) : (
          <>counting tokens…</>
        )}
      </p>
    </div>
  );
});

import { Colorbar } from "./Colorbar";
import { Modal } from "./Modal";

/** The "what am I actually looking at" panel. Attention plots are full of
 * conventions that are obvious once known and opaque until then — which way
 * round the axes go, why half the square is blank, why the colour scale is
 * non-linear. Stating them costs one dialog. */
export function AboutModal({ onClose }: { onClose: () => void }) {
  return (
    <Modal title="How to read this" onClose={onClose} wide>
      <div className="modal__section">
        <h3>The matrix</h3>
        <p>
          Each square is one attention head. Row <strong>i</strong>, column <strong>j</strong> is how much the token at
          position <strong>i</strong> (the <strong>destination</strong>, the position doing the looking) attends to the
          token at position <strong>j</strong> (the <strong>source</strong>, the position being looked at).
        </p>
        <p>
          The top-right half is blank because these are decoder-only models: a position can only attend to itself and
          what came before it. Those cells aren't zero, they're <em>impossible</em>, so they're painted in the page
          background rather than in the colour ramp.
        </p>
      </div>

      <div className="modal__section">
        <h3>Direction</h3>
        <p>
          The toggle above the heads picks which way you read the same matrix when you hover a token.
        </p>
        <p>
          <strong>Destination → Source</strong> reads a <em>row</em>: "when the model is at this token, what does it
          look back at?" A row is a softmax, so those weights sum to 1.
        </p>
        <p>
          <strong>Source → Destination</strong> reads a <em>column</em>: "which later tokens look back at this one?"
          A column is attention <em>received</em> — it is not a distribution and does not sum to 1. A token every
          later position leans on can easily total far more than 1.
        </p>
      </div>

      <div className="modal__section">
        <h3>The colour scale</h3>
        <Colorbar />
        <p style={{ marginTop: 8 }}>
          Weights are sent to your browser as one byte per cell, after a square-root transform. That transform is why
          the tick marks are unevenly spaced: it spends more of the 256 available levels on small weights, so the long
          tail stays visible instead of collapsing to a single flat colour. Full detail in{" "}
          <code>docs/01-wire-format.md</code>.
        </p>
      </div>

      <div className="modal__section">
        <h3>Reproducibility</h3>
        <p>
          Models run in float32 through TransformerLens with the same version ARENA 3.0 pins, so the numbers here
          should match what you get in the 1.2 notebook. The badge in the top bar shows which device produced the
          current run.
        </p>
      </div>
    </Modal>
  );
}

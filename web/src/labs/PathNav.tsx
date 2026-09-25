import { LABS, labById } from "./registry";

/**
 * "You are on step N. Before this: …  Next: …" — the footer every lab carries,
 * so each page is a stop on the path rather than a separate tool. A planned
 * lab is shown but not linked: the path says where it's going.
 */
export function PathNav({ labId }: { labId: string }) {
  const lab = labById(labId);
  const prev = LABS.find((l) => l.step === lab.step - 1);
  const next = LABS.find((l) => l.step === lab.step + 1);

  return (
    <nav className="pathnav" aria-label="Learning path">
      <div className="pathnav__side">
        {prev && (
          <>
            <span className="pathnav__dir">← Step {prev.step}</span>
            {prev.status === "live" ? (
              <a href={prev.path}>{prev.title}</a>
            ) : (
              <span className="muted">{prev.title} (coming)</span>
            )}
          </>
        )}
      </div>
      <a className="pathnav__home" href="/">
        All labs · step {lab.step} of {LABS.length}
      </a>
      <div className="pathnav__side pathnav__side--next">
        {next && (
          <>
            <span className="pathnav__dir">Step {next.step} →</span>
            {next.status === "live" ? (
              <a href={next.path}>{next.title}</a>
            ) : (
              <span className="muted">{next.title} (coming)</span>
            )}
          </>
        )}
      </div>
    </nav>
  );
}

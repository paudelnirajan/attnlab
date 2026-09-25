import { TopBar } from "../../components/TopBar";
import { LABS } from "../registry";

/**
 * The path overview. One card per lab in the order they're meant to be
 * walked; each lab's own footer (PathNav) links back here and on to the next.
 */
export function Home() {
  return (
    <>
      <TopBar />
      <main className="page">
        <header className="labhead">
          <h1 className="labhead__title">Interpretability, one lab at a time</h1>
          <p className="labhead__lede">
            Each lab answers one question about what happens inside a transformer, on your own text and a real model.
            They build on each other: the tokens from step 1 are the positions every later lab is about.
          </p>
        </header>

        <ol className="pathcards">
          {LABS.map((lab) => {
            const live = lab.status === "live";
            const body = (
              <>
                <div className="pathcard__top">
                  <span className="pathcard__step">Step {lab.step}</span>
                  {!live && <span className="badge">coming</span>}
                </div>
                <h2 className="pathcard__title">{lab.title}</h2>
                <p className="pathcard__q">{lab.question}</p>
                <p className="pathcard__summary">{lab.summary}</p>
                <ul className="pathcard__learn">
                  {lab.learn.map((x) => (
                    <li key={x}>{x}</li>
                  ))}
                </ul>
                {live && <span className="pathcard__go">Open lab →</span>}
              </>
            );
            return (
              <li key={lab.id} className={live ? "pathcard" : "pathcard pathcard--planned"}>
                {live ? (
                  <a className="pathcard__link" href={lab.path}>
                    {body}
                  </a>
                ) : (
                  <div className="pathcard__link">{body}</div>
                )}
              </li>
            );
          })}
        </ol>
      </main>
    </>
  );
}

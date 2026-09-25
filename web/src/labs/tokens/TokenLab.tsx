import { useEffect } from "react";
import { TopBar } from "../../components/TopBar";
import { PathNav } from "../PathNav";
import { labById } from "../registry";
import { toklab } from "./api";
import { errText } from "./hooks";
import { useTokLab, writeTokLabUrl, type View } from "./store";
import { BpeView } from "./views/BpeView";
import { CompareView } from "./views/CompareView";
import { InspectView } from "./views/InspectView";
import { LanguagesView } from "./views/LanguagesView";
import { VocabView } from "./views/VocabView";

const TABS: { id: View; label: string; question: string }[] = [
  { id: "inspect", label: "Inspect", question: "How is this text cut, and why?" },
  { id: "compare", label: "Compare", question: "How do different tokenizers cut the same text?" },
  { id: "languages", label: "Languages", question: "What does the same meaning cost in each language?" },
  { id: "bpe", label: "BPE step-through", question: "How is a token built, one merge at a time?" },
  { id: "vocab", label: "Vocabulary", question: "What is in the vocabulary, and who was it built for?" },
];

/**
 * Step 1 of the path. Everything here runs on tokenizers alone — no model is
 * loaded — which is why it can offer tokenizers (GPT-4o's, Qwen's) whose
 * models this app could never host.
 */
export function TokenLab() {
  const lab = labById("tokens");
  const view = useTokLab((s) => s.view);
  const setView = useTokLab((s) => s.setView);
  const tokenizer = useTokLab((s) => s.tokenizer);
  const text = useTokLab((s) => s.text);
  const compare = useTokLab((s) => s.compare);
  const special = useTokLab((s) => s.special);
  const word = useTokLab((s) => s.word);
  const query = useTokLab((s) => s.query);
  const catalogError = useTokLab((s) => s.catalogError);
  const setTokenizers = useTokLab((s) => s.setTokenizers);
  const setCatalogError = useTokLab((s) => s.setCatalogError);

  useEffect(() => {
    toklab
      .tokenizers()
      .then((r) => setTokenizers(r.tokenizers))
      .catch((e) => setCatalogError(errText(e)));
  }, [setTokenizers, setCatalogError]);

  useEffect(() => {
    writeTokLabUrl({ view, tokenizer, text, compare, special, word, query });
  }, [view, tokenizer, text, compare, special, word, query]);

  const tab = TABS.find((t) => t.id === view)!;

  return (
    <>
      <TopBar />
      <main className="page">
        <header className="labhead">
          <p className="labhead__step">Step {lab.step}</p>
          <h1 className="labhead__title">{lab.title}</h1>
          <p className="labhead__lede">{lab.summary}</p>
        </header>

        {catalogError && (
          <div className="banner banner--error" role="alert">
            <span>Couldn't load the tokenizer list: {catalogError}</span>
          </div>
        )}

        <div className="tabs" role="tablist" aria-label="Tokenizer lab views">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              role="tab"
              id={`tab-${t.id}`}
              aria-selected={view === t.id}
              aria-controls="toklab-panel"
              className="tabs__tab"
              onClick={() => setView(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
        <p className="tabs__question">{tab.question}</p>

        <div id="toklab-panel" role="tabpanel" aria-labelledby={`tab-${view}`}>
          {view === "inspect" && <InspectView />}
          {view === "compare" && <CompareView />}
          {view === "languages" && <LanguagesView />}
          {view === "bpe" && <BpeView />}
          {view === "vocab" && <VocabView />}
        </div>

        <PathNav labId={lab.id} />
      </main>
    </>
  );
}

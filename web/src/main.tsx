import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { Home } from "./labs/home/Home";
import { LensLab } from "./labs/lens/LensLab";
import { currentLabId, LABS } from "./labs/registry";
import { TokenLab } from "./labs/tokens/TokenLab";
import "./styles.css";

// Attention-lab permalinks from before the labs existed live at "/?model=…&prompt=…".
// Keep them working by moving them to their new home before anything reads the URL.
if (window.location.pathname === "/" && /[?&](model|prompt)=/.test(window.location.search)) {
  window.history.replaceState(null, "", `/attention${window.location.search}`);
}

// Each lab is a separate page to search engines, so give it its own title,
// description and canonical URL (index.html holds the home page's).
const lab = LABS.find((l) => l.id === currentLabId() && l.status === "live");
if (lab) {
  document.title = `${lab.title}: ${lab.question} — attnlab`;
  document.querySelector('meta[name="description"]')?.setAttribute("content", lab.summary);
  document.querySelector('link[rel="canonical"]')?.setAttribute("href", `https://attnlab.sarvabhaum.ai${lab.path}`);
}

function Root() {
  switch (currentLabId()) {
    case "tokens":
      return <TokenLab />;
    case "attention":
      return <App />;
    case "logit-lens":
      return <LensLab />;
    default:
      return <Home />;
  }
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>,
);

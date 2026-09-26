import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { Home } from "./labs/home/Home";
import { LensLab } from "./labs/lens/LensLab";
import { currentLabId } from "./labs/registry";
import { TokenLab } from "./labs/tokens/TokenLab";
import "./styles.css";

// Attention-lab permalinks from before the labs existed live at "/?model=…&prompt=…".
// Keep them working by moving them to their new home before anything reads the URL.
if (window.location.pathname === "/" && /[?&](model|prompt)=/.test(window.location.search)) {
  window.history.replaceState(null, "", `/attention${window.location.search}`);
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

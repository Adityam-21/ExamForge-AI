import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "katex/dist/katex.min.css";
import "./styles/tokens.css";
import "./styles/app.css";

import App from "./App.jsx";
import { AppProvider } from "./state/AppContext.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <AppProvider>
      <App />
    </AppProvider>
  </StrictMode>
);

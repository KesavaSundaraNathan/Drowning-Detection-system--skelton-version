import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
import { makeQueryClient } from "./queryClient";
import { migrateLegacyStorage } from "./state/legacyStorage";
import { applyTheme } from "./state/theme";
import "./styles/index.css";

migrateLegacyStorage(); // AquaGuard build -> Life Guard: carry over aquaguard.* keys once, before anything reads its own
applyTheme(); // before the first paint: no light flash for a dark-theme user

const root = document.getElementById("root");
if (root === null) {
  throw new Error("#root element missing from index.html");
}

createRoot(root).render(
  <StrictMode>
    <App client={makeQueryClient()} />
  </StrictMode>,
);

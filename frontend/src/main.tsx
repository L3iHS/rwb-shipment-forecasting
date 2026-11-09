import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "@fontsource-variable/sofia-sans/wght.css";
import "@fontsource-variable/sofia-sans-condensed/wght.css";
import "@fontsource-variable/sofia-sans-extra-condensed/wght.css";
import "@fontsource-variable/martian-mono/wdth.css";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

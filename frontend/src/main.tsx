import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "@fontsource-variable/onest/wght.css";
import "@fontsource/yeseva-one/400.css";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

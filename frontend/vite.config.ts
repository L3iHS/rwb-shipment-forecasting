import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The console is served from the domain root, including SPA fallback URLs.
export default defineConfig({
  base: "/",
  plugins: [react(), tailwindcss()],
  server: { proxy: { "/api": "http://localhost:8000" } },
  // Pre-bundle up front: a mid-session re-optimisation loads a second copy of React.
  optimizeDeps: { include: ["react", "react-dom", "react-dom/client", "motion/react", "lucide-react"] },
});

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Relative base so the build works from any path on any host.
export default defineConfig({
  base: "./",
  plugins: [react(), tailwindcss()],
  server: { proxy: { "/api": "http://localhost:8000" } },
  // Pre-bundle up front: a mid-session re-optimisation loads a second copy of React.
  optimizeDeps: { include: ["react", "react-dom", "react-dom/client", "motion/react", "lucide-react"] },
});

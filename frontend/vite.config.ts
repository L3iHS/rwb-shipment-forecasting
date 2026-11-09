import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The dispatcher is served at the root of its own domain.
export default defineConfig({
  base: "/",
  publicDir: false,
  plugins: [react(), tailwindcss()],
  server: { proxy: { "/api": "http://localhost:8000" } },
  // Pre-bundle up front: a mid-session re-optimisation loads a second copy of React.
  optimizeDeps: { include: ["react", "react-dom", "react-dom/client", "recharts", "motion/react", "lucide-react"] },
});

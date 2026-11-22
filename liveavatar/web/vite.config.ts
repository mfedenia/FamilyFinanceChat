import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

// The web app and the orchestrator are one origin from the browser's point of
// view: in dev this proxy forwards /api and /ws to the server (:8787), in
// production the server serves web/dist itself.
export default defineConfig({
  // Two pages: the tutor (index.html) and the two-avatar duo (duo.html).
  build: {
    rollupOptions: {
      input: {
        main: fileURLToPath(new URL("index.html", import.meta.url)),
        duo: fileURLToPath(new URL("duo.html", import.meta.url)),
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8787",
      "/ws": { target: "ws://localhost:8787", ws: true },
    },
  },
});

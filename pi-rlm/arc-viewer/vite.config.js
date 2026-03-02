import { defineConfig } from "vite";

export default defineConfig({
  root: import.meta.dirname,
  server: {
    proxy: {
      "/api": "http://localhost:3334",
      "/ws": { target: "ws://localhost:3334", ws: true },
    },
  },
});

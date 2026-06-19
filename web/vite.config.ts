import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Static SPA; point-cloud data + result figures live in /public/data.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: { outDir: "dist", assetsInlineLimit: 0 },
});

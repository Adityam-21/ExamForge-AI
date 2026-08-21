import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: false,
  },
  build: {
    outDir: "dist",
    sourcemap: false,
    rollupOptions: {
      output: {
        // Markdown + math rendering is the heaviest part of the bundle.
        // Splitting it keeps the initial shell small.
        manualChunks(id) {
          if (!id.includes("node_modules")) return undefined;
          if (/[\\/](katex|hast|mdast|micromark|unified|unist|vfile|remark|rehype|react-markdown|property-information|space-separated-tokens|comma-separated-tokens|character-entities|decode-named-character-reference|zwitch|longest-streak|ccount|escape-string-regexp|markdown-table|trim-lines|html-void-elements|bail|is-plain-obj|trough|extend|devlop|estree|style-to-|web-namespaces)/.test(id)) {
            return "markdown";
          }
          return undefined;
        },
      },
    },
  },
});

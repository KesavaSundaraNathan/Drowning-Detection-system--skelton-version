import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import globals from "globals";
import { defineConfig } from "eslint/config";
import tseslint from "typescript-eslint";

export default defineConfig(
  { ignores: ["dist", "coverage", "node_modules"] },
  {
    files: ["**/*.{ts,tsx}"],
    extends: [js.configs.recommended, ...tseslint.configs.strictTypeChecked],
    languageOptions: {
      ecmaVersion: 2022,
      globals: globals.browser,
      parserOptions: { projectService: true, tsconfigRootDir: import.meta.dirname },
    },
    plugins: { "react-hooks": reactHooks, "react-refresh": reactRefresh },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "react-refresh/only-export-components": ["error", { allowConstantExport: true }],
      "no-console": "error",
    },
  },
  {
    // test helpers export render utilities next to components; fast refresh is irrelevant there
    files: ["src/**/*.test.{ts,tsx}", "src/test/**"],
    rules: { "react-refresh/only-export-components": "off" },
  },
  {
    // Playwright specs and config run in Node, not the browser
    files: ["e2e/**/*.ts", "playwright.config.ts"],
    languageOptions: { globals: { ...globals.node, ...globals.browser } },
    rules: { "react-refresh/only-export-components": "off" },
  },
  {
    files: ["eslint.config.js", "scripts/**/*.mjs"],
    extends: [js.configs.recommended],
    languageOptions: { globals: globals.node, ecmaVersion: 2024, sourceType: "module" },
  },
);

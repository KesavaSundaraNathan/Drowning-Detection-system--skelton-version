// Initial-load budget (plan Phase 6): the entry chunk plus everything it imports STATICALLY (JS + CSS)
// must be <= 250 KB gzipped. Lazy routes (dynamic imports) are excluded - they load on navigation.
// Reads Vite's build manifest (vite.config.ts: build.manifest = true). Exit 1 over budget.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { gzipSync } from "node:zlib";

const BUDGET = 250 * 1024;
const dist = join(import.meta.dirname, "..", "dist");
const manifest = JSON.parse(readFileSync(join(dist, ".vite", "manifest.json"), "utf8"));
const entries = Object.entries(manifest).filter(([, c]) => c.isEntry);
if (entries.length !== 1) {
  console.error(`expected exactly one entry chunk, found ${entries.length}`);
  process.exit(1);
}
const seen = new Set();
const files = [];
const visit = (key) => {
  if (seen.has(key)) return;
  seen.add(key);
  const chunk = manifest[key];
  files.push(chunk.file, ...(chunk.css ?? []));
  for (const imp of chunk.imports ?? []) visit(imp);
};
visit(entries[0][0]);
let total = 0;
for (const f of new Set(files)) {
  const gz = gzipSync(readFileSync(join(dist, f))).length;
  total += gz;
  console.log(`${(gz / 1024).toFixed(1).padStart(7)} KB gz  ${f}`);
}
console.log(`initial load: ${(total / 1024).toFixed(1)} KB gzipped (budget ${BUDGET / 1024} KB)`);
if (total > BUDGET) {
  console.error("OVER BUDGET");
  process.exit(1);
}

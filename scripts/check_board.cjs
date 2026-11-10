// Runtime smoke check using a minimal DOM. This does not replace visual browser QA.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const filename = process.argv[2] || path.join(__dirname, '../outputs/dispatch_board.html');
const html = fs.readFileSync(filename, 'utf8');
const scripts = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1]);
assert(scripts.length > 0, 'No script in board');
const nodes = new Map();
const document = {
  getElementById(id) {
    if (!nodes.has(id)) nodes.set(id, { innerHTML: '', textContent: '', addEventListener() {}, setAttribute() {} });
    return nodes.get(id);
  },
  querySelectorAll() { return []; }
};
const context = vm.createContext({ document, console });
for (const script of scripts) vm.runInContext(script, context, { timeout: 10000 });
vm.runInContext('select(R[R.length - 1].id)', context);
for (const [id, node] of nodes) {
  assert(!/NaN|undefined/.test(node.innerHTML + node.textContent), `Invalid rendered value in ${id}`);
}
// Missing calibration on fallback routes must not be rendered as a zero-valued band.
vm.runInContext('R[0].stale = true; R[0].p10.fill(null); R[0].p50.fill(null); R[0].p90.fill(null); R[0].c90.fill(null); select(R[0].id)', context);
for (const [id, node] of nodes) assert(!/NaN|undefined/.test(node.innerHTML + node.textContent), id);
console.log('Board script and route rendering passed; no visual layout assertion.');

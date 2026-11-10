import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

// plan.ts has no runtime imports, so it runs on its own once transpiled.
const source = await readFile(new URL("../src/plan.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
});
const { clock, spread } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);
const board = JSON.parse(await readFile(new URL("../public/board.json", import.meta.url), "utf8"));

test("the forecast steps go forward from the moment of the forecast in half-hours", () => {
  const now = new Date(board.now).getTime();
  assert.equal(clock(board, 0).getTime(), now);
  assert.equal(clock(board, 8).getTime() - now, 8 * board.step_min * 60000);
});

test("the band width is the 80% range relative to the forecast over the horizon", () => {
  const r = { p10: [5, 5], p50: [10, 10], p90: [15, 25] };
  assert.equal(spread(r), (10 + 20) / 20);
  assert.equal(spread({ p10: [0], p50: [0], p90: [0] }), 0);
});

test("every route in the bundled board has a forecast, a band and its error on the check week", () => {
  assert.equal(board.routes.length, 1000);
  for (const r of board.routes) {
    assert.equal(r.p50.length, 8);
    assert.ok(spread(r) >= 0);
    assert.equal(typeof r.wape, "number");
    assert.equal(typeof r.bias, "number");
  }
});

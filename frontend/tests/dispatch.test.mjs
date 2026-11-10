import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

const source = await readFile(new URL("../src/plan.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
});
const { calledIdle, callStep, fillStep, leadSteps, trucksDue } = await import(
  `data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`
);
const board = JSON.parse(await readFile(new URL("../public/board.json", import.meta.url), "utf8"));
const route = (overrides = {}) => ({
  ...board.routes[0], cap: 100, bl: 0, way: 0, dock: 0,
  cr: Array(8).fill(0), c50: Array(8).fill(0), c90: Array(8).fill(0), ...overrides,
});

test("a bulk call covers every full waiting load, including more than ten trucks", () => {
  const r = route({ bl: 12.2 });
  assert.equal(trucksDue(r, 2, "p50"), 12);
  assert.notEqual(callStep({ ...r, way: 12 }, 2, "p50"), 0);
  assert.equal(callStep({ ...r, way: 11 }, 2, "p50"), 0);
});

test("bulk quantities clear urgent calls for every route and scenario in the bundled snapshot", () => {
  for (const leadH of [0.5, 1, 2]) for (const risk of ["p50", "p90"]) {
    const lead = leadSteps(board, leadH);
    for (const r of board.routes) {
      const n = trucksDue(r, lead, risk);
      assert.ok(Number.isInteger(n) && n >= 0);
      if (callStep(r, lead, risk) === 0) {
        assert.ok(n > 0);
        assert.notEqual(callStep({ ...r, way: r.way + n }, lead, risk), 0, `route ${r.id}`);
        assert.equal(callStep({ ...r, way: r.way + n - 1 }, lead, risk), 0, `route ${r.id}`);
      } else assert.equal(n, 0);
    }
  }
});

test("early-call cost is the wait between arrival and a complete load", () => {
  const r = route({ bl: 0.2, way: 1, cr: [10, 20, 30, 50, 80, 110, 140, 170] });
  assert.equal(fillStep(r, 1, "p50"), 5);
  assert.deepEqual(calledIdle(r, 1, 2, "p50"), { steps: 3, open: false, early: 1, lastFill: 5 });
});

test("loads outside the forecast produce a lower bound, not a promised loading time", () => {
  const r = route({ bl: 0.2, way: 2, cr: [10, 20, 30, 50, 80, 110, 140, 170] });
  assert.deepEqual(calledIdle(r, 1, 2, "p50"), { steps: 6, open: true, early: 1, lastFill: null });
});

test("trucks called for already waiting loads incur no forecast idle time", () => {
  assert.deepEqual(calledIdle(route({ bl: 2.5, way: 2 }), 2, 2, "p50"), {
    steps: 0, open: false, early: 0, lastFill: 0,
  });
});

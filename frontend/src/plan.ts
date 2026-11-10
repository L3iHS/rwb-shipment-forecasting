import type { Board, Route } from "./data";

export type Risk = "p50" | "p90";

export interface Scenario {
  leadH: number; // truck lead time, hours
  risk: Risk; // p50: expected inflow, p90: cautious inflow
}

/** Clock time of step k from the forecast origin. */
export function clock(board: Board, k: number): Date {
  return new Date(new Date(board.now).getTime() + k * board.step_min * 60000);
}

export const leadSteps = (board: Board, leadH: number) => Math.round((leadH * 60) / board.step_min);

/**
 * Steps until the next truck should be called (0 = now, -1 = not within the horizon).
 * Same rule as rwb.dispatch.call_plan: the next truck is needed once backlog + inflow covers one more
 * load than the trucks already coming; it is called `lead` steps before that moment.
 */
export function callStep(r: Route, lead: number, risk: Risk): number {
  const inflow = risk === "p90" ? r.c90 : r.cr;
  const backlog = r.bl * r.cap;
  const need = r.cap * (1 + r.way + r.dock);
  if (backlog >= need) return 0;
  for (let k = 0; k <= inflow.length - lead; k++) {
    if (backlog + inflow[k + lead - 1] >= need) return k;
  }
  return -1;
}

/** Truckloads at the dock after k steps (k = 0..7 over the horizon). */
export const loadsAt = (r: Route, k: number, risk: Risk) => r.bl + (risk === "p90" ? r.c90 : r.c50)[k] / r.cap;

export const inbound = (r: Route) => r.way + r.dock;
/** Truckloads at the dock that no called truck will take: the cargo that is actually waiting for a decision. */
export const uncovered = (r: Route) => Math.max(0, r.bl - inbound(r));

/** Trucks to call right now so the route stops being urgent; 0 when it is not time to call yet. */
export function trucksDue(r: Route, lead: number, risk: Risk): number {
  if (callStep(r, lead, risk) !== 0) return 0;
  const inflow = risk === "p90" ? r.c90 : r.cr;
  const atArrival = r.bl + inflow[lead - 1] / r.cap;
  const limit = Math.max(1, Math.ceil(Math.max(r.bl, atArrival)) - inbound(r));
  let n = 1;
  while (n < limit && callStep({ ...r, way: r.way + n }, lead, risk) === 0) n++;
  return n;
}

/** Step at which the dock holds `level` full truckloads (0 = already), or null if not within the horizon. */
export function fillStep(r: Route, level: number, risk: Risk): number | null {
  if (r.bl >= level) return 0;
  const k = (risk === "p90" ? r.c90 : r.cr).findIndex((c) => r.bl + c / r.cap >= level);
  return k < 0 ? null : k + 1;
}

export interface Idle {
  steps: number; // steps the trucks called here stand at the dock waiting for their load, in total
  open: boolean; // at least one of them is not loaded within the horizon, so the real figure is larger
  early: number; // how many of them wait at all
  lastFill: number | null; // step at which the last one gets its load; null = beyond the horizon
}

/**
 * The price of calling ahead of the rule. A truck called now arrives after `lead` steps; it stands idle until the dock
 * holds a full load for it. `r` already counts the `calledHere` trucks in `way`. A call the rule asked for costs nothing.
 */
export function calledIdle(r: Route, calledHere: number, lead: number, risk: Risk): Idle {
  const base = inbound(r) - calledHere, horizon = r.cr.length;
  const idle: Idle = { steps: 0, open: false, early: 0, lastFill: 0 };
  for (let i = 1; i <= calledHere; i++) {
    const fill = fillStep(r, base + i, risk);
    const wait = fill == null ? horizon - lead : fill - lead;
    idle.lastFill = fill;
    if (fill == null) idle.open = true;
    if (wait > 0) {
      idle.steps += wait;
      idle.early++;
    }
  }
  return idle;
}
export const waitsWithoutTruck = (r: Route) => r.bl >= 1 && inbound(r) === 0;

/** Relative width of the 80% band: how sure the forecast is for this route. */
export const spread = (r: Route) =>
  r.p90.reduce((s, v, k) => s + v - r.p10[k], 0) / Math.max(1, r.p50.reduce((a, b) => a + b, 0));

/** Spread at the 1/3 and 2/3 quantiles over all routes: above the second cut the forecast is unsure. */
export function confidenceCuts(routes: Route[]): [number, number] {
  const s = routes.map(spread).sort((a, b) => a - b);
  return [s[Math.floor(s.length / 3)], s[Math.floor((2 * s.length) / 3)]];
}

/** Step (1..8) at which the network forecast peaks. */
export function peakStep(board: Board): number {
  const p = board.network.p50;
  return p.indexOf(Math.max(...p)) + 1;
}

export interface Kpis {
  callNow: number;
  callHour: number;
  fleet: number; // new trucks needed by the end of the horizon
  fleetCautious: number;
  vol4: number;
  vol4Cautious: number;
  waiting: number;
}

export function kpis(routes: Route[], calls: number[]): Kpis {
  const last = routes[0].c50.length - 1;
  let callNow = 0, callHour = 0, fleet = 0, fleetCautious = 0, vol4 = 0, vol4Cautious = 0, waiting = 0;
  routes.forEach((r, i) => {
    if (calls[i] === 0) callNow++;
    else if (calls[i] > 0 && calls[i] <= 2) callHour++;
    fleet += Math.max(0, Math.floor(loadsAt(r, last, "p50")) - inbound(r));
    fleetCautious += Math.max(0, Math.floor(loadsAt(r, last, "p90")) - inbound(r));
    vol4 += r.c50[last];
    vol4Cautious += r.c90[last];
    if (waitsWithoutTruck(r)) waiting++;
  });
  return { callNow, callHour, fleet, fleetCautious, vol4, vol4Cautious, waiting };
}

// Assistant tools over board.json. The backend (/api/agent) must return the same numbers; see docs/web.md.
import type { Board, Route } from "./data";
import { hhmm } from "./format";
import { callStep, clock, confidenceCuts, inbound, kpis, leadSteps, peakStep, spread, waitsWithoutTruck, type Risk } from "./plan";

export type Action =
  | { type: "open_route"; id: number }
  | { type: "filter"; value: "now" | "hour" | "jam" | "all" }
  | { type: "scenario"; lead_h: number; risk: Risk };

export const plan = (board: Board, leadH: number, risk: Risk) =>
  board.routes.map((r) => callStep(r, leadSteps(board, leadH), risk));

export function routeStatus(board: Board, id: number, leadH: number, risk: Risk) {
  const r = board.routes.find((q) => q.id === id);
  if (!r) return null;
  const call = callStep(r, leadSteps(board, leadH), risk);
  return {
    route_id: r.id,
    call_at: call < 0 ? null : hhmm(clock(board, call)),
    call_now: call === 0,
    backlog_loads: r.bl,
    trucks_called: inbound(r),
    capacity: r.cap,
    forecast_4h: r.c50[7],
    forecast_4h_p90: r.c90[7],
  };
}

type Condition = "call_now" | "call_within_hour" | "waiting_without_truck" | "low_confidence";

export function findRoutes(board: Board, calls: number[], condition: Condition, limit = 10) {
  const unsure = confidenceCuts(board.routes)[1];
  const withinHour = (call: number) => call >= 0 && call <= 2;
  const test: Record<Condition, (r: Route, call: number) => boolean> = {
    call_now: (_, call) => call === 0,
    call_within_hour: (_, call) => withinHour(call),
    waiting_without_truck: waitsWithoutTruck,
    low_confidence: (r, call) => withinHour(call) && spread(r) > unsure,
  };
  const hit = board.routes
    .map((r, i) => ({ r, call: calls[i] }))
    .filter(({ r, call }) => test[condition](r, call))
    .sort((a, b) => b.r.bl - a.r.bl);
  return { count: hit.length, routes: hit.slice(0, limit).map(({ r, call }) => ({ route_id: r.id, backlog_loads: r.bl, call_in_steps: call })) };
}

export function networkSummary(board: Board, calls: number[]) {
  const k = kpis(board.routes, calls);
  const peak = peakStep(board);
  return { call_now: k.callNow, call_within_hour: k.callHour, new_trucks: k.fleet, new_trucks_cautious: k.fleetCautious,
    volume_4h: k.vol4, waiting_without_truck: k.waiting, peak_at: hhmm(clock(board, peak)), peak_volume: board.network.p50[peak - 1] };
}

export function whatIf(board: Board, leadH: number, risk: Risk) {
  const base = plan(board, board.model.lead_h, "p50");
  const scenario = plan(board, leadH, risk);
  return {
    lead_h: leadH, risk,
    call_now: scenario.filter((c) => c === 0).length,
    call_now_base: base.filter((c) => c === 0).length,
    changed: scenario.filter((c, i) => c !== base[i]).length,
  };
}

export function evidence(board: Board) {
  const row = board.backtest.sensitivity.find((s) => s.lead_h === board.model.lead_h && s.fill_hours === board.model.fill_hours)!;
  return { loss_per_truck_min: { reactive: row.reactive, last_2h_pace: row.last_2h_pace, same_time_last_week: row.same_time_last_week, model: row.model_p50 },
    coverage: board.backtest.intervals, week: board.backtest.week };
}

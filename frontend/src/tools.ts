// Assistant tools over board.json. The backend (/api/agent, backend/tools.py) returns the same numbers; see docs/web.md.
import type { Board, Route } from "./data";
import { hhmm } from "./format";
import { clock } from "./plan";

export type Action = { type: "open_route"; id: number };

/** Half-width of the 80% band relative to the forecast over the horizon: 0.44 means ±44%. */
export const band = (p10: number[], p50: number[], p90: number[]) =>
  p90.reduce((s, v, k) => s + v - p10[k], 0) / 2 / Math.max(1, p50.reduce((a, b) => a + b, 0));

const peak = (p50: number[]) => p50.indexOf(Math.max(...p50));
const last = (a: number[]) => a[a.length - 1];

export function routeForecast(board: Board, id: number) {
  const r = board.routes.find((q) => q.id === id);
  if (!r) return null;
  const k = peak(r.p50);
  return {
    route_id: r.id,
    volume_4h: last(r.c50),
    peak_at: hhmm(clock(board, k + 1)),
    peak_per_hour: r.p50[k],
    band_80: band(r.p10, r.p50, r.p90),
    error_wape: r.wape,
    bias: r.bias,
  };
}

export type Order = "largest" | "least_accurate" | "most_accurate" | "widest_band";

export function findRoutes(board: Board, order: Order, limit = 10) {
  const key: Record<Order, (r: Route) => number> = {
    largest: (r) => -last(r.c50),
    least_accurate: (r) => -(r.wape ?? 0),
    most_accurate: (r) => r.wape ?? 0,
    widest_band: (r) => -band(r.p10, r.p50, r.p90),
  };
  const top = [...board.routes].sort((a, b) => key[order](a) - key[order](b) || a.id - b.id).slice(0, limit);
  return { order, routes: top.map((r) => ({ route_id: r.id, volume_4h: last(r.c50), error_wape: r.wape, band_80: band(r.p10, r.p50, r.p90) })) };
}

export function networkForecast(board: Board) {
  const n = board.network;
  const k = peak(n.p50);
  return {
    volume_4h: board.routes.reduce((s, r) => s + last(r.c50), 0),
    horizon_end: hhmm(clock(board, n.p50.length)),
    peak_at: hhmm(clock(board, k + 1)),
    peak_per_hour: n.p50[k],
    band_80: band(n.lo, n.p50, n.hi),
  };
}

export function modelQuality(board: Board) {
  const bt = board.backtest;
  return { week: bt.week, metric_case_from_1030: bt.point_1030 ?? bt.point, band_80_coverage: bt.intervals["80%"], per_day: board.monitor };
}

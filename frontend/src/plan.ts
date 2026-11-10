import type { Board, Route } from "./data";

/** Clock time of step k from the forecast origin. */
export function clock(board: Board, k: number): Date {
  return new Date(new Date(board.now).getTime() + k * board.step_min * 60000);
}

/** Relative width of the 80% band: how sure the forecast is for this route. */
export const spread = (r: Route) =>
  r.p90.reduce((s, v, k) => s + v - r.p10[k], 0) / Math.max(1, r.p50.reduce((a, b) => a + b, 0));

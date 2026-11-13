// Types of board.json written by `python scripts/pipeline.py export` (see docs/web.md).

/** One finding of the file check (src/rwb/checks.py): what was wrong with an uploaded file and what was done before
 * the model read it. fixed: repaired without a guess; filled: values replaced by an estimate; warn: left as it is. */
export interface Check {
  code: string;
  level: "fixed" | "filled" | "warn" | "info";
  text: string;
  rows: number;
}

export interface Route {
  id: number;
  cap: number; // truck capacity, goods
  bl: number; // backlog at the dock, in truckloads
  way: number; // trucks called, on the way
  dock: number; // trucks at the dock waiting for a full load
  call: number; // steps until the next call (0 = now, -1 = not within the horizon)
  h: number[]; // history of target_1h, hist_n points ending at `now`
  p10: number[];
  p50: number[];
  p90: number[]; // forecast of target_1h, 8 steps
  c50: number[];
  c90: number[]; // cumulative inflow to the dock over 1..8 steps
  cr: number[]; // cumulative inflow the call rule uses (calibrated median)
  wape?: number; // the route's error over the validation week
  bias?: number; // the relative bias of its total over that week
}

interface Rule {
  trucks: number;
  idle_h: number;
  late_h: number;
  partial: number;
  fill: number;
  peak_backlog: number;
}

export type RuleKey = "reactive" | "last_2h_pace" | "same_time_last_week" | "model_p50" | "oracle";

export interface Board {
  now: string;
  step_min: number;
  hist_n: number;
  model: {
    trained_until: string;
    lib: string;
    objective: string;
    lead_h: number;
    fill_hours: number;
    max_wait_h: number;
    risk: number;
  };
  network: { hist: number[]; p50: number[]; lo: number[]; hi: number[] };
  routes: Route[];
  checks?: Check[]; // only on a board built from an uploaded file: what its check found
  monitor: { day: string; wape: number; bias: number; cov: number }[];
  alerts: { day: string; signal: string; detail: string; action: string }[];
  backtest: {
    sensitivity: ({ lead_h: number; fill_hours: number } & Record<RuleKey, number>)[];
    per_day: Record<string, Rule>;
    intervals: Record<string, number>;
    point: number;
    point_1030?: number; // the same metric, forecasts from 10:30 only, as in the test
    week: [string, string];
  };
}

export async function loadBoard(): Promise<Board> {
  const res = await fetch(`${import.meta.env.BASE_URL}board.json`);
  if (!res.ok) throw new Error(`board.json: ${res.status}`);
  return res.json();
}

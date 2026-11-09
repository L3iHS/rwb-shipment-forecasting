// Types of board.json written by `python scripts/pipeline.py export` (see docs/web.md).

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
  monitor: { day: string; wape: number; bias: number; cov: number }[];
  alerts: { day: string; signal: string; detail: string; action: string }[];
  backtest: {
    sensitivity: ({ lead_h: number; fill_hours: number } & Record<RuleKey, number>)[];
    per_day: Record<string, Rule>;
    intervals: Record<string, number>;
    point: number;
    week: [string, string];
  };
}

export async function loadBoard(): Promise<Board> {
  const res = await fetch(`${import.meta.env.BASE_URL}board.json`);
  if (!res.ok) throw new Error(`board.json: ${res.status}`);
  return res.json();
}

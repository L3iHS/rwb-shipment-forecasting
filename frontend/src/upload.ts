import type { Board, Route } from "./data";
import { hhmm, plural } from "./format";

// Answer of POST /api/forecast: the service model on an uploaded file against the stored history (src/rwb/upload.py).

/** A set of routes summed: the whole network or the routes the file changed. `b_*` is the forecast without the file. */
export interface Group {
  routes: number;
  hist: number[]; // target_1h with the file, hist_n points up to now
  p50: number[];
  lo: number[];
  hi: number[];
  s13: number[]; // status_1..3: goods through the processing stages of this warehouse, per half-hour
  s46: number[]; // status_4..6: goods coming from the previous warehouse
  vol2h: number; // goods shipped over the next 2 hours
  call_now: number;
  call_hour: number;
  calls: number[]; // routes to call a truck for at step 0..8, the bars of the console's timeline
  b_calls?: number[];
  b_hist?: number[];
  b50?: number[];
  b_lo?: number[];
  b_hi?: number[];
  b_s13?: number[];
  b_s46?: number[];
  b_vol2h?: number;
  b_call_now?: number;
  b_call_hour?: number;
  after?: number[]; // what was actually shipped after "now", when the file is about the past
}

/** A route's forecast without the file, in the order of `routes`. */
export interface BaseRoute {
  bl: number;
  way: number;
  dock: number;
  cr: number[];
  c90: number[];
  p50: number[];
  c50: number[];
}

export interface FileRun {
  now: string;
  step_min: number;
  hist_n: number;
  lead: number; // steps from calling a truck to its arrival
  file: { rows: number; routes: number; from: string; to: string; extended: number };
  baseline: boolean; // the moment is inside the stored history, so there is a forecast to compare with
  dock_from: "board" | "replay"; // cargo at the dock as on the console, or rebuilt from the file's shipments
  changed: number; // routes the file differs from the stored history in
  changed_y: boolean;
  calls: { later: number; earlier: number; held: number };
  all: Group;
  group: Group | null;
  routes: (Route & { after?: number[] })[]; // every route, as in board.json
  base_routes: BaseRoute[] | null;
}

/** Send the file as it is; the server tells CSV from Parquet by its first bytes. */
export async function forecastFile(file: Blob): Promise<FileRun> {
  let res: Response;
  try {
    res = await fetch("/api/forecast", { method: "POST", body: file, headers: { "Content-Type": "application/octet-stream" } });
  } catch {
    throw new Error("Сервер с моделью недоступен");
  }
  if (!res.ok) {
    const detail = await res.json().then((d) => d?.detail, () => null);
    throw new Error(typeof detail === "string" ? detail : res.status === 404 ? "Сервер с моделью недоступен" : `Ошибка сервера: ${res.status}`);
  }
  return res.json();
}

export const DEMOS = [
  {
    key: "zator",
    file: "demo/zator_0900-1030.csv",
    title: "Затор на сортировке",
    note: "С 09:30 встала сортировка, 200 маршрутов.",
  },
  {
    key: "vsplesk",
    file: "demo/vsplesk_0900-1030.csv",
    title: "Всплеск заказов",
    note: "С 09:30 обработка вдвое выше, 200 маршрутов.",
  },
] as const;

/** The file's shift as a board: the console shows it exactly as it shows its own forecast. */
export function fileBoard(source: Board, run: FileRun): Board {
  const a = run.all;
  // The route's error on the check week belongs to the model, not to the file: it comes from the console's board.
  const quality = new Map(source.routes.map((r) => [r.id, { wape: r.wape, bias: r.bias }]));
  const routes = run.routes.map((r) => ({ ...r, ...quality.get(r.id) }));
  return { ...source, now: run.now, hist_n: run.hist_n, network: { hist: a.hist, p50: a.p50, lo: a.lo, hi: a.hi }, routes };
}

/** What the shift chart draws over the file's forecast to show what the file changed. */
export interface Overlay {
  baseFc?: number[]; // network forecast without the file
  baseHist?: number[]; // stored shipments, where the file changed them
  baseRoutes?: BaseRoute[]; // for the outlines of the bars without the file
  mark?: { from: number; label: string; short: string }; // the stretch of history where the file starts to differ
  same?: boolean; // the file repeats the stored history: the forecast is the console's own
}

export function overlayOf(run: FileRun, back: number): Overlay {
  if (!run.baseline || !run.changed) return { same: run.baseline };
  const g = run.group ?? run.all;
  const n = g.s13.length;
  const first = (a?: number[], b?: number[]) => (a && b ? a.findIndex((v, i) => Math.abs(v - b[i]) > 0.5) : -1);
  const hits = [first(g.s13, g.b_s13), first(g.hist, g.b_hist)].filter((i) => i >= 0);
  let mark: Overlay["mark"];
  if (hits.length) {
    const i = Math.min(...hits);
    const since = hhmm(new Date(new Date(run.now).getTime() + (i - (n - 1)) * run.step_min * 60000));
    const share = g.b_s13 ? Math.round((g.s13[n - 1] / Math.max(1, g.b_s13[n - 1])) * 100) : 100;
    const routes = `${g.routes} ${plural(g.routes, "маршрут", "маршрута", "маршрутов")}`;
    const what = share < 50 ? "затор" : share >= 150 ? "всплеск" : "";
    const label = what ? `${what} с ${since}: ${routes}, обработка ${share}%`
      : share !== 100 ? `с ${since} обработка ${share}%: ${routes}` : `с ${since} другие отгрузки: ${routes}`;
    mark = { from: Math.max(0, i - (n - 1 - back)), label, short: what ? `${what} с ${since}` : `с ${since}` };
  }
  return { baseFc: run.all.b50, baseHist: run.all.b_hist, baseRoutes: run.base_routes ?? undefined, mark };
}

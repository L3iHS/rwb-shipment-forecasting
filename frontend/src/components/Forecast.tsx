import { motion, useReducedMotion } from "motion/react";
import { ArrowDown, Search } from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";
import type { Board, Route } from "../data";
import { dec1, hhmm, int, pct, vol } from "../format";
import { base, useIntro } from "../motion";
import { clock, spread } from "../plan";
import type { Overlay } from "../upload";
import { Fan } from "./charts";

const BACK = 8; // steps of history on the network chart: as far back as the forecast looks ahead
const SHOWN = 200; // rows in the list; the rest are found by number
const item = { hidden: { opacity: 0, y: 10 }, show: { opacity: 1, y: 0, transition: base } };

/** Goods shipped over the next 4 hours: the cumulative half-hour inflow after 8 steps. */
const next4h = (r: { c50: number[] }) => r.c50[r.c50.length - 1];
const signed = (v: number) => (v > 0 ? "+" : v < 0 ? "−" : "") + int(Math.abs(v));

interface Props {
  board: Board;
  overlay?: Overlay; // a forecast from an uploaded file: what it was without the file
  selected: number;
  onSelect: (id: number) => void;
}

/** The main screen: the forecast for the next 4 hours, how good it is, and every route. */
export function Forecast({ board, overlay, selected, onSelect }: Props) {
  const reduced = useReducedMotion();
  const intro = useIntro(overlay ? "forecast-file" : "forecast");
  const n = board.network;
  const hist = n.hist.slice(-BACK - 1);
  const times = Array.from({ length: hist.length + n.p50.length }, (_, j) => hhmm(clock(board, j - BACK)));
  const total = useMemo(() => board.routes.reduce((s, r) => s + next4h(r), 0), [board]);
  const baseTotal = useMemo(() => overlay?.baseRoutes?.reduce((s, b) => s + next4h(b), 0), [overlay]);
  const peak = n.p50.indexOf(Math.max(...n.p50));
  const end = hhmm(clock(board, n.p50.length));

  return (
    <motion.div className="space-y-10" initial={intro ? "hidden" : false} animate="show"
      variants={{ show: { transition: { staggerChildren: reduced ? 0 : 0.07 } } }}>
      {overlay?.same && (
        <motion.p variants={item} className="w-fit rounded-[10px] bg-alert-soft px-3.5 py-2 text-[15px] text-ink" role="status">
          В файле те же данные, что в истории, поэтому прогноз не изменился.
        </motion.p>
      )}
      <motion.section variants={item} className="grid items-center gap-x-6 gap-y-5 @5xl:grid-cols-[minmax(0,1fr)_clamp(400px,35cqw,490px)]">
        <div className="min-w-0">
          <div className="flex flex-col gap-4 @xl:flex-row @xl:items-center @xl:gap-[clamp(16px,1.6cqw,24px)]">
            <span className={`self-start whitespace-nowrap font-hero text-[clamp(80px,9cqw,112px)] font-extrabold tabular-nums leading-[0.8] tracking-[-0.02em] @xl:self-auto ${
              overlay ? "text-alert" : "text-forecast"}`}>
              {vol(total)}
            </span>
            <h1 className="font-display text-[clamp(28px,3.3cqw,40px)] font-semibold leading-[1.06]">
              Отгрузим<br /><span className="whitespace-nowrap">до {end}</span>
            </h1>
          </div>
          {baseTotal != null && overlay?.baseFc ? (
            <dl className="mt-3.5 flex flex-wrap gap-x-6 gap-y-3">
              <Figure value={`${signed(((total / baseTotal) - 1) * 100)}%`} label={`исходный прогноз ${vol(baseTotal)}`} />
            </dl>
          ) : null}
        </div>
        <ForecastMetrics board={board} peak={vol(n.p50[peak])} peakAt={times[BACK + 1 + peak]} />
      </motion.section>

      <motion.section variants={item}>
        <h2 className="mb-3 font-display text-[22px] font-semibold leading-7">
          Отгрузки
          <span className="num ml-2.5 text-[13px] font-medium text-muted">{times[0]}–{times[times.length - 1]}</span>
        </h2>
        <Fan peak={!overlay?.baseFc} height={210} times={times} hist={hist} fc={n.p50} lo={n.lo} hi={n.hi}
          baseFc={overlay?.baseFc} histBase={overlay?.baseHist?.slice(-BACK - 1)} mark={overlay?.mark} />
      </motion.section>

      <motion.div variants={item}>
        <Routes board={board} overlay={overlay} selected={selected} onSelect={onSelect} />
      </motion.div>

      {!overlay && (
        <>
          <motion.div variants={item}><Days board={board} /></motion.div>
        </>
      )}
    </motion.div>
  );
}

function Figure({ value, label }: { value: ReactNode; label: string }) {
  return (
    <div className="border-l border-line pl-6 first:border-0 first:pl-0">
      <dd className="num whitespace-nowrap text-[20px] font-semibold leading-6">{value}</dd>
      <dt className="whitespace-nowrap text-[13px] leading-[18px] text-ink-2">{label}</dt>
    </div>
  );
}

/** Why the number can be trusted: the model on a week it never saw. */
function ForecastMetrics({ board, peak, peakAt }: { board: Board; peak: string; peakAt: string }) {
  const bt = board.backtest;
  return (
    <div className="w-full max-w-[520px] @5xl:max-w-none">
      <dl className="grid grid-cols-2 items-baseline gap-y-4 @lg:grid-cols-[max-content_max-content_minmax(0,1fr)] @lg:gap-x-5">
        <div className="col-span-2 @lg:col-span-1">
          <dd className="num whitespace-nowrap text-[20px] font-semibold leading-6">{peak}</dd>
          <dt className="text-[13px] leading-[18px] text-ink-2">в час на пике, в {peakAt}</dt>
        </div>
        <div className="@lg:border-l @lg:border-line @lg:pl-5">
          <dd className="num whitespace-nowrap text-[28px] font-semibold leading-8">{(bt.point_1030 ?? bt.point).toFixed(3).replace(".", ",")}</dd>
          <dt className="text-[13px] leading-[18px] text-ink-2">ошибка</dt>
        </div>
        <div className="border-l border-line pl-5">
          <dd className="num whitespace-nowrap text-[28px] font-semibold leading-8">{pct(bt.intervals["80%"] ?? 0)}</dd>
          <dt className="text-[13px] leading-[18px] text-ink-2">факт в диапазоне прогноза</dt>
        </div>
      </dl>
    </div>
  );
}

type Sort = "volume" | "error" | "change" | "number";

function Routes({ board, overlay, selected, onSelect }: Props) {
  const [query, setQuery] = useState("");
  const base = overlay?.baseRoutes;
  // With a file the list is about what it changed; otherwise about volume.
  const [sort, setSort] = useState<Sort>(base ? "change" : "volume");
  const change = (i: number) => (base ? next4h(board.routes[i]) / Math.max(1, next4h(base[i])) - 1 : 0);
  const rows = useMemo(() => {
    const all = board.routes.map((r, i) => ({ r, i, v: next4h(r), d: change(i) }));
    const hit = query ? all.filter((x) => String(x.r.id).startsWith(query)) : all;
    const by: Record<Sort, (a: typeof all[0], b: typeof all[0]) => number> = {
      volume: (a, b) => b.v - a.v,
      error: (a, b) => (b.r.wape ?? 0) - (a.r.wape ?? 0),
      change: (a, b) => Math.abs(b.d) - Math.abs(a.d),
      number: (a, b) => a.r.id - b.r.id,
    };
    return hit.sort((a, b) => by[sort](a, b) || a.r.id - b.r.id);
  }, [board, base, query, sort]);
  const shown = rows.slice(0, SHOWN);
  const maxV = Math.max(1, ...board.routes.map(next4h));
  const r = board.routes.find((q) => q.id === selected) ?? board.routes[0];
  const i = board.routes.indexOf(r);
  const cols = "grid-cols-[72px_minmax(0,1fr)_64px]";
  const head = (key: Sort, label: string, right = false) => (
    <button onClick={() => setSort(key)} aria-pressed={sort === key}
      className={`flex items-center gap-1 transition-colors hover:text-ink aria-pressed:text-ink ${right ? "justify-end" : ""}`}>
      {label}<ArrowDown size={11} className={sort === key ? "" : "opacity-0"} />
    </button>
  );

  return (
    <div className="grid gap-5 @3xl:grid-cols-[clamp(340px,34cqw,400px)_minmax(0,1fr)]">
      <div className="relative h-[360px] @3xl:h-auto">
        <section className="panel flex h-full min-h-0 flex-col overflow-hidden @3xl:absolute @3xl:inset-0">
          <div className="flex items-center justify-between gap-3 px-5 pb-3 pt-4">
            <h2 className="font-display text-[22px] font-semibold leading-7">
              Маршруты<span className="num ml-2.5 text-[13px] font-medium text-muted">{rows.length}</span>
            </h2>
            <label className="flex w-[132px] items-center gap-2 rounded-[10px] bg-raised px-2.5 py-1.5 text-[14px] focus-within:ring-1 focus-within:ring-ink-2">
              <Search size={14} className="shrink-0 text-muted" />
              <input value={query} onChange={(e) => setQuery(e.target.value.trim())} placeholder="№ маршрута" aria-label="Номер маршрута"
                inputMode="numeric" className="num w-full bg-transparent outline-none placeholder:font-sans placeholder:text-muted" />
            </label>
          </div>
          <div className={`grid ${cols} gap-2.5 border-b border-line px-5 pb-2 text-[11px] text-muted`}>
            {head("number", "Маршрут")}
            {head("volume", "За 4 ч")}
            {base ? head("change", "Изменение", true) : head("error", "Ошибка", true)}
          </div>
          <ul className="min-h-0 flex-1 overflow-y-auto p-2" role="listbox" aria-label="Маршруты">
            {shown.map(({ r: q, v, d }) => (
              <li key={q.id}>
                <button role="option" aria-selected={selected === q.id} onClick={() => onSelect(q.id)}
                  className={`grid w-full ${cols} items-center gap-2.5 rounded-[10px] px-3 py-2 text-left text-[14px] transition-colors duration-100 hover:bg-raised/60 aria-selected:bg-raised aria-selected:ring-1 aria-selected:ring-inset aria-selected:ring-line`}>
                  <span className="num font-medium">№ {q.id}</span>
                  <span className="flex items-center gap-2.5">
                    <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-line">
                      <span className="block h-full rounded-full bg-forecast" style={{ width: `${(v / maxV) * 100}%` }} />
                    </span>
                    <span className="num w-[60px] whitespace-nowrap text-right text-xs text-ink-2">{vol(v)}</span>
                  </span>
                  <span className={`num text-right text-xs ${base ? (Math.abs(d) >= 0.05 ? "text-alert" : "text-muted") : "text-ink-2"}`}>
                    {base ? `${signed(d * 100)}%` : q.wape != null ? pct(q.wape) : ""}
                  </span>
                </button>
              </li>
            ))}
            {rows.length > shown.length && (
              <li className="px-2 py-3 text-center text-xs text-muted">Показаны первые {shown.length}, остальные ищите по номеру</li>
            )}
            {rows.length === 0 && <li className="px-2 py-12 text-center text-[15px] text-ink-2">Маршрута № {query} нет</li>}
          </ul>
        </section>
      </div>
      <RouteForecast board={board} r={r} baseP50={base?.[i]?.p50} />
    </div>
  );
}

function RouteForecast({ board, r, baseP50 }: { board: Board; r: Route; baseP50?: number[] }) {
  const h0 = r.h.length - 1;
  const times = Array.from({ length: r.h.length + r.p50.length }, (_, j) => hhmm(clock(board, j - h0)));
  const peak = r.p50.indexOf(Math.max(...r.p50));
  const band = spread(r) / 2;
  const changed = !!baseP50 && baseP50.some((v, k) => Math.abs(v - r.p50[k]) > 0.5);
  return (
    <section className="panel @container min-w-0 overflow-hidden p-5 sm:p-6">
      <div className="flex flex-col gap-2 @lg:flex-row @lg:items-start @lg:gap-x-4">
        <h2 className="shrink-0 font-display text-[clamp(30px,6.5cqw,40px)] font-semibold leading-[1.08]">
          <span className="num mr-3 text-[0.6em] text-muted">№</span>{r.id}
        </h2>
        <dl className="flex min-w-0 flex-1 flex-wrap gap-x-4 gap-y-1 text-[14px] text-ink-2 @lg:justify-end">
          <div className="whitespace-nowrap"><dt className="inline">за 4 ч </dt><dd className="num inline font-semibold text-ink">{vol(next4h(r))}</dd></div>
          <div className="whitespace-nowrap"><dt className="inline">пик в </dt><dd className="num inline font-semibold text-ink">{times[h0 + 1 + peak]}</dd></div>
          <div className="whitespace-nowrap"><dt className="inline">диапазон </dt><dd className="num inline font-semibold text-ink">±{Math.round(band * 100)}%</dd></div>
          {r.wape != null && (
            <div className="whitespace-nowrap"><dt className="inline">ошибка </dt><dd className="num inline font-semibold text-ink">{pct(r.wape)}</dd>
              {r.bias != null && <span className="num">, сдвиг {r.bias > 0 ? "+" : ""}{dec1(r.bias * 100)}%</span>}
            </div>
          )}
        </dl>
      </div>
      <div className="mt-3">
        <Fan zero height={230} times={times} hist={r.h} fc={r.p50} lo={r.p10} hi={r.p90} baseFc={changed ? baseP50 : undefined} shadeGap={false} />
      </div>
    </section>
  );
}

/** How wrong the forecast was day by day over the check week: error, bias of the total, and how often the band held. */
function Days({ board }: { board: Board }) {
  const days = board.monitor;
  const max = Math.max(...days.map((d) => d.wape)) * 1.1;
  return (
    <section>
      <h2 className="font-display text-[22px] font-semibold leading-7">По дням проверки</h2>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[520px] text-[14px]">
          <thead>
            <tr className="border-b border-line text-left text-[11px] text-muted">
              <th className="py-2 pr-4 font-normal">День</th>
              <th className="py-2 pr-4 font-normal">Ошибка (WAPE)</th>
              <th className="py-2 pr-4 text-right font-normal">Сдвиг суммы</th>
              <th className="py-2 text-right font-normal">Факт в диапазоне</th>
            </tr>
          </thead>
          <tbody>
            {days.map((d) => (
              <tr key={d.day} className="border-b border-line/60">
                <td className="num py-2 pr-4 text-ink-2">{d.day}</td>
                <td className="py-2 pr-4">
                  <span className="flex items-center gap-3">
                    <span className="h-1.5 w-40 overflow-hidden rounded-full bg-line">
                      <span className="block h-full rounded-full bg-ink-2" style={{ width: `${(d.wape / max) * 100}%` }} />
                    </span>
                    <span className="num">{pct(d.wape)}</span>
                  </span>
                </td>
                <td className="num py-2 pr-4 text-right">{d.bias > 0 ? "+" : d.bias < 0 ? "−" : ""}{dec1(Math.abs(d.bias) * 100)}%</td>
                <td className="num py-2 text-right">{pct(d.cov)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

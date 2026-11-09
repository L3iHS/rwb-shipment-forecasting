import { motion, useReducedMotion } from "motion/react";
import { useMemo, useRef } from "react";
import type { Board } from "../data";
import { hhmm, int, plural } from "../format";
import { base } from "../motion";
import { clock, confidenceCuts, type Kpis, type Scenario } from "../plan";
import { Buckets } from "./Buckets";
import { bucketOf, CallQueue, type Filter } from "./CallQueue";
import { Fan } from "./charts";
import { RouteCard } from "./RouteCard";
import { Ticker } from "./Ticker";
import { WhatIf } from "./WhatIf";

interface Props {
  board: Board;
  calls: number[];
  k: Kpis;
  lead: number;
  scenario: Scenario;
  onScenario: (s: Scenario) => void;
  filter: Filter;
  onFilter: (f: Filter) => void;
  query: string;
  onQuery: (q: string) => void;
  selected: number;
  onSelect: (id: number) => void;
}

const BACK = 8; // steps of history on the timeline: as far back as the forecast looks ahead, so "now" sits in the middle
const item = { hidden: { opacity: 0, y: 10 }, show: { opacity: 1, y: 0, transition: base } };

function Key({ swatch, children }: { swatch: string; children: string }) {
  return <span className="flex items-center gap-1.5"><i className={`inline-block ${swatch}`} />{children}</span>;
}

/** Screen "Вызовы": how many trucks to call now, the shift on one time axis, then who exactly and why. */
export function Shift({ board, calls, k, lead, scenario, onScenario, filter, onFilter, query, onQuery, selected, onSelect }: Props) {
  const reduced = useReducedMotion();
  const card = useRef<HTMLDivElement>(null);
  const cuts = useMemo(() => confidenceCuts(board.routes), [board]);
  const i = board.routes.findIndex((r) => r.id === selected);
  const n = board.network;
  const hist = n.hist.slice(-BACK - 1);
  const times = Array.from({ length: hist.length + n.p50.length }, (_, j) => hhmm(clock(board, j - BACK)));
  // Routes to call a truck for at each step from now: the wave the dispatcher has to ride.
  const wave = useMemo(() => {
    const w = new Array<number>(n.p50.length + 1).fill(0);
    for (const c of calls) if (c >= 0 && c < w.length) w[c]++;
    return w;
  }, [calls, n.p50.length]);
  const active = (step: number) => !query && (filter === "hour" ? step <= 2 : bucketOf(step) === filter);

  const select = (id: number) => {
    onSelect(id);
    if (window.innerWidth < 1024) card.current?.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" });
  };

  return (
    <motion.div className="space-y-8" initial="hidden" animate="show"
      variants={{ show: { transition: { staggerChildren: reduced ? 0 : 0.07 } } }}>
      <motion.section variants={item} className="flex flex-wrap items-end justify-between gap-x-10 gap-y-6">
        <div className="flex items-end gap-4 sm:gap-6">
          {k.callNow > 0 && (
            <span className="text-[72px] font-bold leading-[0.82] tracking-[-0.04em] text-signal sm:text-[96px]">
              <Ticker value={k.callNow} from={0} />
            </span>
          )}
          <div className="pb-0.5">
            <h1 className="text-[24px] font-semibold leading-[1.1] tracking-[-0.02em] sm:text-[34px]">
              {k.callNow > 0 ? <>{plural(k.callNow, "маршруту", "маршрутам", "маршрутам")} нужна<br />машина сейчас</> : "Сейчас вызывать не нужно"}
            </h1>
            <p className="mt-2 text-ink-2">
              Ещё <span className="font-medium text-ink">{int(k.callHour)}</span> — в течение часа
              <span className="mx-2 text-muted">·</span>
              <span className="whitespace-nowrap font-medium text-ink">{int(k.fleet)}</span> {plural(k.fleet, "машина", "машины", "машин")} до {hhmm(clock(board, 8))}
            </p>
          </div>
        </div>
        <WhatIf board={board} s={scenario} onChange={onScenario} />
      </motion.section>

      <motion.section variants={item}>
        <div className="mb-1 flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
          <h2 className="text-[18px] font-semibold leading-6 tracking-tight">
            Смена на одной оси <span className="ml-1.5 text-[13px] font-normal tracking-normal text-muted">нагрузка склада, товаров в час</span>
          </h2>
          <div className="flex gap-4 text-xs text-ink-2">
            <Key swatch="h-0.5 w-3.5 rounded-full bg-ink">факт</Key>
            <Key swatch="h-0.5 w-3.5 rounded-full bg-forecast">прогноз</Key>
            <Key swatch="h-2.5 w-2 rounded-sm bg-signal">вызовы</Key>
          </div>
        </div>
        <Fan peak times={times} hist={hist} fc={n.p50} lo={n.lo} hi={n.hi} calls={wave} active={active}
          onSlot={(step) => onFilter(bucketOf(step))} />
      </motion.section>

      <motion.div variants={item}>
        <Buckets calls={calls} waiting={k.waiting} active={query ? null : filter} onPick={onFilter} />
      </motion.div>

      <motion.div variants={item} className="grid gap-5 lg:grid-cols-[400px_1fr]">
        <div className="h-[480px] lg:h-[640px]">
          <CallQueue board={board} calls={calls} filter={filter} query={query} onQuery={onQuery} selected={selected} onSelect={select} />
        </div>
        <div ref={card} className="scroll-mt-20">
          <RouteCard board={board} r={board.routes[i]} call={calls[i]} lead={lead} leadH={scenario.leadH} risk={scenario.risk} cuts={cuts} />
        </div>
      </motion.div>

      <motion.p variants={item} className="num text-[11px] text-muted">
        {board.model.lib === "lgbm" ? "LightGBM" : board.model.lib} · обучен до{" "}
        {new Date(board.model.trained_until).toLocaleString("ru-RU", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" })}
      </motion.p>
    </motion.div>
  );
}

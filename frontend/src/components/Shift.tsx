import { motion, useReducedMotion } from "motion/react";
import { Undo2 } from "lucide-react";
import { useMemo, useRef } from "react";
import type { Board } from "../data";
import { hhmm, int, plural } from "../format";
import { base, flow } from "../motion";
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
  log: number[]; // trucks called from the console, in order, as route ids
  called: Record<number, number>;
  onCalled: (id: number) => void;
  onUndo: (id?: number) => void;
}

const BACK = 8; // steps of history on the timeline: as far back as the forecast looks ahead, so "now" sits in the middle
const item = { hidden: { opacity: 0, y: 10 }, show: { opacity: 1, y: 0, transition: base } };

function Key({ swatch, children }: { swatch: string; children: string }) {
  return <span className="flex items-center gap-1.5"><i className={`inline-block ${swatch}`} />{children}</span>;
}

/** Screen "Вызовы": how many trucks to call now, the shift on one time axis, then who exactly and why. */
export function Shift({ board, calls, k, lead, scenario, onScenario, filter, onFilter, query, onQuery, selected, onSelect, log, called, onCalled, onUndo }: Props) {
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
        <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:gap-6">
          {k.callNow > 0 && (
            // The hidden copy holds the final width, so the heading beside it does not slide while the number counts up.
            <span className="relative inline-block self-start font-hero text-[120px] font-extrabold tabular-nums leading-[0.78] tracking-[-0.02em] text-signal sm:self-auto sm:text-[152px]">
              <span className="invisible">{int(k.callNow)}</span>
              <span className="absolute inset-0"><Ticker value={k.callNow} from={0} /></span>
            </span>
          )}
          <div className="pb-0.5">
            <h1 className="font-display text-[34px] font-semibold leading-[1.02] sm:text-[46px]">
              {k.callNow > 0 ? <>{plural(k.callNow, "маршруту", "маршрутам", "маршрутам")} нужна<br />машина сейчас</> : "Сейчас вызывать не нужно"}
            </h1>
            <p className="mt-2 text-ink-2">
              Ещё <span className="font-semibold text-ink">{int(k.callHour)}</span> в течение часа,{" "}
              <span className="whitespace-nowrap">всего до {hhmm(clock(board, 8))} нужно</span>{" "}
              <span className="whitespace-nowrap"><span className="font-semibold text-ink">{int(k.fleet)}</span> {plural(k.fleet, "машина", "машины", "машин")}</span>
            </p>
            {/* Progress of the shift: trucks called here against the routes still waiting. Always present, so nothing jumps. */}
            <div className="mt-4 flex items-center gap-3 text-[14px] text-ink-2">
              <span className="relative h-1.5 w-28 shrink-0 overflow-hidden rounded-full bg-line sm:w-40" aria-hidden>
                <motion.span className="absolute inset-0 origin-left rounded-full bg-signal" initial={false}
                  animate={{ scaleX: log.length / Math.max(1, log.length + k.callNow) }} transition={flow} />
              </span>
              <span className="whitespace-nowrap">
                вызвано <span className="num font-semibold text-ink"><Ticker value={log.length} /></span> из{" "}
                <span className="num"><Ticker value={log.length + k.callNow} /></span>
              </span>
              <motion.button onClick={() => onUndo()} disabled={!log.length} aria-hidden={!log.length} initial={false}
                animate={{ opacity: log.length ? 1 : 0 }} transition={base}
                className="inline-flex h-7 items-center gap-1.5 rounded-lg px-2 text-[13px] text-ink-2 ring-1 ring-inset ring-line hover:text-ink disabled:pointer-events-none"
                title="Отменить последний вызов">
                <Undo2 size={13} />
                <span className="num">{log.length ? `№ ${log[log.length - 1]}` : "№ 000"}</span>
              </motion.button>
            </div>
          </div>
        </div>
        <WhatIf board={board} s={scenario} onChange={onScenario} />
      </motion.section>

      <motion.section variants={item}>
        <div className="mb-1 flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
          <h2 className="font-display text-[24px] font-semibold leading-7">
            Смена на одной оси <span className="ml-1.5 hidden font-sans text-[14px] font-normal text-muted sm:inline">нагрузка склада, товаров в час</span>
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

      <motion.div variants={item} className="grid gap-5 lg:grid-cols-[400px_minmax(0,1fr)]">
        {/* On a wide screen the queue takes exactly the card's height, so the two panels end on one line. */}
        <div className="queue-slot">
          <CallQueue board={board} calls={calls} filter={filter} query={query} onQuery={onQuery} selected={selected} onSelect={select} onCalled={onCalled} />
        </div>
        <div ref={card} className="min-w-0 scroll-mt-20">
          <RouteCard board={board} r={board.routes[i]} call={calls[i]} lead={lead} leadH={scenario.leadH} risk={scenario.risk} cuts={cuts}
            calledHere={called[selected] ?? 0} onCalled={() => onCalled(selected)} onUndo={() => onUndo(selected)} />
        </div>
      </motion.div>

    </motion.div>
  );
}

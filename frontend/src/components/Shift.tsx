import { motion, useReducedMotion } from "motion/react";
import { CheckCheck, Undo2 } from "lucide-react";
import { useMemo, useRef } from "react";
import type { Board } from "../data";
import { hhmm, int, leadLabel, plural } from "../format";
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
  total: number; // trucks called from the console
  last?: number[]; // the latest step: route ids of one call or of a "call all"
  called: Record<number, number>;
  planned: number[]; // call steps before any call made here
  due: number; // trucks the rule asks for right now, over all routes
  idle: { steps: number; open: boolean }; // trucks called ahead of the rule stand this long without a load
  onCalled: (id: number) => void;
  onCallAll: () => void;
  onUndo: (id?: number) => void;
}

const BACK = 8; // steps of history on the timeline: as far back as the forecast looks ahead, so "now" sits in the middle
const item = { hidden: { opacity: 0, y: 10 }, show: { opacity: 1, y: 0, transition: base } };

function Key({ swatch, children }: { swatch: string; children: string }) {
  return <span className="flex items-center gap-1.5"><i className={`inline-block ${swatch}`} />{children}</span>;
}

/** Screen "Вызовы": how many trucks to call now, the shift on one time axis, then who exactly and why. */
export function Shift({ board, calls, k, lead, scenario, onScenario, filter, onFilter, query, onQuery, selected, onSelect, total, last, called, planned, due, idle, onCalled, onCallAll, onUndo }: Props) {
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
  const before = useMemo(() => {
    const w = new Array<number>(n.p50.length + 1).fill(0);
    for (const c of planned) if (c >= 0 && c < w.length) w[c]++;
    return w;
  }, [planned, n.p50.length]);
  const active = (step: number) => !query && (filter === "hour" ? step <= 2 : bucketOf(step) === filter);

  const select = (id: number) => {
    onSelect(id);
    // When the card sits under the queue rather than beside it, bring it into view.
    if (card.current && card.current.getBoundingClientRect().left < 200) card.current.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" });
  };

  return (
    <motion.div className="space-y-8" initial="hidden" animate="show"
      variants={{ show: { transition: { staggerChildren: reduced ? 0 : 0.07 } } }}>
      {/* On the left what is happening, on the right the one thing to do about it. Side by side when the console is wide
          enough, else one under the other. Sizes follow the console's own width (container queries), because the
          assistant panel takes its share of the window when open. */}
      <motion.section variants={item} className="grid items-end gap-x-10 gap-y-6 @5xl:grid-cols-[minmax(0,1fr)_400px]">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:gap-6">
          {/* The hidden copy holds the final width, so the heading beside it does not slide while the number counts. */}
          <span className={`relative inline-block self-start font-hero text-[104px] font-extrabold tabular-nums leading-[0.8] tracking-[-0.02em] transition-colors duration-300 sm:min-w-[3.4ch] sm:self-auto sm:text-[128px] ${k.callNow > 0 ? "text-signal" : "text-forecast"}`}>
            <span className="invisible">{int(k.callNow)}</span>
            <span className="absolute inset-0"><Ticker value={k.callNow} from={0} /></span>
          </span>
          <div className="pb-0.5">
            <h1 className="font-display text-[30px] font-semibold leading-[1.06] sm:text-[40px]">
              {k.callNow > 0 ? <>{plural(k.callNow, "маршруту", "маршрутам", "маршрутам")} нужна<br />машина сейчас</> : <>срочных вызовов<br />не осталось</>}
            </h1>
            <p className="mt-2 min-h-[46px] text-ink-2 sm:min-h-0">
              Ещё <span className="font-semibold text-ink">{int(k.callHour)}</span> в течение часа,{" "}
              <span className="whitespace-nowrap">всего до {hhmm(clock(board, 8))} нужно</span>{" "}
              <span className="whitespace-nowrap"><span className="font-semibold text-ink">{int(k.fleet)}</span> {plural(k.fleet, "машина", "машины", "машин")}</span>
            </p>
          </div>
        </div>

        {/* The one action of the screen, with the progress it makes. Every line keeps its place whatever it says. */}
        <div className="w-full @2xl:max-w-[400px] @5xl:justify-self-end">
          <motion.button onClick={onCallAll} disabled={!due} whileTap={{ scale: 0.98 }}
            className="inline-flex h-12 w-full items-center justify-center gap-2.5 rounded-[12px] bg-signal px-5 text-[17px] font-semibold text-on-signal transition-colors disabled:bg-raised disabled:text-muted"
            title="Вызвать машины всем маршрутам, которым пора: столько, сколько просит прогноз">
            <CheckCheck size={19} strokeWidth={2.25} />
            Вызвать всем
            <span className="num min-w-[4ch] text-left text-[15px] font-medium opacity-80">{due || ""}</span>
          </motion.button>
          <div className="mt-3 flex items-center gap-3 text-[14px] text-ink-2">
            <span className="whitespace-nowrap">
              вызвано <span className="num font-semibold text-ink"><Ticker value={total} /></span> из{" "}
              <span className="num"><Ticker value={total + due} /></span>
            </span>
            <span className="relative h-1.5 min-w-8 flex-1 overflow-hidden rounded-full bg-line" aria-hidden>
              <motion.span className="absolute inset-0 origin-left rounded-full bg-signal" initial={false}
                animate={{ scaleX: total / Math.max(1, total + due) }} transition={flow} />
            </span>
            <motion.button onClick={() => onUndo()} disabled={!last} aria-hidden={!last} initial={false}
              animate={{ opacity: last ? 1 : 0 }} transition={base}
              className="inline-flex h-7 items-center gap-1.5 rounded-lg px-2 text-[13px] text-ink-2 ring-1 ring-inset ring-line hover:text-ink disabled:pointer-events-none"
              title={last && last.length > 1 ? `Отменить вызов всем: ${last.length} машин` : "Отменить последний вызов"}>
              <Undo2 size={13} />
              <span className="num min-w-[5ch] text-left">{!last ? "" : last.length > 1 ? `все ${last.length}` : `№ ${last[0]}`}</span>
            </motion.button>
          </div>
          {/* The price of the dispatcher's own decisions; empty until there is something to say. */}
          <p className="mt-1.5 h-5 text-[14px] leading-5">
            {idle.steps > 0 ? (
              <span className="text-alert">Ранние вызовы: машины простоят {idle.open ? "больше " : ""}{leadLabel((idle.steps * board.step_min) / 60)}</span>
            ) : total > 0 ? (
              <span className="text-muted">Все вызовы вовремя, простоя нет</span>
            ) : null}
          </p>
        </div>
      </motion.section>

      <motion.section variants={item}>
        {/* The conditions the plan is computed under sit right over the chart they redraw. */}
        <div className="flex flex-wrap items-center justify-between gap-x-8 gap-y-3">
          <h2 className="font-display text-[22px] font-semibold leading-7">Смена на одной оси</h2>
          <WhatIf board={board} s={scenario} onChange={onScenario} />
        </div>
        <div className="mb-1 mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-2">
          <span className="hidden text-muted sm:inline">нагрузка склада, товаров в час</span>
          <Key swatch="h-0.5 w-3.5 rounded-full bg-ink">факт</Key>
          <Key swatch="h-0.5 w-3.5 rounded-full bg-forecast">прогноз</Key>
          <Key swatch="h-2.5 w-2 rounded-sm bg-signal">вызовы</Key>
        </div>
        <Fan peak times={times} hist={hist} fc={n.p50} lo={n.lo} hi={n.hi} calls={wave} active={active}
          ghost={total ? before : undefined} enRoute={total && lead <= n.p50.length ? { k: lead, count: total } : undefined}
          onSlot={(step) => onFilter(bucketOf(step))} />
      </motion.section>

      <motion.div variants={item}>
        <Buckets calls={calls} waiting={k.waiting} active={query ? null : filter} onPick={onFilter} />
      </motion.div>

      <motion.div variants={item} className="grid gap-5 @[840px]:grid-cols-[380px_minmax(0,1fr)] @6xl:grid-cols-[400px_minmax(0,1fr)]">
        {/* On a wide screen the queue takes exactly the card's height, so the two panels end on one line. */}
        <div className="relative h-[480px] @[840px]:h-auto @[840px]:min-h-[560px]">
          <div className="h-full @[840px]:absolute @[840px]:inset-0">
            <CallQueue board={board} calls={calls} filter={filter} query={query} onQuery={onQuery} selected={selected} onSelect={select} onCalled={onCalled} onFilter={onFilter} />
          </div>
        </div>
        <div ref={card} className="min-w-0 scroll-mt-20">
          <RouteCard board={board} r={board.routes[i]} call={calls[i]} lead={lead} leadH={scenario.leadH} risk={scenario.risk} cuts={cuts}
            calledHere={called[selected] ?? 0} onCalled={() => onCalled(selected)} onUndo={() => onUndo(selected)} />
        </div>
      </motion.div>

      <motion.p variants={item} className="num text-[11px] text-muted">
        {board.model.lib === "lgbm" ? "LightGBM" : board.model.lib}, обучен до{" "}
        {new Date(board.model.trained_until).toLocaleString("ru-RU", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" })}
      </motion.p>
    </motion.div>
  );
}

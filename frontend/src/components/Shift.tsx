import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useMemo, useRef } from "react";
import type { Board } from "../data";
import { hhmm, int, routesWord, vol } from "../format";
import { base } from "../motion";
import { clock, confidenceCuts, peakStep, type Kpis, type Scenario } from "../plan";
import { Buckets } from "./Buckets";
import { CallQueue, type Filter } from "./CallQueue";
import { Fan, fanSeries } from "./charts";
import type { Tab } from "./Header";
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
  onTab: (t: Tab) => void;
}

const SEGMENT_FILTERS: Filter[] = ["now", "soon", "later", "none"];
const item = { hidden: { opacity: 0, y: 8 }, show: { opacity: 1, y: 0, transition: base } };

/** Screen "Вызовы": who to call a truck for and when, and why for the selected route. */
export function Shift({ board, calls, k, lead, scenario, onScenario, filter, onFilter, query, onQuery, selected, onSelect, onTab }: Props) {
  const reduced = useReducedMotion();
  const card = useRef<HTMLDivElement>(null);
  const cuts = useMemo(() => confidenceCuts(board.routes), [board]);
  const i = board.routes.findIndex((r) => r.id === selected);
  const n = board.network;
  const net = fanSeries(board, n.hist, n.p50, n.lo, n.hi);
  const peak = net.data[n.hist.length - 1 + peakStep(board)];
  const end = hhmm(clock(board, 8));

  const select = (id: number) => {
    onSelect(id);
    if (window.innerWidth < 1024) card.current?.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" });
  };

  return (
    <motion.div className="space-y-8" initial="hidden" animate="show"
      variants={{ show: { transition: { staggerChildren: reduced ? 0 : 0.05 } } }}>
      <motion.div variants={item} className="grid gap-8 lg:grid-cols-12">
        <div className="lg:col-span-7">
          <h1 className="text-[28px] font-semibold leading-tight tracking-tight sm:text-[34px]">
            {k.callNow > 0 ? (
              <>Вызвать машину сейчас: <span className="num text-signal"><Ticker value={k.callNow} /></span> {routesWord(k.callNow)}</>
            ) : (
              "Сейчас вызывать машины не нужно"
            )}
          </h1>
          <p className="mt-2 text-ink-2">
            Ещё {int(k.callHour)} — в течение часа.{" "}
            <button onClick={() => onTab("evidence")}
              className="font-medium text-ink underline decoration-line underline-offset-4 hover:decoration-ink">
              Почему верить →
            </button>
          </p>
          <div className="mt-5">
            <WhatIf board={board} s={scenario} onChange={onScenario} />
          </div>
        </div>
        <div className="lg:col-span-5">
          <div className="flex items-baseline justify-between gap-3">
            <h2 className="text-sm font-semibold">Нагрузка склада</h2>
            <span className="flex gap-3 text-xs text-ink-2">
              <span><i className="mr-1 inline-block h-0.5 w-3 bg-ink align-middle" />факт</span>
              <span><i className="mr-1 inline-block h-0.5 w-3 bg-accent align-middle" />прогноз</span>
            </span>
          </div>
          <Fan compact data={net.data} nowLabel={net.nowLabel} peak={peak} animate={!reduced} />
          <p className="num text-xs text-ink-2">
            Пик {peak.t} — {vol(peak.fc!)} в час · до {end} нужно {int(k.fleet)} машин
          </p>
        </div>
      </motion.div>

      <AnimatePresence initial={false}>
        {k.waiting > 0 && (
          <motion.div key="alert" initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }}
            transition={base}
            className="flex flex-wrap items-center justify-between gap-3 rounded-r-lg border-l-2 border-critical bg-card py-2.5 pl-4 pr-3">
            <p className="text-sm">
              Груз ждёт без машины: <b className="num">{k.waiting}</b> {routesWord(k.waiting)}
            </p>
            <button onClick={() => onFilter("jam")}
              className="rounded-md border border-line px-3 py-1 text-xs font-medium transition-colors duration-100 hover:bg-line/40">
              Показать
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      <motion.div variants={item}>
        <Buckets board={board} calls={calls} active={!query && SEGMENT_FILTERS.includes(filter) ? filter : null} onPick={onFilter} />
      </motion.div>

      <motion.div variants={item} className="grid gap-5 lg:grid-cols-[380px_1fr]">
        <div className="h-[480px] lg:h-[600px]">
          <CallQueue board={board} calls={calls} filter={filter} query={query} onQuery={onQuery} selected={selected} onSelect={select} />
        </div>
        <div ref={card} className="scroll-mt-20">
          <RouteCard board={board} r={board.routes[i]} call={calls[i]} lead={lead} leadH={scenario.leadH} risk={scenario.risk} cuts={cuts} />
        </div>
      </motion.div>

      <motion.p variants={item} className="text-xs text-muted">
        {board.model.lib === "lgbm" ? "LightGBM" : board.model.lib}, обучен до{" "}
        {new Date(board.model.trained_until).toLocaleString("ru-RU", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" })}.
        Кузов и время подачи — параметры сервиса.
      </motion.p>
    </motion.div>
  );
}

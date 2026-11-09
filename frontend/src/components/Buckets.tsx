import { motion } from "motion/react";
import { snap } from "../motion";
import { bucketOf, type Filter } from "./CallQueue";
import { Ticker } from "./Ticker";

// The dot repeats the colour of the bars on the shift timeline, so the chips read as its legend.
const CHIPS: [Filter, string, string][] = [
  ["now", "Сейчас", "bg-signal"],
  ["soon", "В течение часа", "bg-signal opacity-50"],
  ["later", "Позже", "bg-ink-2 opacity-40"],
  ["none", "Не нужно", "border border-muted"],
];

interface Props {
  calls: number[];
  waiting: number; // routes whose cargo waits with no truck called
  active: Filter | null; // null while searching by number
  onPick: (f: Filter) => void;
}

/** Filter chips over the queue: when to call, plus the one alert — cargo waiting with no truck. */
export function Buckets({ calls, waiting, active, onPick }: Props) {
  const counts: Partial<Record<Filter, number>> = { now: 0, soon: 0, later: 0, none: 0 };
  for (const c of calls) counts[bucketOf(c)]!++;
  const chip = (f: Filter, label: string, dot: string, n: number, alert = false) => (
    <button key={f} onClick={() => onPick(f)} aria-pressed={active === f}
      className={`relative flex items-baseline gap-2 rounded-[10px] px-3.5 py-2 text-[13px] font-medium transition-colors duration-150 ${
        alert ? "bg-alert-soft text-ink" : "bg-surface text-ink-2 hover:text-ink aria-pressed:text-ink"} ring-1 ring-inset ring-line`}>
      {active === f && <motion.span layoutId="chip" transition={snap} className="absolute inset-0 rounded-[10px] ring-[1.5px] ring-inset ring-ink" />}
      <i className={`size-2 shrink-0 self-center rounded-full ${dot}`} />
      <b className="num text-[15px] font-semibold text-ink"><Ticker value={n} /></b>
      {label}
    </button>
  );
  return (
    <div className="flex flex-wrap gap-2" role="group" aria-label="Фильтр очереди">
      {CHIPS.map(([f, label, dot]) => chip(f, label, dot, counts[f]!))}
      {waiting > 0 && <span className="ml-auto">{chip("jam", "Груз без машины", "bg-alert", waiting, true)}</span>}
    </div>
  );
}

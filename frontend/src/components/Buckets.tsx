import { motion, useReducedMotion } from "motion/react";
import type { Board } from "../data";
import { hhmm } from "../format";
import { flow, snap, useFirstRender } from "../motion";
import { clock } from "../plan";
import { bucketOf, type Filter } from "./CallQueue";
import { Ticker } from "./Ticker";

type Segment = "now" | "soon" | "later" | "none";
const SEGMENTS: [Segment, string, string][] = [
  ["now", "Сейчас", "bg-signal text-white"],
  ["soon", "В течение часа", "bg-signal-soft text-signal"],
  ["later", "Позже", "bg-line text-ink-2"],
  ["none", "Не нужно", "border border-line text-muted"],
];

interface Props {
  board: Board;
  calls: number[];
  active: Filter | null; // highlighted segment; null while searching or showing an alert filter
  onPick: (f: Filter) => void;
}

/** All routes split by when to call; each segment's width is its share and clicking it filters the list. */
export function Buckets({ board, calls, active, onPick }: Props) {
  const reduced = useReducedMotion();
  const first = useFirstRender();
  const counts: Record<Segment, number> = { now: 0, soon: 0, later: 0, none: 0 };
  for (const c of calls) counts[bucketOf(c) as Segment]++;
  const hints: Partial<Record<Segment, string>> = {
    later: `Позже — с ${hhmm(clock(board, 3))} до ${hhmm(clock(board, 8))}`,
    none: "Не нужно — вызванных машин хватает или груза на кузов не наберётся",
  };
  return (
    <section>
      <h2 className="text-sm font-semibold">Когда вызывать</h2>
      <div className="mt-2.5 flex gap-1">
        {SEGMENTS.map(([f, label, cls], i) => (
          <motion.button key={f} onClick={() => onPick(f)} aria-pressed={active === f} title={hints[f]}
            initial={{ flexGrow: 0 }} animate={{ flexGrow: counts[f] }}
            transition={reduced ? { duration: 0 } : { ...flow, delay: first ? i * 0.05 : 0 }}
            className={`relative min-w-[64px] basis-0 sm:min-w-[88px] rounded-md text-left ${cls}`}>
            {active === f && (
              <motion.span layoutId="bucket" transition={snap} className="absolute inset-0 rounded-md ring-2 ring-inset ring-ink" />
            )}
            <span className="relative flex flex-col px-2.5 py-1.5">
              <span className="num text-base font-semibold leading-tight"><Ticker value={counts[f]} /></span>
              <span className="text-[11px] leading-tight">{label}</span>
            </span>
          </motion.button>
        ))}
      </div>
    </section>
  );
}

import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useEffect, useRef } from "react";
import { Search } from "lucide-react";
import type { Board, Route } from "../data";
import { dec2, hhmm } from "../format";
import { base, flow, reorder, snap } from "../motion";
import { clock, inbound, waitsWithoutTruck } from "../plan";

/** now/soon/later/none are the segments of the "when to call" strip; jam, hour and all come from alerts and the assistant. */
export type Filter = "now" | "soon" | "later" | "none" | "jam" | "hour" | "all";

const TITLES: Record<Filter, string> = {
  now: "Вызвать сейчас",
  soon: "В течение часа",
  later: "Позже",
  none: "Не нужно",
  jam: "Груз без машины",
  hour: "В ближайший час, включая срочные",
  all: "Все маршруты",
};
const SHOWN = 120;

/** The strip segment a call step falls into. */
export function bucketOf(call: number): Filter {
  if (call === 0) return "now";
  if (call > 0 && call <= 2) return "soon";
  return call > 2 ? "later" : "none";
}

function matches(r: Route, call: number, f: Filter) {
  if (f === "jam") return waitsWithoutTruck(r);
  if (f === "hour") return call >= 0 && call <= 2;
  if (f === "all") return true;
  return bucketOf(call) === f;
}

/** Queue rows: a search by number looks over all routes; soonest call first, then most cargo at the dock. */
export function queueRows(board: Board, calls: number[], filter: Filter, query = "") {
  const order = (call: number) => (call < 0 ? 99 : call); // "not needed" (-1) sorts last
  return board.routes
    .map((r, i) => ({ r, call: calls[i] }))
    .filter(({ r, call }) => (query ? String(r.id).startsWith(query) : matches(r, call, filter)))
    .sort((a, b) => order(a.call) - order(b.call) || b.r.bl - a.r.bl);
}

export function WhenChip({ board, call }: { board: Board; call: number }) {
  const [text, cls] =
    call === 0 ? ["сейчас", "bg-signal font-semibold text-white"]
    : call > 0 && call <= 2 ? [`через ${call * board.step_min} мин`, "bg-signal-soft font-medium text-signal"]
    : call > 0 ? [hhmm(clock(board, call)), "num bg-line text-ink-2"]
    : ["не нужно", "text-muted"];
  return (
    <AnimatePresence mode="popLayout" initial={false}>
      <motion.span key={text} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }}
        transition={base} className={`inline-block rounded-md px-2 py-0.5 text-xs ${cls}`}>
        {text}
      </motion.span>
    </AnimatePresence>
  );
}

interface Props {
  board: Board;
  calls: number[];
  filter: Filter;
  query: string;
  onQuery: (q: string) => void;
  selected: number;
  onSelect: (id: number) => void;
}

const COLS = "grid-cols-[64px_92px_1fr_36px]";

export function CallQueue({ board, calls, filter, query, onQuery, selected, onSelect }: Props) {
  const reduced = useReducedMotion();
  const list = useRef<HTMLUListElement>(null);
  const rows = queueRows(board, calls, filter, query);
  // Render at least down to the selected row, so a route opened from the assistant is always in the list.
  const shown = rows.slice(0, Math.max(SHOWN, rows.findIndex((x) => x.r.id === selected) + 1));

  useEffect(() => {
    const row = list.current?.querySelector<HTMLElement>(`[data-id="${selected}"]`);
    row?.scrollIntoView({ block: "nearest", behavior: reduced ? "auto" : "smooth" });
  }, [selected, filter, query, reduced]);
  return (
    <section className="flex h-full min-h-0 flex-col rounded-xl border border-line bg-card">
      <div className="space-y-2 border-b border-line p-3">
        <h2 className="font-semibold">
          {query ? "Найдено" : TITLES[filter]} · <span className="num">{rows.length}</span>
        </h2>
        <label className="flex items-center gap-2 rounded-md border border-line px-2.5 py-1.5 text-sm">
          <Search size={14} className="text-muted" />
          <input value={query} onChange={(e) => onQuery(e.target.value.trim())} placeholder="Номер маршрута"
            inputMode="numeric" className="w-full bg-transparent outline-none placeholder:text-muted" />
        </label>
      </div>
      <div className={`grid ${COLS} gap-2 px-4 pt-2 text-[11px] text-muted`}>
        <span>Маршрут</span><span>Вызвать</span><span title="Груз у ворот, в кузовах">У ворот</span><span className="text-center">Едут</span>
      </div>
      <motion.ul ref={list} layoutScroll className="relative min-h-0 flex-1 overflow-y-auto p-1.5" role="listbox" aria-label="Маршруты">
        <AnimatePresence initial={false} mode="popLayout">
          {shown.map(({ r, call }, i) => (
            <motion.li key={r.id} data-id={r.id} layout={i < 40 ? "position" : false} transition={reorder}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0, transition: { ...base, delay: reduced ? 0 : Math.min(i, 12) * 0.015 } }}
              exit={{ opacity: 0, transition: { duration: 0.12 } }}>
              <button role="option" aria-selected={selected === r.id} onClick={() => onSelect(r.id)}
                className={`relative grid w-full ${COLS} items-center gap-2 rounded-lg px-2.5 py-1.5 text-left text-sm transition-colors duration-100 hover:bg-line/40`}>
                {selected === r.id && (
                  <motion.span layoutId="row-sel" transition={snap} className="absolute inset-0 rounded-lg bg-ink/[0.06]" />
                )}
                <span className="num relative font-medium">№ {r.id}</span>
                <span className="relative"><WhenChip board={board} call={call} /></span>
                <span className="relative flex items-center gap-2" title={`${dec2(r.bl)} кузова`}>
                  <span className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-line">
                    <motion.span initial={false} animate={{ scaleX: Math.min(r.bl / 2, 1) }} transition={flow}
                      className={`absolute inset-0 origin-left rounded-full ${waitsWithoutTruck(r) ? "bg-critical" : "bg-ink-2/50"}`} />
                    <span className="absolute inset-y-0 left-1/2 w-px bg-card" />
                  </span>
                  <span className="num w-9 text-right text-xs text-ink-2">{dec2(r.bl)}</span>
                </span>
                <span className="num relative text-center text-ink-2">{inbound(r) || "—"}</span>
              </button>
            </motion.li>
          ))}
        </AnimatePresence>
        {rows.length > shown.length && (
          <li className="px-2 py-3 text-center text-xs text-muted">
            Ещё {rows.length - shown.length} — найдите по номеру
          </li>
        )}
        {rows.length === 0 && (
          <li className="px-2 py-6 text-center text-sm text-muted">
            {query
              ? `Маршрута № ${query} нет`
              : "Здесь пусто"}
          </li>
        )}
      </motion.ul>
    </section>
  );
}

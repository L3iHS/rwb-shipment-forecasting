import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useEffect, useRef } from "react";
import { Search, Truck } from "lucide-react";
import type { Board, Route } from "../data";
import { dec2, hhmm } from "../format";
import { base, flow, reorder, snap } from "../motion";
import { clock, inbound, waitsWithoutTruck } from "../plan";

/** now/soon/later/none are the filter chips and the bars of the shift timeline; jam, hour and all come from the assistant. */
export type Filter = "now" | "soon" | "later" | "none" | "jam" | "hour" | "all";

const TITLES: Record<Filter, string> = {
  now: "Вызвать сейчас",
  soon: "В течение часа",
  later: "Позже",
  none: "Вызов не нужен",
  jam: "Груз без машины",
  hour: "Ближайший час",
  all: "Все маршруты",
};
const SHOWN = 120;

/** The bucket a call step falls into. */
export function bucketOf(call: number): Filter {
  if (call === 0) return "now";
  if (call > 0 && call <= 2) return "soon";
  return call > 2 ? "later" : "none";
}

/** Whether a route with this call step belongs to the filter. */
export function inFilter(r: Route, call: number, f: Filter) {
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
    .filter(({ r, call }) => (query ? String(r.id).startsWith(query) : inFilter(r, call, filter)))
    .sort((a, b) => order(a.call) - order(b.call) || b.r.bl - a.r.bl);
}

export function WhenChip({ board, call }: { board: Board; call: number }) {
  // Only "now" carries the signal colour; the nearer the call, the heavier the chip.
  const [text, cls] =
    call === 0 ? ["сейчас", "bg-signal-soft font-semibold text-signal-ink"]
    : call > 0 && call <= 2 ? [`через ${call * board.step_min} мин`, "text-ink ring-1 ring-inset ring-line"]
    : call > 0 ? [hhmm(clock(board, call)), "text-ink-2"]
    : ["не нужно", "text-muted"];
  return (
    <AnimatePresence mode="popLayout" initial={false}>
      <motion.span key={text} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }}
        transition={base} className={`num inline-block rounded-md px-2 py-0.5 text-xs ${cls}`}>
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
  onCalled: (id: number) => void;
  onFilter: (f: Filter) => void;
}

const COLS = "grid-cols-[62px_112px_1fr_30px]";

export function CallQueue({ board, calls, filter, query, onQuery, selected, onSelect, onCalled, onFilter }: Props) {
  const reduced = useReducedMotion();
  const list = useRef<HTMLUListElement>(null);
  const rows = queueRows(board, calls, filter, query);
  // Render at least down to the selected row, so a route opened from the assistant is always in the list.
  const shown = rows.slice(0, Math.max(SHOWN, rows.findIndex((x) => x.r.id === selected) + 1));

  // Keep the selected row in view by scrolling the list only: scrollIntoView would also drag the page.
  useEffect(() => {
    const ul = list.current;
    const row = ul?.querySelector<HTMLElement>(`[data-id="${selected}"]`);
    if (!ul || !row) return;
    const top = row.offsetTop, bottom = top + row.offsetHeight;
    const to = top < ul.scrollTop ? top - 8 : bottom > ul.scrollTop + ul.clientHeight ? bottom - ul.clientHeight + 8 : null;
    if (to != null) ul.scrollTo({ top: to, behavior: reduced ? "auto" : "smooth" });
    // Working from the keyboard: focus follows the selection, also when a called route leaves the list.
    if (ul.contains(document.activeElement)) row.querySelector("button")?.focus({ preventScroll: true });
  }, [selected, filter, query, reduced]);

  /** Arrows walk the list; Enter on the open route marks its truck as called. */
  function onKey(e: React.KeyboardEvent) {
    const at = shown.findIndex((x) => x.r.id === selected);
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      const next = shown[at + (e.key === "ArrowDown" ? 1 : -1)];
      if (!next) return;
      e.preventDefault();
      onSelect(next.r.id);
      list.current?.querySelector<HTMLElement>(`[data-id="${next.r.id}"] button`)?.focus({ preventScroll: true });
    } else if (e.key === "Enter") {
      const focused = Number((e.target as HTMLElement).closest("[data-id]")?.getAttribute("data-id"));
      if (focused !== selected || at < 0 || shown[at].call !== 0) return; // Enter on another row just opens it
      e.preventDefault();
      onCalled(selected);
    }
  }
  return (
    <section className="panel flex h-full min-h-0 flex-col overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-2.5 px-4 pb-3 pt-4 sm:px-5">
        <h2 className="whitespace-nowrap font-display text-[22px] font-semibold leading-7">
          {query ? "Найдено" : TITLES[filter]}
          <AnimatePresence mode="popLayout" initial={false}>
            <motion.span key={rows.length} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
              transition={base} className="num ml-2.5 inline-block text-[13px] font-medium text-muted">{rows.length}</motion.span>
          </AnimatePresence>
        </h2>
        <label className="flex w-full items-center gap-2 rounded-[10px] bg-raised px-2.5 py-1.5 text-[14px] focus-within:ring-1 focus-within:ring-ink-2 sm:w-[132px]">
          <Search size={14} className="shrink-0 text-muted" />
          <input value={query} onChange={(e) => onQuery(e.target.value.trim())} placeholder="№ маршрута" aria-label="Номер маршрута"
            inputMode="numeric" className="num w-full bg-transparent outline-none placeholder:font-sans placeholder:text-muted" />
        </label>
      </div>
      <div className={`grid ${COLS} gap-2.5 border-b border-line px-5 pb-2 text-[11px] text-muted max-sm:px-[20px]`}>
        <span>Маршрут</span><span>Вызвать</span><span className="whitespace-nowrap" title="Груз у ворот в кузовах. Риска на полосе отмечает один полный кузов">У ворот<span className="hidden sm:inline">, кузовов</span></span>
        <span className="flex justify-end" title="Машин вызвано"><Truck size={13} /></span>
      </div>
      <motion.ul ref={list} layoutScroll onKeyDown={onKey} className="relative min-h-0 flex-1 overflow-y-auto p-2" role="listbox" aria-label="Маршруты">
        <AnimatePresence initial={false} mode="popLayout">
          {shown.map(({ r, call }, i) => (
            <motion.li key={r.id} data-id={r.id} layout={i < 40 ? "position" : false} transition={reorder}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0, transition: { ...base, delay: reduced ? 0 : Math.min(i, 12) * 0.015 } }}
              exit={{ opacity: 0, transition: { duration: 0.12 } }}>
              <button role="option" aria-selected={selected === r.id} onClick={() => onSelect(r.id)}
                className={`relative grid w-full ${COLS} items-center gap-2.5 rounded-[10px] px-3 py-2 text-left text-[14px] transition-colors duration-100 hover:bg-raised/60`}>
                {selected === r.id && (
                  <motion.span layoutId="row-sel" transition={snap} className="absolute inset-0 rounded-[10px] bg-raised ring-1 ring-inset ring-line" />
                )}
                <span className="num relative font-medium">№ {r.id}</span>
                <span className="relative"><WhenChip board={board} call={call} /></span>
                <span className="relative flex items-center gap-2.5" title={`${dec2(r.bl)} кузова у ворот`}>
                  <span className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-line">
                    <motion.span initial={false} animate={{ scaleX: Math.min(r.bl / 2, 1) }} transition={flow}
                      className={`absolute inset-0 origin-left rounded-full ${waitsWithoutTruck(r) ? "bg-alert" : "bg-ink-2"}`} />
                    <span className="absolute inset-y-0 left-1/2 w-0.5 bg-surface" />
                  </span>
                  <span className="num w-9 text-right text-xs text-ink-2">{dec2(r.bl)}</span>
                </span>
                <span className="num relative text-right text-ink-2">{inbound(r) || <span className="text-muted/60">0</span>}</span>
              </button>
            </motion.li>
          ))}
        </AnimatePresence>
        {rows.length > shown.length && (
          <li className="px-2 py-3 text-center text-xs text-muted">Показаны первые {shown.length}. Остальные {rows.length - shown.length} ищите по номеру</li>
        )}
        {rows.length === 0 && (
          <li className="flex flex-col items-center gap-3 px-2 py-12 text-center text-[15px] text-ink-2">
            {query ? `Маршрута № ${query} нет` : filter === "now" ? "Срочных вызовов нет" : "Здесь пусто"}
            {!query && filter === "now" && (
              <button onClick={() => onFilter("soon")}
                className="rounded-[10px] px-3.5 py-2 text-[14px] font-medium text-ink ring-1 ring-inset ring-line hover:bg-raised">
                Показать ближайший час
              </button>
            )}
          </li>
        )}
      </motion.ul>
      <div className="hidden items-center gap-2 border-t border-line px-5 py-2.5 text-xs text-muted lg:flex">
        <kbd className="num rounded bg-raised px-1.5 py-0.5 text-[10px] text-ink-2">↑</kbd>
        <kbd className="num rounded bg-raised px-1.5 py-0.5 text-[10px] text-ink-2">↓</kbd>
        <span className="mr-2">выбрать</span>
        <kbd className="num rounded bg-raised px-1.5 py-0.5 text-[10px] text-ink-2">Enter</kbd>
        <span>вызвал</span>
      </div>
    </section>
  );
}

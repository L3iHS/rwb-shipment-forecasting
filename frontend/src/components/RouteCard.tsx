import { AnimatePresence, motion } from "motion/react";
import { Check, Undo2 } from "lucide-react";
import { useState, type ReactNode } from "react";
import type { Board, Route } from "../data";
import { dec2, hhmm, leadLabel, plural, vol } from "../format";
import { base } from "../motion";
import { clock, inbound, loadsAt, spread, type Risk } from "../plan";
import { Dock, Fan, Meter, TruckGauge } from "./charts";
import { Segmented } from "./Segmented";

type View = "dock" | "flow";
const VIEWS: [View, string][] = [["dock", "Груз у ворот"], ["flow", "Отгрузки"]];

function confidence(r: Route, cuts: [number, number]): { label: string; level: 1 | 2 | 3; note: string } {
  const s = spread(r);
  if (s <= cuts[0]) return { label: "надёжный", level: 3, note: "Узкий коридор прогноза" };
  if (s <= cuts[1]) return { label: "обычный", level: 2, note: "Обычный коридор прогноза" };
  return { label: "неуверенный", level: 1, note: "Широкий коридор: держите резервную машину" };
}

type Tone = "act" | "done" | "plain";

/** The headline of the card. After a call made here it reports the call first, then what comes next. */
function verdict(board: Board, r: Route, call: number, lead: number, leadH: number, calledHere: number): { title: string; why: string; tone: Tone } {
  const n = inbound(r);
  const end = hhmm(clock(board, 8));
  if (calledHere) {
    const arrives = `${calledHere > 1 ? `Вызвано ${calledHere}, приедут` : "Приедет"} в ${hhmm(clock(board, lead))}.`;
    if (call === 0) return { title: "Нужна ещё одна", tone: "act", why: `${arrives} Груза уже больше, чем ${calledHere > 1 ? "они увезут" : "она увезёт"}.` };
    if (call > 0) return { title: "Машина вызвана", tone: "done", why: `${arrives} Следующую вызывать в ${hhmm(clock(board, call))}.` };
    return { title: "Машина вызвана", tone: "done", why: `${arrives} До ${end} больше не нужно.` };
  }
  return { ...plan(board, call, lead, leadH, n, end), tone: call === 0 ? "act" : "plain" };
}

function plan(board: Board, call: number, lead: number, leadH: number, n: number, end: string) {
  if (call === 0) return { title: "Вызвать сейчас", why: `Кузов наберётся к ${hhmm(clock(board, lead))}, как раз к приезду машины.` };
  if (call > 0) return { title: `Вызвать в ${hhmm(clock(board, call))}`, why: `Кузов наберётся к ${hhmm(clock(board, call + lead))}, машина едет ${leadLabel(leadH)}.` };
  return { title: "Вызов не нужен", why: n ? `Вызванных машин хватает до ${end}.` : `До ${end} кузов не наберётся.` };
}

function Stat({ label, title, children }: { label: string; title?: string; children: ReactNode }) {
  return (
    <div className="bg-surface px-3.5 py-3" title={title}>
      <dt className="text-[11px] leading-4 text-muted">{label}</dt>
      <dd className="mt-1 flex h-7 items-center gap-2 whitespace-nowrap text-[18px] font-semibold leading-6 tracking-tight">{children}</dd>
    </div>
  );
}
const unit = (s: string) => <span className="text-[14px] font-normal tracking-normal text-ink-2">{s}</span>;

interface Props {
  board: Board;
  r: Route;
  call: number;
  lead: number;
  leadH: number;
  risk: Risk;
  cuts: [number, number];
  calledHere: number; // trucks called for this route from the console
  onCalled: () => void;
  onUndo: () => void;
}

export function RouteCard({ board, r, call, lead, leadH, risk, cuts, calledHere, onCalled, onUndo }: Props) {
  const [view, setView] = useState<View>("dock");
  const times = Array.from({ length: 9 }, (_, k) => hhmm(clock(board, k)));
  // The curve is the inflow the call rule decides on (cr, or c90 when cautious), so it meets the level at arrival.
  const line = [r.bl, ...(risk === "p90" ? r.c90 : r.cr).map((c) => r.bl + c / r.cap)];
  const cautious = line.map((v, k) => (k === 0 ? v : Math.max(v, loadsAt(r, k - 1, "p90"))));
  const h0 = r.h.length - 1;
  const flowTimes = Array.from({ length: r.h.length + r.p50.length }, (_, i) => hhmm(clock(board, i - h0)));
  const v = verdict(board, r, call, lead, leadH, calledHere);
  const conf = confidence(r, cuts);
  const n = inbound(r);
  return (
    <section className="panel @container relative overflow-hidden p-5 sm:p-6">
      <AnimatePresence mode="popLayout" initial={false}>
        <motion.div key={r.id} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, transition: { duration: 0.12 } }} transition={base}>
          <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div className="min-w-0">
          <div className="eyebrow">Маршрут № {r.id}</div>
          <AnimatePresence mode="wait" initial={false}>
            <motion.div key={v.title} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }}>
              <h2 className={`mt-1 whitespace-nowrap font-display text-[34px] font-semibold leading-[1.08] sm:text-[40px] ${v.tone === "act" ? "text-signal" : v.tone === "done" ? "text-forecast" : ""}`}>{v.title}</h2>
              {/* Two lines are reserved on a phone, so the tiles below never move when the wording changes. */}
              <p className="mt-1.5 min-h-[46px] text-ink-2 sm:min-h-[23px]">{v.why}</p>
            </motion.div>
          </AnimatePresence>
          </div>
          {/* Both controls always keep their place: the button is live only when it is time to call,
              the undo only after a call made here. */}
          <div className="flex shrink-0 items-center gap-2 sm:flex-col sm:items-end sm:pt-1">
            <motion.button onClick={onCalled} disabled={call !== 0} whileTap={{ scale: 0.96 }}
              className={`inline-flex h-11 flex-1 items-center justify-center gap-2 rounded-[10px] bg-signal px-5 text-[16px] font-semibold text-on-signal transition-colors disabled:bg-raised sm:flex-none ${calledHere ? "disabled:text-forecast" : "disabled:text-muted"}`}
              title={call === 0 ? "Отметить, что машина вызвана: план маршрута пересчитается" : calledHere ? "Машина уже вызвана" : "Пока вызывать рано"}>
              <Check size={18} strokeWidth={2.25} />
              <span className="whitespace-nowrap">{call === 0 ? (calledHere ? "Вызвать ещё" : "Вызвать машину") : calledHere ? "Вызвана" : "Вызвать машину"}</span>
            </motion.button>
            <motion.button onClick={onUndo} disabled={!calledHere} aria-hidden={!calledHere} initial={false}
              animate={{ opacity: calledHere ? 1 : 0 }} transition={base}
              className="inline-flex h-11 items-center gap-1.5 rounded-[10px] px-3 text-[14px] text-ink-2 ring-1 ring-inset ring-line hover:text-ink disabled:pointer-events-none sm:h-8 sm:ring-0"
              title="Отменить последний вызов по этому маршруту">
              <Undo2 size={14} />
              <span className="whitespace-nowrap">отменить{calledHere > 1 ? ` (${calledHere})` : ""}</span>
            </motion.button>
          </div>
          </div>

          <dl className="mt-5 grid grid-cols-2 gap-px overflow-hidden rounded-[14px] bg-line ring-1 ring-line @xl:grid-cols-4">
            <Stat label="У ворот" title="Полный кузов это одна машина">
              <TruckGauge loads={r.bl} />
              <span className="num">{dec2(r.bl)}</span>
            </Stat>
            <Stat label="Уже вызвано">
              <span className="num">{n}</span>{unit(plural(n, "машина", "машины", "машин"))}
            </Stat>
            <Stat label="Кузов вмещает">
              <span>{vol(r.cap)}</span>{unit("товаров")}
            </Stat>
            <Stat label="Прогноз" title={conf.note}>
              <Meter level={conf.level} />
              <span className="text-[15px] font-medium tracking-normal">{conf.label}</span>
            </Stat>
          </dl>
        </motion.div>
      </AnimatePresence>

      <div className="mt-6 flex flex-wrap items-center justify-between gap-3">
        <Segmented label="График" value={view} options={VIEWS} onChange={setView} />
        <span className="text-xs text-muted">{view === "dock" ? "кузовов у ворот" : "товаров в час"}</span>
      </div>
      <AnimatePresence mode="wait" initial={false}>
        <motion.div key={view} className="mt-2" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, transition: { duration: 0.1 } }} transition={base}>
          {view === "dock" ? (
            <Dock times={times} line={line} hi={cautious} need={n + 1} runKey={`${r.id}-${call}-${lead}`}
              made={calledHere && lead <= 8 ? { arriveK: lead, count: calledHere } : undefined}
              callK={call >= 0 ? call : undefined} arriveK={call >= 0 && call + lead <= 8 ? call + lead : undefined} />
          ) : (
            <Fan zero height={210} times={flowTimes} hist={r.h} fc={r.p50} lo={r.p10} hi={r.p90} />
          )}
        </motion.div>
      </AnimatePresence>
    </section>
  );
}

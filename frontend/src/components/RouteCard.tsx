import { AnimatePresence, motion } from "motion/react";
import { Check, Clock, Undo2 } from "lucide-react";
import { useState, type ReactNode } from "react";
import type { Board, Route } from "../data";
import { dec2, hhmm, leadLabel, plural, vol } from "../format";
import { base } from "../motion";
import { calledIdle, clock, fillStep, inbound, loadsAt, spread, trucksDue, type Idle, type Risk } from "../plan";
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

type Tone = "act" | "done" | "warn" | "plain";
const TONE: Record<Tone, string> = { act: "text-signal", done: "text-forecast", warn: "text-alert", plain: "" };

/** "30 мин", "1,5 ч", "больше 3 ч" for a number of forecast steps. */
const span = (board: Board, steps: number, open = false) => `${open ? "больше " : ""}${leadLabel((steps * board.step_min) / 60)}`;

/** The headline of the card. After a call made here it reports the call first: when the truck comes and what it costs. */
function verdict(board: Board, call: number, lead: number, leadH: number, n: number, calledHere: number, idle: Idle): { title: string; why: string; tone: Tone } {
  const end = hhmm(clock(board, 8)), at = hhmm(clock(board, lead));
  if (calledHere && idle.steps > 0) {
    const wait = span(board, idle.steps, idle.open);
    if (calledHere > 1) return { title: "Вызваны раньше", tone: "warn", why: `Вызвано ${calledHere}, приедут в ${at}. Без груза простоят ${wait}.` };
    return { title: "Вызвана раньше", tone: "warn", why: idle.lastFill == null
      ? `Приедет в ${at}, а кузов до ${end} не наберётся. Простоит ${wait}.`
      : `Приедет в ${at}, кузов наберётся к ${hhmm(clock(board, idle.lastFill))}. Простоит ${wait}.` };
  }
  if (calledHere) {
    const arrives = `${calledHere > 1 ? `Вызвано ${calledHere}, приедут` : "Приедет"} в ${at}.`;
    if (call === 0) return { title: "Нужна ещё одна", tone: "act", why: `${arrives} Груза уже больше, чем ${calledHere > 1 ? "они увезут" : "она увезёт"}.` };
    if (call > 0) return { title: calledHere > 1 ? "Машины вызваны" : "Машина вызвана", tone: "done", why: `${arrives} Следующую вызывать в ${hhmm(clock(board, call))}.` };
    return { title: calledHere > 1 ? "Машины вызваны" : "Машина вызвана", tone: "done", why: `${arrives} До ${end} больше не нужно.` };
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
  const idle = calledIdle(r, calledHere, lead, risk);
  const v = verdict(board, call, lead, leadH, inbound(r), calledHere, idle);
  const due = trucksDue(r, lead, risk); // trucks the rule asks for now; 0 = any call would be ahead of the rule
  // What one more truck called right now would cost: it arrives after the lead time and waits for its load.
  const nextFill = fillStep(r, inbound(r) + 1, risk);
  const nextWait = nextFill == null ? 8 - lead : nextFill - lead;
  const canEarly = idle.early < 5 && lead <= 8; // a few are a decision, a dozen is a mistake
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
              <h2 className={`mt-1 whitespace-nowrap font-display text-[34px] font-semibold leading-[1.08] sm:text-[40px] ${TONE[v.tone]}`}>{v.title}</h2>
              {/* Two lines are always reserved, so the tiles below never move when the wording changes. */}
              <p className="mt-1.5 min-h-[46px] text-ink-2">{v.why}</p>
            </motion.div>
          </AnimatePresence>
          </div>
          {/* The controls always keep their place. When the rule asks for a truck the button is the coral action and
              calls as many as are due; otherwise it is a quiet "call early" that names its price first. */}
          <div className="flex shrink-0 items-start gap-2 sm:flex-col sm:items-end sm:pt-1">
            <div className="flex flex-1 flex-col sm:flex-none sm:items-end">
              <motion.button onClick={onCalled} disabled={!due && !canEarly} whileTap={{ scale: 0.96 }}
                className={`inline-flex h-11 w-full items-center justify-center gap-2 rounded-[10px] px-5 text-[16px] font-semibold transition-colors disabled:text-muted sm:w-auto ${
                  due ? "bg-signal text-on-signal" : "text-ink ring-1 ring-inset ring-line hover:bg-raised"}`}
                title={due ? "Отметить, что машины вызваны: план маршрута пересчитается" : canEarly ? "Вызвать машину раньше, чем просит прогноз" : "Сверх плана вызвано уже достаточно"}>
                {due ? <Check size={18} strokeWidth={2.25} /> : <Clock size={16} strokeWidth={2} />}
                <span className="whitespace-nowrap">
                  {due > 1 ? `Вызвать ${due} ${plural(due, "машину", "машины", "машин")}` : due ? (calledHere ? "Вызвать ещё" : "Вызвать машину") : canEarly ? "Вызвать раньше" : "Машин хватает"}
                </span>
              </motion.button>
              <div className="mt-1 h-5 whitespace-nowrap text-[13px] leading-5 text-alert sm:text-right">
                {!due && canEarly && (nextWait > 0 ? `простоит ${span(board, nextWait, nextFill == null)}` : "")}
              </div>
            </div>
            <motion.button onClick={onUndo} disabled={!calledHere} aria-hidden={!calledHere} initial={false}
              animate={{ opacity: calledHere ? 1 : 0 }} transition={base}
              className="inline-flex h-11 items-center gap-1.5 rounded-[10px] px-3 text-[14px] text-ink-2 ring-1 ring-inset ring-line hover:text-ink disabled:pointer-events-none sm:h-7 sm:ring-0"
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
              made={calledHere && lead <= 8 ? {
                arriveK: lead, count: calledHere,
                wait: idle.steps > 0 ? { untilK: idle.lastFill, label: `ждёт ${span(board, idle.steps, idle.open)}` } : undefined,
              } : undefined}
              callK={call >= 0 ? call : undefined} arriveK={call >= 0 && call + lead <= 8 ? call + lead : undefined} />
          ) : (
            <Fan zero height={210} times={flowTimes} hist={r.h} fc={r.p50} lo={r.p10} hi={r.p90} />
          )}
        </motion.div>
      </AnimatePresence>
    </section>
  );
}

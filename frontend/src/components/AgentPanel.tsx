import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { ArrowUp, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { localAnswer } from "../brief";
import { dec2, int } from "../format";
import type { Board } from "../data";
import type { Risk } from "../plan";
import { base, fast, panel } from "../motion";
import { findRoutes, networkSummary, type Action } from "../tools";

interface Msg { role: "user" | "assistant"; content: string; local?: boolean; why?: string }

const KEEP = 8; // the backend reads only the latest messages
const TIMEOUT_MS = 90_000;

/** Why the model did not answer, from the backend's `detail`. */
async function reason(res: Response): Promise<string> {
  if (res.status === 404) return "Сервер ассистента не подключён";
  try {
    const { detail } = await res.json();
    if (typeof detail === "string") return detail;
  } catch { /* not JSON: a proxy or the dev server answered */ }
  return res.status >= 500 ? "Сервер недоступен" : "Запрос отклонён";
}

const SUGGEST = ["Где груз ждёт без машины?", "Что если машина едет 2 часа?", "Почему маршрут 485 вызывать сейчас?", "Когда пик?"];

interface Props {
  open: boolean;
  onClose: () => void;
  board: Board;
  calls: number[];
  leadH: number;
  risk: Risk;
  view: { tab: string; route: number; called: Record<number, number> };
  onAction: (a: Action) => void;
}

const tile = { hidden: { opacity: 0, y: 6 }, show: { opacity: 1, y: 0, transition: base } };

/** The shift at a glance: four numbers to press and the three routes with the most cargo. No prose. */
function Summary({ board, calls, onAction }: { board: Board; calls: number[]; onAction: (a: Action) => void }) {
  const reduced = useReducedMotion();
  const n = networkSummary(board, calls);
  const top = findRoutes(board, calls, "call_now", 3).routes;
  const tiles: [string, string, Action | null, string][] = [
    ["Вызвать сейчас", int(n.call_now), { type: "filter", value: "now" }, n.call_now ? "text-signal" : "text-forecast"],
    ["В течение часа", int(n.call_within_hour), { type: "filter", value: "soon" }, "text-ink"],
    ["Груз без машины", int(n.waiting_without_truck), { type: "filter", value: "jam" }, n.waiting_without_truck ? "text-alert" : "text-ink"],
    ["Пик отгрузок", n.peak_at, null, "text-ink"],
  ];
  return (
    <motion.div initial="hidden" animate="show"
      variants={{ show: { transition: { staggerChildren: reduced ? 0 : 0.05, delayChildren: reduced ? 0 : 0.12 } } }}>
      <div className="grid grid-cols-2 gap-2">
        {tiles.map(([label, value, action, tone]) => {
          const body = (
            <>
              <div className={`font-display text-[32px] font-semibold leading-9 ${tone}`}>{value}</div>
              <div className="mt-0.5 text-[14px] leading-5 text-ink-2">{label}</div>
            </>
          );
          return action ? (
            <motion.button key={label} variants={tile} onClick={() => onAction(action)} whileTap={{ scale: 0.97 }}
              className="rounded-[14px] bg-raised px-4 py-3 text-left ring-1 ring-inset ring-transparent transition-shadow hover:ring-line">
              {body}
            </motion.button>
          ) : (
            <motion.div key={label} variants={tile} className="rounded-[14px] bg-raised px-4 py-3">{body}</motion.div>
          );
        })}
      </div>
      {top.length > 0 && (
        <motion.div variants={tile} className="mt-4">
          <div className="text-[14px] leading-5 text-ink-2">Больше всего груза у ворот</div>
          <div className="mt-2 flex flex-wrap gap-2">
            {top.map((r) => (
              <button key={r.route_id} onClick={() => onAction({ type: "open_route", id: r.route_id })}
                className="flex items-baseline gap-2 rounded-[10px] px-3 py-2 ring-1 ring-inset ring-line transition-colors hover:bg-raised">
                <span className="num text-[14px] font-medium text-ink">№ {r.route_id}</span>
                <span className="text-[14px] text-ink-2">{dec2(r.backlog_loads)} кузова</span>
              </button>
            ))}
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}

/** Three dots while the answer is on its way; static under reduced motion. */
function Thinking() {
  const reduced = useReducedMotion();
  return (
    <div className="flex gap-1 py-1" aria-label="Считаю">
      {[0, 1, 2].map((i) => (
        <motion.span key={i} className="size-1.5 rounded-full bg-signal"
          animate={reduced ? { opacity: 0.6 } : { opacity: [0.3, 1, 0.3] }}
          transition={reduced ? { duration: 0 } : { duration: 1.2, repeat: Infinity, delay: i * 0.15 }} />
      ))}
    </div>
  );
}

export function AgentPanel({ open, onClose, board, calls, leadH, risk, view, onAction }: Props) {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [keyboardFocus, setKeyboardFocus] = useState(false);
  const reduced = useReducedMotion();
  const scroller = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    setKeyboardFocus(false);
    const keyboard = (event: KeyboardEvent) => {
      if (event.key === "Tab") setKeyboardFocus(true);
    };
    const pointer = () => setKeyboardFocus(false);
    document.addEventListener("keydown", keyboard, true);
    document.addEventListener("pointerdown", pointer, true);
    return () => {
      document.removeEventListener("keydown", keyboard, true);
      document.removeEventListener("pointerdown", pointer, true);
    };
  }, [open]);

  useEffect(() => {
    const el = scroller.current;
    if (el && msgs.length) el.scrollTo({ top: el.scrollHeight, behavior: reduced ? "auto" : "smooth" });
  }, [msgs.length, busy, reduced]);

  // On a phone the panel covers the console, so close it to show what the action changed.
  const act = (a: Action) => {
    onAction(a);
    if (window.innerWidth < 1024) onClose();
  };

  async function ask(q: string) {
    if (!q.trim() || busy) return;
    const history: Msg[] = [...msgs, { role: "user", content: q }];
    setMsgs(history);
    setText("");
    setBusy(true);
    let why = "Сервер недоступен";
    try {
      const res = await fetch("/api/agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: history.slice(-KEEP).map(({ role, content }) => ({ role, content: content.slice(0, 1000) })), view: { ...view, lead_h: leadH, risk } }),
        signal: AbortSignal.timeout(TIMEOUT_MS),
      });
      if (!res.ok) {
        why = await reason(res);
        throw new Error(why);
      }
      const data: { reply?: string; actions?: Action[] } = await res.json();
      if (!data.reply) throw new Error("empty reply");
      setMsgs((m) => [...m, { role: "assistant", content: data.reply! }]);
      data.actions?.forEach(act);
    } catch (e) {
      if (e instanceof DOMException && e.name === "TimeoutError") why = "Модель не ответила вовремя";
      const a = localAnswer(board, calls, q, leadH, risk);
      setMsgs((m) => [...m, {
        role: "assistant", local: true, why,
        content: a?.text ?? "Без языковой модели я отвечаю на вопросы о маршруте по номеру, о грузе без машины, о времени подачи и о пике.",
      }]);
      a?.actions.forEach(act);
    } finally {
      setBusy(false);
    }
  }

  return (
    <AnimatePresence>
      {open && (
        <motion.div key="backdrop" className="fixed inset-0 z-30 bg-black lg:hidden" onClick={onClose}
          initial={{ opacity: 0 }} animate={{ opacity: 0.4 }} exit={{ opacity: 0 }} transition={fast} />
      )}
      {open && (
        <motion.aside key="panel" initial={{ x: "100%" }} animate={{ x: 0, transition: panel }}
          exit={{ x: "100%", transition: { duration: 0.2, ease: [0.4, 0, 1, 1] } }}
          className="fixed inset-y-0 right-0 z-40 flex w-full max-w-[420px] flex-col border-l border-line bg-surface shadow-[var(--shadow-pop)]">
          <div className="flex items-center justify-between border-b border-line px-5 py-3.5">
            <div>
              <h2 className="font-display text-[22px] font-semibold leading-7">Ассистент<span className="text-signal">.</span></h2>
            </div>
            <button onClick={onClose} className="grid size-8 place-items-center rounded-[10px] text-ink-2 transition-colors hover:bg-raised hover:text-ink" aria-label="Закрыть"><X size={16} /></button>
          </div>
          <div ref={scroller} className="scroll-quiet flex-1 space-y-5 overflow-y-auto p-5 text-[16px] leading-[24px]">
            <Summary board={board} calls={calls} onAction={act} />
            {msgs.map((m, i) => (
              <motion.div key={i} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={base}
                className={m.role === "user" ? "ml-auto w-fit max-w-[85%] rounded-[14px] rounded-br-md bg-ink px-3.5 py-2 text-ground" : "mr-4"}>
                {m.content}
                {m.local && <div className="mt-1.5 text-[13px] leading-5 text-muted">Ответ без языковой модели. {m.why}</div>}
              </motion.div>
            ))}
            {busy && <Thinking />}
          </div>
          <div className="border-t border-line p-4">
            {msgs.length === 0 && (
              <div className="mb-2 flex flex-wrap gap-1.5">
                {SUGGEST.map((s) => (
                  <button key={s} onClick={() => ask(s)} className="rounded-full px-3.5 py-2 text-[14px] leading-5 text-ink-2 ring-1 ring-inset ring-line transition-colors hover:bg-raised hover:text-ink">{s}</button>
                ))}
              </div>
            )}
            <form onSubmit={(e) => { e.preventDefault(); ask(text); }} className={`agent-composer flex items-center gap-2 rounded-[14px] bg-raised px-3.5 py-2.5 ${keyboardFocus ? "keyboard-focus" : ""}`}>
              <input value={text} onChange={(e) => setText(e.target.value)} placeholder="Вопрос" aria-label="Вопрос ассистенту" maxLength={500}
                className="min-w-0 flex-1 bg-transparent text-[16px] outline-none placeholder:text-muted" />
              <button type="submit" disabled={!text.trim() || busy} className="grid size-7 place-items-center rounded-lg bg-ink text-ground transition-opacity disabled:opacity-30" aria-label="Отправить">
                <ArrowUp size={14} />
              </button>
            </form>
          </div>
        </motion.aside>
      )}
    </AnimatePresence>
  );
}

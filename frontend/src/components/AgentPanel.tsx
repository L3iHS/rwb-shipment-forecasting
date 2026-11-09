import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { ArrowUp, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { brief, localAnswer, type Line } from "../brief";
import type { Board } from "../data";
import type { Risk } from "../plan";
import { base, fast, panel } from "../motion";
import type { Action } from "../tools";

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

const briefItem = { hidden: { opacity: 0, y: 4 }, show: { opacity: 1, y: 0, transition: base } };

function BriefLine({ line, onAction }: { line: Line; onAction: (a: Action) => void }) {
  return (
    <motion.li variants={briefItem} className="leading-relaxed">
      {line.map((p, i) => typeof p === "string" ? <span key={i}>{p}</span> : (
        <button key={i} onClick={() => onAction(p.action)} className="num font-semibold text-ink underline decoration-signal/60 decoration-2 underline-offset-4 transition-colors hover:decoration-signal">
          {p.label}
        </button>
      ))}
    </motion.li>
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
              <h2 className="font-display text-[24px] font-semibold leading-7">Ассистент<span className="text-signal">.</span></h2>
            </div>
            <button onClick={onClose} className="grid size-8 place-items-center rounded-[10px] text-ink-2 transition-colors hover:bg-raised hover:text-ink" aria-label="Закрыть"><X size={16} /></button>
          </div>
          <div ref={scroller} className="scroll-quiet flex-1 space-y-4 overflow-y-auto p-5 text-[15px] leading-[22px]">
            <div className="rounded-[14px] bg-raised p-4">
              <div className="eyebrow mb-2">Сводка смены</div>
              <motion.ul className="space-y-2" initial="hidden" animate="show"
                variants={{ show: { transition: { staggerChildren: reduced ? 0 : 0.04, delayChildren: reduced ? 0 : 0.12 } } }}>
                {brief(board, calls).map((l, i) => <BriefLine key={i} line={l} onAction={act} />)}
              </motion.ul>
            </div>
            {msgs.map((m, i) => (
              <motion.div key={i} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={base}
                className={m.role === "user" ? "ml-auto w-fit max-w-[85%] rounded-[14px] rounded-br-md bg-ink px-3.5 py-2 text-ground" : "mr-6 leading-relaxed"}>
                {m.content}
                {m.local && <div className="mt-1 text-[11px] text-muted">Ответ без языковой модели. {m.why}</div>}
              </motion.div>
            ))}
            {busy && <Thinking />}
          </div>
          <div className="border-t border-line p-4">
            {msgs.length === 0 && (
              <div className="mb-2 flex flex-wrap gap-1.5">
                {SUGGEST.map((s) => (
                  <button key={s} onClick={() => ask(s)} className="rounded-full px-3 py-1.5 text-xs text-ink-2 ring-1 ring-inset ring-line transition-colors hover:bg-raised hover:text-ink">{s}</button>
                ))}
              </div>
            )}
            <form onSubmit={(e) => { e.preventDefault(); ask(text); }} className={`agent-composer flex items-center gap-2 rounded-[14px] bg-raised px-3.5 py-2.5 ${keyboardFocus ? "keyboard-focus" : ""}`}>
              <input value={text} onChange={(e) => setText(e.target.value)} placeholder="Вопрос" aria-label="Вопрос ассистенту" maxLength={500}
                className="min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-muted" />
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

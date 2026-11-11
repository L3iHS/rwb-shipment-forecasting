import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { ArrowUp, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { localAnswer } from "../brief";
import { pct, vol } from "../format";
import type { Board } from "../data";
import { base, fast, panel } from "../motion";
import { findRoutes, modelQuality, networkForecast, type Action } from "../tools";

interface Msg { role: "user" | "assistant"; content: string; local?: boolean; why?: string }

const KEEP = 8; // the backend reads only the latest messages
const TIMEOUT_MS = 90_000;

/** Why the model did not answer, from the backend's `detail`. */
async function reason(res: Response): Promise<string> {
  try {
    const { detail } = await res.json();
    if (typeof detail === "string") return detail;
  } catch { /* not JSON: a proxy or the dev server answered */ }
  return res.status >= 500 ? "Сервер недоступен" : "Запрос отклонён";
}

const SUGGEST = ["Сколько отгрузим за 4 часа?", "Где прогноз хуже всего?", "Что с маршрутом 702?", "Насколько точна модель?"];

interface Props {
  open: boolean;
  onClose: () => void;
  board: Board;
  view: { tab: string; route: number };
  onAction: (a: Action) => void;
  /** Set on a shift the server's model cannot see (an uploaded file): answer from this board only, and say why. */
  offline?: string;
}

const tile = { hidden: { opacity: 0, y: 6 }, show: { opacity: 1, y: 0, transition: base } };

/** The forecast at a glance: four numbers and the three largest routes. No prose. */
function Summary({ board, onAction }: { board: Board; onAction: (a: Action) => void }) {
  const reduced = useReducedMotion();
  const n = networkForecast(board);
  const q = modelQuality(board);
  const top = findRoutes(board, "largest", 3).routes;
  const tiles: [string, string, string][] = [
    [`товаров до ${n.horizon_end}`, vol(n.volume_4h), "text-forecast"],
    ["пик отгрузок", n.peak_at, "text-ink"],
    ["ошибка на проверке", q.metric_case_from_1030.toFixed(3).replace(".", ","), "text-ink"],
    ["факт в диапазоне", pct(q.band_80_coverage ?? 0), "text-ink"],
  ];
  return (
    <motion.div initial="hidden" animate="show"
      variants={{ show: { transition: { staggerChildren: reduced ? 0 : 0.05, delayChildren: reduced ? 0 : 0.12 } } }}>
      <div className="grid grid-cols-2 gap-2">
        {tiles.map(([label, value, tone]) => (
          <motion.div key={label} variants={tile} className="rounded-[14px] bg-raised px-4 py-3">
            <div className={`num whitespace-nowrap text-[26px] font-semibold leading-9 ${tone}`}>{value}</div>
            <div className="mt-0.5 text-[14px] leading-5 text-ink-2">{label}</div>
          </motion.div>
        ))}
      </div>
      <motion.div variants={tile} className="mt-4">
        <div className="text-[14px] leading-5 text-ink-2">Крупнейшие маршруты</div>
        <div className="mt-2 flex flex-wrap gap-2">
          {top.map((r) => (
            <button key={r.route_id} onClick={() => onAction({ type: "open_route", id: r.route_id })}
              className="flex items-baseline gap-2 rounded-[10px] px-3 py-2 ring-1 ring-inset ring-line transition-colors hover:bg-raised">
              <span className="num text-[14px] font-medium text-ink">№ {r.route_id}</span>
              <span className="num text-[14px] text-ink-2">{vol(r.volume_4h)}</span>
            </button>
          ))}
        </div>
      </motion.div>
    </motion.div>
  );
}

function Thinking() {
  return <div role="status" className="text-[14px] leading-5 text-muted">Считаю…</div>;
}

/** "Сервер недоступен" -> "сервер недоступен", leaving acronyms like "API" alone. */
const lower = (t: string) => (/^.[a-zа-яё]/.test(t) ? t[0].toLowerCase() + t.slice(1) : t);

export function AgentPanel({ open, onClose, board, view, onAction, offline }: Props) {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const reduced = useReducedMotion();
  const scroller = useRef<HTMLDivElement>(null);

  // Keep the Escape handler attached when the parent supplies a new callback.
  const close = useRef(onClose);
  close.current = onClose;
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") close.current(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  useEffect(() => {
    const el = scroller.current;
    if (el && msgs.length) el.scrollTo({ top: el.scrollHeight, behavior: reduced ? "auto" : "smooth" });
  }, [msgs.length, busy, reduced]);

  // On a phone the panel covers the console, so close it to show what the action changed.
  const act = (a: Action) => {
    onAction(a);
    if (window.innerWidth < 1280) onClose();
  };

  async function ask(q: string) {
    if (!q.trim() || busy) return;
    const history: Msg[] = [...msgs, { role: "user", content: q }];
    setMsgs(history);
    setText("");
    setBusy(true);
    let why = offline ?? "Сервер недоступен";
    try {
      if (offline) throw new Error(offline); // straight to the answer computed here, from the board on screen
      const res = await fetch("/api/agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: history.slice(-KEEP).map(({ role, content }) => ({ role, content: content.slice(0, 1000) })), view }),
        signal: AbortSignal.timeout(TIMEOUT_MS),
      });
      if (!res.ok) {
        why = await reason(res);
        throw new Error(why);
      }
      const data: { reply?: string; actions?: Action[] } = await res.json();
      if (!data.reply) {
        why = "Модель ответила пусто";
        throw new Error(why);
      }
      setMsgs((m) => [...m, { role: "assistant", content: data.reply! }]);
      data.actions?.forEach(act);
    } catch (e) {
      if (e instanceof DOMException && e.name === "TimeoutError") why = "Модель не ответила вовремя";
      const a = localAnswer(board, q);
      setMsgs((m) => [...m, {
        role: "assistant", local: true, why,
        content: a?.text ?? "Без модели отвечу про маршрут по номеру, общий прогноз, точность модели, крупные и неточные маршруты.",
      }]);
      a?.actions.forEach(act);
    } finally {
      setBusy(false);
    }
  }

  return (
    <AnimatePresence>
      {open && (
        <motion.div key="backdrop" className="fixed inset-0 z-30 bg-black xl:hidden" onClick={onClose}
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
          <div ref={scroller} className="flex-1 space-y-5 overflow-y-auto p-5 text-[16px] leading-[24px]">
            <Summary board={board} onAction={act} />
            {msgs.map((m, i) => (
              <motion.div key={i} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={base}
                className={m.role === "user" ? "ml-auto w-fit max-w-[85%] rounded-[14px] rounded-br-md bg-ink px-3.5 py-2 text-ground" : "mr-4"}>
                {m.content}
                {m.local && <div className="mt-1.5 text-[13px] leading-5 text-muted">офлайн{m.why ? `: ${lower(m.why)}` : ""}</div>}
              </motion.div>
            ))}
            {busy && <Thinking />}
          </div>
          <div className="border-t border-line p-4">
            {/* The examples stay after the first answer: one tap asks the next question. */}
            <div className="mb-2 flex flex-wrap gap-1.5">
              {SUGGEST.map((s) => (
                <button key={s} onClick={() => ask(s)} disabled={busy} className="rounded-full px-3.5 py-2 text-[14px] leading-5 text-ink-2 ring-1 ring-inset ring-line transition-colors hover:bg-raised hover:text-ink disabled:opacity-40">{s}</button>
                ))}
            </div>
            <form onSubmit={(e) => { e.preventDefault(); ask(text); }} className="flex items-center gap-2 rounded-[14px] bg-raised px-3.5 py-2.5">
              <input value={text} onChange={(e) => setText(e.target.value)} placeholder="Вопрос" aria-label="Вопрос ассистенту" maxLength={500}
                className="assistant-question min-w-0 flex-1 bg-transparent text-[16px] outline-none placeholder:text-muted" />
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

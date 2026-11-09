import { AnimatePresence, motion } from "motion/react";
import { Moon, Sparkles, Sun } from "lucide-react";
import type { Board } from "../data";
import { hhmm } from "../format";
import { fast, snap } from "../motion";

export type Tab = "shift" | "evidence";
const TABS: [Tab, string][] = [
  ["shift", "Вызовы"],
  ["evidence", "Почему верить"],
];

interface Props {
  board: Board;
  tab: Tab;
  onTab: (t: Tab) => void;
  onAgent: () => void;
  agentOpen: boolean;
  dark: boolean;
  onTheme: () => void;
}

export function Header({ board, tab, onTab, onAgent, agentOpen, dark, onTheme }: Props) {
  const now = new Date(board.now);
  const day = now.toLocaleDateString("ru-RU", { weekday: "short", day: "2-digit", month: "2-digit" });
  return (
    <header className="sticky top-0 z-30 border-b border-line bg-ground/80 backdrop-blur-md">
      <div className="mx-auto flex max-w-[1440px] flex-wrap items-center gap-3 px-4 py-2.5 sm:px-6 md:grid md:grid-cols-[1fr_auto_1fr]">
        <div className="order-1 flex min-w-0 items-center gap-3.5 md:order-none">
          <span className="text-[15px] font-semibold tracking-tight">Пульт<span className="text-signal">.</span></span>
          <span className="num hidden items-center gap-2 rounded-full bg-raised px-2.5 py-1 text-[11px] leading-4 text-ink-2 sm:flex"
            title="Момент, на который построен прогноз">
            <i className="live-dot size-1.5 rounded-full bg-signal" />
            {day} · {hhmm(now)}
          </span>
        </div>

        <nav className="order-3 flex w-full rounded-[10px] bg-raised p-[3px] md:order-none md:w-auto" aria-label="Экраны">
          {TABS.map(([k, label]) => (
            <button key={k} onClick={() => onTab(k)} aria-pressed={tab === k}
              className="relative flex-1 rounded-md px-4 py-1.5 text-[13px] font-medium text-ink-2 transition-colors duration-150 hover:text-ink aria-pressed:text-ink md:flex-none">
              {tab === k && <motion.span layoutId="tab" transition={snap} className="absolute inset-0 rounded-md bg-surface ring-1 ring-inset ring-line" />}
              <span className="relative">{label}</span>
            </button>
          ))}
        </nav>

        <div className="order-2 ml-auto flex items-center gap-2 md:order-none md:justify-self-end">
          <button onClick={onTheme} aria-label={dark ? "Светлая тема" : "Тёмная тема"}
            className="grid size-9 place-items-center overflow-hidden rounded-[10px] text-ink-2 ring-1 ring-inset ring-line transition-colors hover:text-ink">
            <AnimatePresence mode="wait" initial={false}>
              <motion.span key={dark ? "sun" : "moon"} initial={{ rotate: -90, opacity: 0 }} animate={{ rotate: 0, opacity: 1 }}
                exit={{ rotate: 90, opacity: 0 }} transition={fast}>
                {dark ? <Sun size={16} /> : <Moon size={16} />}
              </motion.span>
            </AnimatePresence>
          </button>
          <motion.button onClick={onAgent} aria-pressed={agentOpen} whileTap={{ scale: 0.97 }}
            className="inline-flex h-9 items-center gap-2 rounded-[10px] bg-ink px-3.5 text-[13px] font-semibold text-ground transition-opacity hover:opacity-90">
            <Sparkles size={15} />
            Ассистент
          </motion.button>
        </div>
      </div>
    </header>
  );
}

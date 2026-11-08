import { AnimatePresence, motion } from "motion/react";
import { Bot, Moon, Sun } from "lucide-react";
import type { Board } from "../data";
import { hhmm } from "../format";
import { fast, snap } from "../motion";
import { clock } from "../plan";

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
  const day = now.toLocaleDateString("ru-RU", { weekday: "short", day: "2-digit", month: "2-digit", year: "numeric" });
  return (
    <header className="sticky top-0 z-30 border-b border-line bg-bg/85 backdrop-blur">
      <div className="mx-auto flex max-w-[1440px] flex-wrap items-center gap-3 px-4 py-2.5 sm:px-6 md:grid md:grid-cols-[1fr_auto_1fr]">
        <div className="order-1 flex min-w-0 items-baseline gap-3 md:order-none">
          <span className="text-[15px] font-semibold tracking-tight">Пульт диспетчера</span>
          <span className="num hidden truncate text-xs text-ink-2 sm:inline">
            {day.charAt(0).toUpperCase() + day.slice(1)} · данные на {hhmm(now)}, план до {hhmm(clock(board, 8))}
          </span>
        </div>

        <nav className="order-3 flex w-full rounded-lg bg-line/60 p-0.5 md:order-none md:w-auto" aria-label="Экраны">
          {TABS.map(([k, label]) => (
            <button key={k} onClick={() => onTab(k)} aria-pressed={tab === k}
              className="relative flex-1 rounded-md px-3.5 py-1.5 text-sm font-medium md:flex-none text-ink-2 transition-colors duration-150 aria-pressed:text-ink">
              {tab === k && <motion.span layoutId="tab" transition={snap} className="absolute inset-0 rounded-md bg-card shadow-sm" />}
              <span className="relative">{label}</span>
            </button>
          ))}
        </nav>

        <div className="order-2 ml-auto flex items-center gap-2 md:order-none md:justify-self-end">
          <button onClick={onTheme} aria-label={dark ? "Светлая тема" : "Тёмная тема"}
            className="grid size-9 place-items-center overflow-hidden rounded-lg border border-line text-ink-2 hover:text-ink">
            <AnimatePresence mode="wait" initial={false}>
              <motion.span key={dark ? "sun" : "moon"} initial={{ rotate: -90, opacity: 0 }} animate={{ rotate: 0, opacity: 1 }}
                exit={{ rotate: 90, opacity: 0 }} transition={fast}>
                {dark ? <Sun size={16} /> : <Moon size={16} />}
              </motion.span>
            </AnimatePresence>
          </button>
          <button onClick={onAgent} aria-pressed={agentOpen}
            className="inline-flex h-9 items-center gap-2 rounded-lg bg-ink px-3.5 text-sm font-medium text-bg transition-opacity hover:opacity-90">
            <Bot size={16} />
            Ассистент
          </button>
        </div>
      </div>
    </header>
  );
}

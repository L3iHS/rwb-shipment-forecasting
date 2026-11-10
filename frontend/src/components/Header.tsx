import { AnimatePresence, motion } from "motion/react";
import { Headset, Moon, Sun } from "lucide-react";
import type { Board } from "../data";
import { hhmm } from "../format";
import { fast } from "../motion";

interface Props {
  board: Board;
  onAgent: () => void;
  agentOpen: boolean;
  dark: boolean;
  onTheme: () => void;
}

export function Header({ board, onAgent, agentOpen, dark, onTheme }: Props) {
  const now = new Date(board.now);
  const day = now.toLocaleDateString("ru-RU", { weekday: "short", day: "2-digit", month: "2-digit" });
  return (
    <header className="sticky top-0 z-30 border-b border-line bg-ground/80 backdrop-blur-md">
      <div className="mx-auto flex max-w-[1440px] items-center justify-between gap-3 px-4 py-2.5 sm:px-6">
        <div className="flex min-w-0 items-center gap-3">
          <span className="font-display text-[22px] font-bold leading-7">Пульт<span className="text-signal">.</span></span>
          <span className="num flex items-center gap-2 whitespace-nowrap rounded-full bg-raised px-2.5 py-1 text-[11px] leading-4 text-ink-2"
            title="Момент, на который построен прогноз">
            <i className="live-dot size-1.5 rounded-full bg-signal" />
            <span className="hidden sm:inline">{day}</span> {hhmm(now)}
          </span>
        </div>

        <div className="flex shrink-0 items-center gap-2">
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
            className="inline-flex h-9 items-center gap-2 rounded-[10px] bg-ink px-3.5 text-[15px] font-semibold text-ground transition-opacity hover:opacity-90">
            <Headset size={15} />
            Ассистент
          </motion.button>
        </div>
      </div>
    </header>
  );
}

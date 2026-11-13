import { motion } from "motion/react";
import { FileUp, Headset, Moon, Sun } from "lucide-react";
import { Segmented } from "./Segmented";

export type Tab = "shift" | "file";
const TABS: [Tab, string][] = [["shift", "Прогноз"], ["file", "Свой файл"]];

interface Props {
  onAgent?: () => void; // absent only on the screen where a file is chosen: there is no shift to ask about yet
  agentOpen?: boolean;
  onOtherFile?: () => void; // on a file's shift: back to choosing a file
  dark: boolean;
  onTheme: () => void;
  tab: Tab;
  onTab: (t: Tab) => void;
}

export function Header({ onAgent, agentOpen, onOtherFile, dark, onTheme, tab, onTab }: Props) {
  return (
    <header className="sticky top-0 z-30 border-b border-line bg-ground/80 backdrop-blur-md">
      <div className="mx-auto flex max-w-[1440px] items-center justify-between gap-3 px-4 py-2.5 sm:px-6">
        <div className="flex min-w-0 items-center gap-3">
          <span className="font-display text-[22px] font-bold leading-7">Пульт<span className="text-signal">.</span></span>
          <nav className="whitespace-nowrap sm:ml-3"><Segmented label="Экран" value={tab} options={TABS} onChange={onTab} /></nav>
        </div>

        <div className="flex shrink-0 items-center gap-2">
          {onOtherFile && (
            // On a phone there is no room for the words next to the tabs: the icon stands for them.
            <button onClick={onOtherFile} aria-label="Другой файл"
              className="grid h-9 place-items-center px-2 text-[14px] text-ink-2 underline-offset-4 transition-colors hover:text-ink hover:underline max-sm:w-9 max-sm:rounded-full max-sm:ring-1 max-sm:ring-inset max-sm:ring-line">
              <span className="max-sm:hidden">Другой файл</span>
              <FileUp size={16} className="sm:hidden" />
            </button>
          )}
          <button onClick={onTheme} aria-label={dark ? "Светлая тема" : "Тёмная тема"}
            className="grid size-9 place-items-center rounded-full text-ink-2 ring-1 ring-inset ring-line transition-colors hover:text-ink">
            {dark ? <Sun size={16} /> : <Moon size={16} />}
          </button>
          {onAgent && <motion.button onClick={onAgent} aria-pressed={agentOpen} whileTap={{ scale: 0.97 }}
            aria-label="Ассистент" className="inline-flex h-9 items-center gap-2 rounded-full bg-ink px-2.5 sm:px-3.5 text-[15px] font-semibold text-ground transition-opacity hover:opacity-90">
            <Headset size={15} />
            <span className="max-sm:hidden">Ассистент</span>
          </motion.button>}
        </div>
      </div>
    </header>
  );
}

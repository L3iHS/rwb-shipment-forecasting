import { AnimatePresence, MotionConfig, motion, useReducedMotion } from "motion/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { AgentPanel } from "./components/AgentPanel";
import { bucketOf, queueRows, type Filter } from "./components/CallQueue";
import { Evidence } from "./components/Evidence";
import { Header, type Tab } from "./components/Header";
import { Shift } from "./components/Shift";
import { loadBoard, type Board } from "./data";
import { base } from "./motion";
import { callStep, kpis, leadSteps, type Scenario } from "./plan";
import type { Action } from "./tools";

function useTheme(): [boolean, () => void] {
  const [dark, setDark] = useState(() => {
    const sys = matchMedia("(prefers-color-scheme: dark)").matches;
    try {
      const t = localStorage.getItem("theme");
      return t ? t === "dark" : sys;
    } catch {
      return sys;
    }
  });
  useEffect(() => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    try {
      localStorage.setItem("theme", dark ? "dark" : "light");
    } catch {
      /* storage unavailable */
    }
  }, [dark]);
  return [dark, () => setDark((d) => !d)];
}

export default function App() {
  const [board, setBoard] = useState<Board | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    loadBoard().then(setBoard, (e) => setError(String(e)));
  }, []);
  if (error) return <Centered>Не удалось загрузить данные пульта: {error}</Centered>;
  if (!board) return <Centered>Загружаю прогноз…</Centered>;
  return (
    <MotionConfig reducedMotion="user" transition={base}>
      <Console board={board} />
    </MotionConfig>
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  return <div className="grid h-full place-items-center p-6 text-ink-2">{children}</div>;
}

function Console({ board }: { board: Board }) {
  const reduced = useReducedMotion();
  const [dark, toggleTheme] = useTheme();
  const [tab, setTab] = useState<Tab>("shift");
  const [agentOpen, setAgentOpen] = useState(false);
  const [scenario, setScenario] = useState<Scenario>({ leadH: board.model.lead_h, risk: "p50" });
  const [filter, setFilter] = useState<Filter>("now");
  const lead = leadSteps(board, scenario.leadH);
  const calls = useMemo(() => board.routes.map((r) => callStep(r, lead, scenario.risk)), [board, lead, scenario.risk]);
  const [selected, setSelected] = useState(() => (queueRows(board, calls, "now")[0]?.r ?? board.routes[0]).id);
  const [query, setQuery] = useState("");
  const k = useMemo(() => kpis(board.routes, calls), [board, calls]);

  // Latest scenario and selection for handlers: one assistant reply can carry several actions in a row.
  const live = useRef({ scenario, selected });
  live.current = { scenario, selected };
  const callsNow = () =>
    board.routes.map((r) => callStep(r, leadSteps(board, live.current.scenario.leadH), live.current.scenario.risk));
  const select = (id: number) => {
    live.current.selected = id;
    setSelected(id);
  };

  function openTab(t: Tab) {
    if (t !== tab) window.scrollTo({ top: 0, behavior: reduced ? "auto" : "smooth" });
    setTab(t);
  }

  /** Filter the list; if the open route is not in it, open the top row (done here, not in an effect, so open_route wins). */
  function pick(f: Filter) {
    setQuery("");
    setFilter(f);
    const rows = queueRows(board, callsNow(), f);
    if (rows.length && !rows.some((x) => x.r.id === live.current.selected)) select(rows[0].r.id);
  }

  function act(a: Action) {
    if (a.type === "tab") return openTab(a.value);
    openTab("shift");
    if (a.type === "filter") pick(a.value);
    if (a.type === "scenario") {
      live.current.scenario = { leadH: a.lead_h, risk: a.risk };
      setScenario(live.current.scenario);
    }
    if (a.type === "open_route") {
      const i = board.routes.findIndex((r) => r.id === a.id);
      if (i < 0) return;
      setQuery("");
      setFilter(bucketOf(callsNow()[i]));
      select(a.id);
    }
  }

  return (
    <div className="min-h-full">
      <Header board={board} tab={tab} onTab={openTab} onAgent={() => setAgentOpen((o) => !o)} agentOpen={agentOpen}
        dark={dark} onTheme={toggleTheme} />
      <main className="mx-auto max-w-[1440px] px-4 py-8 sm:px-6">
        <AnimatePresence mode="wait" initial={false}>
          <motion.div key={tab} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, transition: { duration: 0.12 } }} transition={base}>
            {tab === "shift" ? (
              <Shift board={board} calls={calls} k={k} lead={lead} scenario={scenario} onScenario={setScenario} filter={filter}
                onFilter={pick} query={query} onQuery={setQuery} selected={selected} onSelect={select} onTab={openTab} />
            ) : (
              <Evidence board={board} />
            )}
          </motion.div>
        </AnimatePresence>
      </main>
      <AgentPanel open={agentOpen} onClose={() => setAgentOpen(false)} board={board} calls={calls} leadH={scenario.leadH}
        risk={scenario.risk} view={{ tab, route: selected }} onAction={act} />
    </div>
  );
}

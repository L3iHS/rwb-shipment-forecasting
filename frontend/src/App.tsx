import { MotionConfig } from "motion/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { AgentPanel } from "./components/AgentPanel";
import { bucketOf, inFilter, queueRows, type Filter } from "./components/CallQueue";
import { Header } from "./components/Header";
import { Shift } from "./components/Shift";
import { loadBoard, type Board } from "./data";
import { base } from "./motion";
import { callStep, kpis, leadSteps, type Scenario } from "./plan";
import type { Action } from "./tools";

function useTheme(): [boolean, () => void] {
  // Night is the product's default look; day is a choice the user makes and we remember.
  const [dark, setDark] = useState(() => {
    try {
      return localStorage.getItem("theme") !== "light";
    } catch {
      return true;
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

function Console({ board: source }: { board: Board }) {
  // Trucks called from this console, as batches of route ids: one call is a batch of one, "call all" is one big batch,
  // so either is taken back in one step. Nothing is stored: a reload starts the shift from the forecast again.
  const [log, setLog] = useState<number[][]>([]);
  const called = useMemo(() => {
    const n: Record<number, number> = {};
    for (const batch of log) for (const id of batch) n[id] = (n[id] ?? 0) + 1;
    return n;
  }, [log]);
  const total = useMemo(() => log.reduce((sum, batch) => sum + batch.length, 0), [log]);
  // A called truck is on its way, so the rule plans the route's next call around it: the whole console reads this board.
  const board = useMemo<Board>(
    () => (log.length ? { ...source, routes: source.routes.map((r) => (called[r.id] ? { ...r, way: r.way + called[r.id] } : r)) } : source),
    [source, called, log.length],
  );
  const [dark, toggleTheme] = useTheme();
  const [agentOpen, setAgentOpen] = useState(false);
  const [scenario, setScenario] = useState<Scenario>({ leadH: board.model.lead_h, risk: "p50" });
  const [filter, setFilter] = useState<Filter>("now");
  const lead = leadSteps(board, scenario.leadH);
  const calls = useMemo(() => board.routes.map((r) => callStep(r, lead, scenario.risk)), [board, lead, scenario.risk]);
  const [selected, setSelected] = useState(() => (queueRows(board, calls, "now")[0]?.r ?? board.routes[0]).id);
  const [query, setQuery] = useState("");
  const k = useMemo(() => kpis(board.routes, calls), [board, calls]);
  // The plan as the forecast gave it, before any call made here: the timeline shows it as a ghost behind the current one.
  const planned = useMemo(() => source.routes.map((r) => callStep(r, lead, scenario.risk)), [source, lead, scenario.risk]);
  // Every truck the rule asks for right now: a route whose cargo already exceeds its trucks needs more than one.
  const dueNow = useMemo(() => {
    const ids: number[] = [];
    board.routes.forEach((r, i) => {
      if (calls[i] !== 0) return;
      let n = 1;
      while (n < 10 && callStep({ ...r, way: r.way + n }, lead, scenario.risk) === 0) n++;
      for (let j = 0; j < n; j++) ids.push(r.id);
    });
    return ids;
  }, [board, calls, lead, scenario.risk]);

  // Latest scenario and selection for handlers: one assistant reply can carry several actions in a row.
  const live = useRef({ scenario, selected });
  live.current = { scenario, selected };
  const callsNow = () =>
    board.routes.map((r) => callStep(r, leadSteps(board, live.current.scenario.leadH), live.current.scenario.risk));
  const select = (id: number) => {
    live.current.selected = id;
    setSelected(id);
  };

  /** Filter the list; if the open route is not in it, open the top row (done here, not in an effect, so open_route wins). */
  function pick(f: Filter) {
    setQuery("");
    setFilter(f);
    const rows = queueRows(board, callsNow(), f);
    if (rows.length && !rows.some((x) => x.r.id === live.current.selected)) select(rows[0].r.id);
  }

  /** One more truck called for the route. If that takes it out of the list, move on to the next row. */
  function markCalled(id: number) {
    const r = board.routes.find((q) => q.id === id);
    if (!r) return;
    const rows = queueRows(board, calls, filter, query);
    const at = rows.findIndex((x) => x.r.id === id);
    const after = callStep({ ...r, way: r.way + 1 }, lead, scenario.risk);
    const next = rows[at + 1] ?? rows[at - 1];
    if (!query && at >= 0 && next && !inFilter({ ...r, way: r.way + 1 }, after, filter)) select(next.r.id);
    setLog((l) => [...l, [id]]);
  }
  /** Call every truck that is due now, as one step that one undo takes back. */
  function callAll() {
    if (dueNow.length) setLog((l) => [...l, dueNow]);
  }
  /** Take back the latest step (a call or a "call all"), or the latest call of one route. */
  function undoCall(id?: number) {
    setLog((l) => {
      if (id == null) return l.slice(0, -1);
      let at = l.length - 1;
      while (at >= 0 && !l[at].includes(id)) at--;
      if (at < 0) return l;
      const batch = [...l[at]];
      batch.splice(batch.lastIndexOf(id), 1);
      return batch.length ? [...l.slice(0, at), batch, ...l.slice(at + 1)] : [...l.slice(0, at), ...l.slice(at + 1)];
    });
  }

  function act(a: Action) {
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
    <div className="min-h-full overflow-x-clip">
      <Header board={board} onAgent={() => setAgentOpen((o) => !o)} agentOpen={agentOpen} dark={dark} onTheme={toggleTheme} />
      <main className="mx-auto max-w-[1440px] px-4 py-6 sm:px-6 sm:py-8">
        <Shift board={board} calls={calls} k={k} lead={lead} scenario={scenario} onScenario={setScenario} filter={filter}
          onFilter={pick} query={query} onQuery={setQuery} selected={selected} onSelect={select}
          total={total} last={log[log.length - 1]} called={called} planned={planned} due={dueNow.length}
          onCalled={markCalled} onCallAll={callAll} onUndo={undoCall} />
      </main>
      <AgentPanel open={agentOpen} onClose={() => setAgentOpen(false)} board={board} calls={calls} leadH={scenario.leadH}
        risk={scenario.risk} view={{ tab: "shift", route: selected, called }} onAction={act} />
    </div>
  );
}

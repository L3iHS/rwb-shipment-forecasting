import { MotionConfig, motion, useReducedMotion } from "motion/react";
import { useEffect, useMemo, useState } from "react";
import { AgentPanel } from "./components/AgentPanel";
import { FileForecast, useFileRun } from "./components/FileForecast";
import { Header, type Tab } from "./components/Header";
import { Forecast } from "./components/Forecast";
import { loadBoard, type Board } from "./data";
import { base, IntroScope, panel } from "./motion";
import type { Action } from "./tools";
import { fileBoard, overlayOf, type Overlay } from "./upload";

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

/** The open screen, kept in the address (#file) so a link or a reload opens the same one. */
function useTab(): [Tab, (t: Tab) => void] {
  const read = (): Tab => (location.hash === "#file" ? "file" : "shift");
  const [tab, setTab] = useState<Tab>(read);
  useEffect(() => {
    const on = () => setTab(read());
    addEventListener("hashchange", on);
    return () => removeEventListener("hashchange", on);
  }, []);
  return [tab, (t: Tab) => {
    history.replaceState(null, "", t === "file" ? "#file" : location.pathname + location.search);
    setTab(t);
    scrollTo({ top: 0 });
  }];
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
      <Root board={board} />
    </MotionConfig>
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  return <div className="grid h-full place-items-center p-6 text-ink-2">{children}</div>;
}

/** Whether a media query matches, kept up to date. */
function useMedia(query: string): boolean {
  const [matches, setMatches] = useState(() => matchMedia(query).matches);
  useEffect(() => {
    const mq = matchMedia(query);
    const update = () => setMatches(mq.matches);
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, [query]);
  return matches;
}

const AGENT_WIDTH = 420; // px, the assistant panel (see AgentPanel)
const BACK = 8; // steps of history on the network chart (Forecast.tsx)

interface Chrome {
  dark: boolean;
  onTheme: () => void;
  tab: Tab;
  onTab: (t: Tab) => void;
  onOtherFile?: () => void;
}

/**
 * Two dashboards: the console's own forecast, and the forecast from an uploaded file, shown the same way.
 * Both stay mounted while the other tab is open, so each keeps its open route.
 */
function Root({ board }: { board: Board }) {
  const [dark, onTheme] = useTheme();
  const [tab, onTab] = useTab();
  const file = useFileRun();
  const chrome: Chrome = { dark, onTheme, tab, onTab };
  const run = file.state.status === "done" ? file.state.run : null;
  const fb = useMemo(() => (run ? fileBoard(board, run) : null), [board, run]);
  const overlay = useMemo(() => (run ? overlayOf(run, BACK) : undefined), [run]);
  return (
    <>
      {/* The two dashboards are the same console and behave the same: the same entrance, the same assistant panel.
          Only the highlight colour differs (.from-file), on purpose. */}
      <div hidden={tab !== "shift"} className="tab h-full">
        <IntroScope.Provider value="shift">
          <Console board={board} chrome={chrome} />
        </IntroScope.Provider>
      </div>
      <div hidden={tab !== "file"} className="from-file tab h-full">
        {fb && file.state.status === "done" ? (
          <IntroScope.Provider value={`file:${file.state.name}-${run!.now}`}>
            <Console key={`${file.state.name}-${run!.now}`} board={fb} overlay={overlay} chrome={{ ...chrome, onOtherFile: file.reset }}
              offline="по своему файлу отвечаю без модели: она знает только прогноз пульта" />
          </IntroScope.Provider>
        ) : (
          <div className="dotted min-h-full overflow-x-clip">
            <Header {...chrome} />
            <main className="@container mx-auto max-w-[1440px] px-4 py-6 sm:px-6 sm:py-8">
              <FileForecast state={file.state} onFile={(f) => file.send(f, f.name)} onDemo={file.demo} />
            </main>
          </div>
        )}
      </div>
    </>
  );
}

/** `offline`: why the assistant here answers without the model (a file's forecast is not what the server's tools see). */
function Console({ board, overlay, chrome, offline }: { board: Board; overlay?: Overlay; chrome: Chrome; offline?: string }) {
  const reduced = useReducedMotion();
  const wide = useMedia("(min-width: 1280px)"); // room for the assistant beside a console that keeps its layout; below it overlays
  const [agentOpen, setAgentOpen] = useState(false);
  // The route open in the card: the largest by volume at first, or the one the file changed most.
  const [selected, setSelected] = useState(() => firstRoute(board, overlay));

  function act(a: Action) {
    if (a.type === "open_route" && board.routes.some((r) => r.id === a.id)) setSelected(a.id);
  }

  return (
    // With the assistant open on a wide screen the console moves over and re-lays itself in the room that is left;
    // the charts follow the width as it changes.
    <motion.div className="dotted min-h-full overflow-x-clip" initial={false}
      animate={{ paddingRight: agentOpen && wide ? AGENT_WIDTH : 0 }} transition={reduced ? { duration: 0 } : panel}>
      <Header {...chrome} onAgent={() => setAgentOpen((o) => !o)} agentOpen={agentOpen} />
      <main className="@container mx-auto max-w-[1440px] px-4 py-6 sm:px-6 sm:py-8">
        <Forecast board={board} overlay={overlay} selected={selected} onSelect={setSelected} />
      </main>
      <AgentPanel open={agentOpen} onClose={() => setAgentOpen(false)} board={board}
        view={{ tab: "shift", route: selected }} onAction={act} offline={offline} />
    </motion.div>
  );
}

function firstRoute(board: Board, overlay?: Overlay): number {
  const last = (a: number[]) => a[a.length - 1];
  const score = (i: number) => {
    const b = overlay?.baseRoutes?.[i];
    return b ? Math.abs(last(board.routes[i].c50) / Math.max(1, last(b.c50)) - 1) : last(board.routes[i].c50);
  };
  let best = 0;
  for (let i = 1; i < board.routes.length; i++) if (score(i) > score(best)) best = i;
  return board.routes[best].id;
}

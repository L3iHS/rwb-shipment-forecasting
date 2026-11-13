import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { Download, LoaderCircle } from "lucide-react";
import { useCallback, useRef, useState } from "react";
import { base, useIntro } from "../motion";
import { DEMOS, forecastFile, type FileRun } from "../upload";

export type FileState =
  | { status: "idle"; error?: string }
  | { status: "busy"; name: string }
  | { status: "done"; name: string; run: FileRun };

/** The tab's state lives in the console, so switching tabs keeps the last result. */
export function useFileRun() {
  const [state, setState] = useState<FileState>({ status: "idle" });
  const send = useCallback(async (file: Blob, name: string) => {
    setState({ status: "busy", name });
    try {
      setState({ status: "done", name, run: await forecastFile(file) });
    } catch (e) {
      setState({ status: "idle", error: e instanceof Error ? e.message : String(e) });
    }
  }, []);
  const demo = useCallback(async (path: string) => {
    const name = path.split("/").pop()!;
    setState({ status: "busy", name });
    try {
      const res = await fetch(`${import.meta.env.BASE_URL}${path}`);
      if (!res.ok) throw new Error(`Пример не найден: ${res.status}`);
      await send(await res.blob(), name);
    } catch (e) {
      setState({ status: "idle", error: e instanceof Error ? e.message : String(e) });
    }
  }, [send]);
  return { state, send, demo, reset: () => setState({ status: "idle" }) };
}

const item = { hidden: { opacity: 0, y: 10 }, show: { opacity: 1, y: 0, transition: base } };

/** Screen "Свой файл" before a result: the two demos and a file of your own. The result is the shift screen itself. */
export function FileForecast({ state, onFile, onDemo }: {
  state: FileState;
  onFile: (f: File) => void;
  onDemo: (path: string) => void;
}) {
  const reduced = useReducedMotion();
  const intro = useIntro("file");
  return (
    <motion.div initial={intro ? "hidden" : false} animate="show" variants={{ show: { transition: { staggerChildren: reduced ? 0 : 0.07 } } }}>
      <Pick state={state} onFile={onFile} onDemo={onDemo} />
    </motion.div>
  );
}

function Pick({ state, onFile, onDemo }: { state: FileState; onFile: (f: File) => void; onDemo: (path: string) => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const busy = state.status === "busy";
  return (
    <div className="space-y-8">
      <motion.h1 variants={item} className="font-display text-[28px] font-semibold leading-9">Прогноз по своему файлу</motion.h1>

      <motion.section variants={item}>
        {DEMOS.map((d) => (
          <div key={d.key} className="flex items-center justify-between gap-4 border-b border-line py-4">
            <div className="min-w-0">
              <h2 className="font-display text-[22px] font-semibold leading-7">{d.title}</h2>
              <p className="mt-0.5 text-ink-2">{d.note}</p>
            </div>
            <div className="flex shrink-0 flex-wrap items-center justify-end gap-x-3 gap-y-1">
              <a href={`${import.meta.env.BASE_URL}${d.file}`} download
                className="inline-flex h-10 items-center gap-1.5 rounded-full px-3 text-ink-2 transition-colors hover:text-ink">
                <Download size={15} />
                <span className="num text-[13px]">{d.file.split("/").pop()}</span>
              </a>
              <button onClick={() => onDemo(d.file)} disabled={busy}
                className="inline-flex h-10 items-center rounded-full px-5 text-[15px] font-semibold text-ink ring-1 ring-inset ring-line transition-colors hover:bg-raised disabled:opacity-40">
                Посчитать
              </button>
            </div>
          </div>
        ))}
      </motion.section>

      <motion.section variants={item}>
        <div
          onDragOver={(e) => { e.preventDefault(); setOver(true); }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setOver(false);
            const f = e.dataTransfer.files[0];
            if (f && !busy) onFile(f);
          }}
          className={`flex h-14 items-center justify-between gap-4 rounded-[22px] border border-dashed px-4 transition-colors duration-150 ${
            over ? "border-ink bg-raised" : "border-line"}`}>
          {busy ? (
            <span className="flex min-w-0 items-center gap-2 text-ink-2">
              <LoaderCircle size={16} className="shrink-0 animate-spin" />
              <span className="num truncate text-[13px]">{state.name}</span>
            </span>
          ) : (
            <span className="min-w-0 truncate text-ink-2">Свой файл: .csv или .parquet, до 16 МБ</span>
          )}
          <button onClick={() => input.current?.click()} disabled={busy}
            className="inline-flex h-9 shrink-0 items-center rounded-full bg-ink px-5 text-[15px] font-semibold text-ground transition-opacity hover:opacity-90 disabled:opacity-40">
            Выбрать
          </button>
          <input ref={input} type="file" accept=".csv,.parquet,text/csv" className="hidden"
            onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ""; if (f) onFile(f); }} />
        </div>
        <AnimatePresence>
          {state.status === "idle" && state.error && (
            <motion.p initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={base}
              className="mt-3 w-fit rounded-[16px] bg-alert-soft px-3.5 py-2 text-[14px] text-ink" role="alert">
              {state.error}
            </motion.p>
          )}
        </AnimatePresence>
        <p className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-[13px] text-muted">
          <span className="num">route_id, timestamp, status_1 … status_6, target_1h</span>
          <a href={`${import.meta.env.BASE_URL}demo/obrazec.csv`} download
            className="inline-flex items-center gap-1.5 text-ink-2 underline-offset-4 transition-colors hover:text-ink hover:underline">
            <Download size={14} />
            Скачать образец
          </a>
        </p>
      </motion.section>
    </div>
  );
}

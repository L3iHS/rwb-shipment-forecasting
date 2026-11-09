import { AnimatePresence, motion } from "motion/react";
import type { Board } from "../data";
import { leadLabel } from "../format";
import { base, fast } from "../motion";
import type { Risk, Scenario } from "../plan";
import { whatIf } from "../tools";
import { Segmented } from "./Segmented";

const LEADS = [0.5, 1, 2].map((h): [number, string] => [h, leadLabel(h)]);
const RISKS: [Risk, string][] = [["p50", "обычный"], ["p90", "осторожный"]];

/** Conditions the plan depends on; once changed, shows how the "call now" count moved and a reset. */
export function WhatIf({ board, s, onChange }: { board: Board; s: Scenario; onChange: (s: Scenario) => void }) {
  const changed = s.leadH !== board.model.lead_h || s.risk !== "p50";
  const w = changed ? whatIf(board, s.leadH, s.risk) : null;
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
      <label className="flex items-center gap-2 text-xs text-ink-2">Машина едет
        <Segmented value={s.leadH} options={LEADS} onChange={(leadH) => onChange({ ...s, leadH })} />
      </label>
      <label className="flex items-center gap-2 text-xs text-ink-2">Прогноз
        <Segmented value={s.risk} options={RISKS} onChange={(risk) => onChange({ ...s, risk })} />
      </label>
      <AnimatePresence mode="wait" initial={false}>
        {w && (
          <motion.span key="result" initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, transition: fast }} transition={base} className="text-xs text-muted">
            сейчас <span className="num font-medium text-ink">{w.call_now_base} → {w.call_now}</span>
            <button onClick={() => onChange({ leadH: board.model.lead_h, risk: "p50" })}
              className="ml-2 font-medium text-ink underline decoration-line underline-offset-4 hover:decoration-ink">
              сбросить
            </button>
          </motion.span>
        )}
      </AnimatePresence>
    </div>
  );
}

import { motion } from "motion/react";
import { RotateCcw } from "lucide-react";
import type { Board } from "../data";
import { leadLabel } from "../format";
import { base } from "../motion";
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
    <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-[14px] text-ink-2">
      <div className="flex items-center gap-2.5">Машина едет
        <Segmented label="Машина едет" value={s.leadH} options={LEADS} onChange={(leadH) => onChange({ ...s, leadH })} />
      </div>
      <div className="flex items-center gap-2.5">Прогноз
        <Segmented label="Прогноз" value={s.risk} options={RISKS} onChange={(risk) => onChange({ ...s, risk })} />
        {/* Always in the layout, shown only for a changed scenario: appearing must not push the controls around. */}
        <motion.button onClick={() => onChange({ leadH: board.model.lead_h, risk: "p50" })} disabled={!w} aria-hidden={!w}
          initial={false} animate={{ opacity: w ? 1 : 0, scale: w ? 1 : 0.9 }} transition={base}
          className="grid size-[30px] place-items-center rounded-[10px] text-ink-2 ring-1 ring-inset ring-line hover:text-ink disabled:pointer-events-none"
          title={w ? `Вернуть исходный сценарий: срочных было ${w.call_now_base}, стало ${w.call_now}` : undefined} aria-label="Вернуть исходный сценарий">
          <RotateCcw size={14} />
        </motion.button>
      </div>
    </div>
  );
}

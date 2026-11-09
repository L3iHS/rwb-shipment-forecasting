import { motion, useReducedMotion } from "motion/react";
import { useState } from "react";
import type { Board, RuleKey } from "../data";
import { dec1, int, leadLabel, pct, signedPct } from "../format";
import { flow, useFirstRender } from "../motion";
import { Disclosure } from "./Disclosure";
import { Segmented } from "./Segmented";
import { Ticker } from "./Ticker";

const RULES: [RuleKey, string][] = [
  ["reactive", "Вызов, когда кузов полон"],
  ["last_2h_pace", "По темпу последних 2 часов"],
  ["same_time_last_week", "Как в это время неделю назад"],
  ["model_p50", "Модель"],
];
const TABLE_RULES: [RuleKey, string][] = [...RULES, ["oracle", "Идеальный прогноз (предел)"]];
// Rules that also forecast the inflow; "call when the body is full" never leaves a truck under-filled, it makes cargo wait.
const FORECAST_RULES: RuleKey[] = ["last_2h_pace", "same_time_last_week"];

const day = (s: string) => new Date(s).toLocaleDateString("ru-RU", { day: "2-digit", month: "2-digit" });

/** Screen "Почему верить": the measured effect of the call rule on a week the model never saw. */
export function Evidence({ board }: { board: Board }) {
  const reduced = useReducedMotion();
  const first = useFirstRender();
  const bt = board.backtest;
  const [lead, setLead] = useState(board.model.lead_h);
  const [fill, setFill] = useState(board.model.fill_hours);
  const row = bt.sensitivity.find((s) => s.lead_h === lead && s.fill_hours === fill)!;
  const max = Math.max(...RULES.map(([k]) => row[k]));
  const partial = FORECAST_RULES.map((k) => bt.per_day[k].partial);
  const maxBias = Math.max(...board.monitor.map((m) => Math.abs(m.bias)));
  const proofs = [
    ["Недогруз", `${int(bt.per_day.model_p50.partial)} машин в день`,
      `против ${int(Math.min(...partial))}–${int(Math.max(...partial))} у простых правил`],
    ["Диапазоны честные", `${pct(bt.intervals["80%"])} из 80%`, `${pct(bt.intervals["90%"])} из 90%`],
    ["Самоконтроль", `${board.monitor.length} дней без тревог`, `перекос суммы до ${pct(maxBias)}`],
  ];

  return (
    <div className="space-y-10">
      <section>
        <div className="eyebrow">Неделя {day(bt.week[0])}–{day(bt.week[1])}, которую модель не видела</div>
        <h1 className="mt-2 text-[28px] font-semibold leading-tight tracking-tight sm:text-[34px]">
          С моделью машина теряет <span className="num text-signal"><Ticker value={row.model_p50} format={dec1} /></span> мин
          вместо <span className="num"><Ticker value={row.reactive} format={dec1} /></span>
        </h1>
        <p className="mt-2 text-ink-2">Простой у ворот плюс ожидание груза, на одну машину.</p>
      </section>

      <section>
        <div className="flex flex-wrap gap-x-5 gap-y-2">
          <label className="flex items-center gap-2 text-xs text-ink-2">Машина едет
            <Segmented value={lead} options={[0.5, 1, 2].map((h): [number, string] => [h, leadLabel(h)])} onChange={setLead} />
          </label>
          <label className="flex items-center gap-2 text-xs text-ink-2">Кузов за
            <Segmented value={fill} options={[1, 2, 4].map((h): [number, string] => [h, `${h} ч`])} onChange={setFill} />
          </label>
        </div>
        <div className="mt-5 space-y-2.5">
          {RULES.map(([k, name], i) => (
            <div key={k} className="grid grid-cols-[minmax(140px,240px)_1fr_56px] items-center gap-3 text-sm">
              <span className={k === "model_p50" ? "font-semibold" : "text-ink-2"}>{name}</span>
              <span className="h-5 overflow-hidden rounded bg-line/50">
                <motion.span initial={{ scaleX: 0 }} animate={{ scaleX: Math.max(row[k] / max, 0.006) }}
                  transition={reduced ? { duration: 0 } : { ...flow, delay: first ? i * 0.05 : 0 }}
                  className={`block h-full origin-left rounded ${k === "model_p50" ? "bg-signal" : "bg-ink-2/40"}`} />
              </span>
              <span className={`num text-right ${k === "model_p50" ? "font-semibold" : ""}`}>
                <Ticker value={row[k]} format={dec1} />
              </span>
            </div>
          ))}
        </div>
        <p className="mt-3 text-xs text-muted">
          Модель лучше во всех {bt.sensitivity.length} сочетаниях.
        </p>
      </section>

      <section className="grid gap-6 sm:grid-cols-3">
        {proofs.map(([title, value, note]) => (
          <div key={title}>
            <div className="eyebrow">{title}</div>
            <div className="num mt-1 text-xl font-semibold">{value}</div>
            <div className="mt-0.5 text-xs text-ink-2">{note}</div>
          </div>
        ))}
      </section>

      <section>
        <Disclosure title="Таблицы проверки">
          <div className="grid gap-8 lg:grid-cols-2">
            <div>
              <h3 className="text-sm font-semibold">Машины за день</h3>
              <table className="num mt-3 w-full text-sm">
                <thead className="text-left text-[11px] text-muted">
                  <tr><th className="pb-2 font-normal">Правило</th><th className="pb-2 text-right font-normal">Простой, ч</th>
                    <th className="pb-2 text-right font-normal">Груз ждёт, ч</th><th className="pb-2 text-right font-normal">Недогруз, машин</th></tr>
                </thead>
                <tbody>
                  {TABLE_RULES.map(([k, name]) => {
                    const r = bt.per_day[k];
                    return (
                      <tr key={k} className={`border-t border-line ${k === "model_p50" ? "font-semibold" : ""}`}>
                        <td className="py-2 font-sans">{name}</td>
                        <td className="py-2 text-right">{int(r.idle_h)}</td>
                        <td className="py-2 text-right">{int(r.late_h)}</td>
                        <td className="py-2 text-right">{int(r.partial)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <div>
              <h3 className="text-sm font-semibold">Журнал качества</h3>
              <table className="num mt-3 w-full text-sm">
                <thead className="text-left text-[11px] text-muted">
                  <tr><th className="pb-2 font-normal">День</th><th className="pb-2 text-right font-normal">WAPE</th>
                    <th className="pb-2 text-right font-normal">Перекос суммы</th><th className="pb-2 text-right font-normal">В 80% диапазоне</th></tr>
                </thead>
                <tbody>
                  {board.monitor.map((m) => (
                    <tr key={m.day} className="border-t border-line">
                      <td className="py-1.5">{m.day}</td>
                      <td className="py-1.5 text-right">{m.wape.toFixed(3)}</td>
                      <td className={`py-1.5 text-right ${Math.abs(m.bias) > 0.02 ? "text-warn" : ""}`}>{signedPct(m.bias)}</td>
                      <td className="py-1.5 text-right">{pct(m.cov)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </Disclosure>
      </section>
    </div>
  );
}

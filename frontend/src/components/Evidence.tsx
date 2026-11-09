import { motion, useReducedMotion } from "motion/react";
import { Check } from "lucide-react";
import { useState, type ReactNode } from "react";
import type { Board, RuleKey } from "../data";
import { dec1, int, leadLabel, pct, signedPct } from "../format";
import { base, flow, useFirstRender } from "../motion";
import { useWidth } from "./charts";
import { Disclosure } from "./Disclosure";
import { Segmented } from "./Segmented";
import { Ticker } from "./Ticker";

const RULES: [RuleKey, string][] = [
  ["reactive", "Вызов, когда кузов полон"],
  ["last_2h_pace", "По темпу последних 2 часов"],
  ["same_time_last_week", "Как неделю назад"],
  ["model_p50", "Модель"],
];
const TABLE_RULES: [RuleKey, string][] = [...RULES, ["oracle", "Идеальный прогноз (предел)"]];
const LEADS = [0.5, 1, 2], FILLS = [1, 2, 4];

const day = (s: string) => new Date(s).toLocaleDateString("ru-RU", { day: "2-digit", month: "2-digit" });
const min1 = (v: number) => v.toLocaleString("ru-RU", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const item = { hidden: { opacity: 0, y: 10 }, show: { opacity: 1, y: 0, transition: base } };

function Tile({ title, value, note, children }: { title: string; value: ReactNode; note: string; children: ReactNode }) {
  return (
    <motion.section variants={item} className="panel flex flex-col p-5">
      <div className="eyebrow">{title}</div>
      <div className="mt-2 text-[26px] font-semibold leading-8 tracking-[-0.02em]">{value}</div>
      <div className="mt-0.5 text-[13px] text-ink-2">{note}</div>
      <div className="mt-5 flex flex-1 flex-col justify-end">{children}</div>
    </motion.section>
  );
}

/** A horizontal bar that grows from the left; the model's is the only coloured one. */
function Bar({ share, model, delay, thin }: { share: number; model?: boolean; delay: number; thin?: boolean }) {
  const reduced = useReducedMotion();
  return (
    <span className={`block overflow-hidden rounded-r-[5px] ${thin ? "h-2.5" : "h-5"}`}>
      <motion.span initial={{ scaleX: 0 }} animate={{ scaleX: Math.max(share, 0.004) }}
        transition={reduced ? { duration: 0 } : { ...flow, delay }}
        className={`block h-full origin-left rounded-r-[5px] ${model ? "bg-signal" : "bg-ink-2/45"}`} />
    </span>
  );
}

/** Share of facts inside the 80 % band, day by day, against the promised 80 %. */
function CoverageDots({ days }: { days: Board["monitor"] }) {
  const [ref, W] = useWidth<HTMLDivElement>();
  const H = 96, L = 34, R = 10, top = 8, B = 70;
  const y = (v: number) => B - ((v - 0.7) / 0.2) * (B - top);
  const x = (i: number) => L + ((W - L - R) * (i + 0.5)) / days.length;
  return (
    <div ref={ref}>
      {W > 0 && (
        <svg width={W} height={H} role="img" aria-label="Попадание в коридор 80 % по дням">
          {[0.7, 0.9].map((v) => (
            <g key={v}>
              <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke="var(--line)" />
              <text className="tick" x={L - 8} y={y(v) + 4} textAnchor="end">{v * 100}%</text>
            </g>
          ))}
          <line x1={L} x2={W - R} y1={y(0.8)} y2={y(0.8)} stroke="var(--ink-2)" strokeDasharray="6 4" />
          <text className="mark" x={L - 8} y={y(0.8) + 4} textAnchor="end">80%</text>
          {days.map((m, i) => (
            <g key={m.day}>
              <motion.circle cx={x(i)} r={5} fill="var(--forecast)" stroke="var(--surface)" strokeWidth={2}
                initial={{ cy: y(0.8), opacity: 0 }} animate={{ cy: y(m.cov), opacity: 1 }} transition={{ ...flow, delay: 0.3 + i * 0.05 }}>
                <title>{`${m.day}: ${pct(m.cov)} в коридоре, перекос суммы ${signedPct(m.bias)}`}</title>
              </motion.circle>
              <text className="tick" x={x(i)} y={H - 4} textAnchor="middle">{m.day.slice(0, 2)}</text>
            </g>
          ))}
        </svg>
      )}
    </div>
  );
}

/** Screen "Почему верить": the measured effect of the call rule on a week the model never saw. */
export function Evidence({ board }: { board: Board }) {
  const reduced = useReducedMotion();
  const first = useFirstRender();
  const bt = board.backtest;
  const [lead, setLead] = useState(board.model.lead_h);
  const [fill, setFill] = useState(board.model.fill_hours);
  const row = bt.sensitivity.find((s) => s.lead_h === lead && s.fill_hours === fill)!;
  const max = Math.max(...RULES.map(([k]) => row[k]));
  const wins = (s: (typeof bt.sensitivity)[number]) => RULES.every(([k]) => k === "model_p50" || s.model_p50 < s[k]);
  const won = bt.sensitivity.filter(wins).length;
  const hours = RULES.map(([k, name]) => ({ k, name, idle: bt.per_day[k].idle_h, late: bt.per_day[k].late_h }));
  const maxHours = Math.max(...hours.map((h) => h.idle + h.late));
  const partial = RULES.map(([k, name]) => ({ k, name, v: bt.per_day[k].partial }));
  const maxPartial = Math.max(...partial.map((p) => p.v));
  const others = partial.filter((p) => p.k !== "model_p50" && p.v > 0).map((p) => p.v);
  const maxBias = Math.max(...board.monitor.map((m) => Math.abs(m.bias)));
  const bands = (["50%", "80%", "90%"] as const).filter((b) => bt.intervals[b] != null);
  const delay = (i: number) => (first ? 0.25 + i * 0.07 : 0);

  return (
    <motion.div className="space-y-8" initial="hidden" animate="show" variants={{ show: { transition: { staggerChildren: reduced ? 0 : 0.07 } } }}>
      <motion.section variants={item} className="flex flex-wrap items-end justify-between gap-x-10 gap-y-6">
        <div className="flex items-end gap-4 sm:gap-6">
          <span className="text-[72px] font-bold leading-[0.82] tracking-[-0.04em] text-signal sm:text-[96px]">
            <Ticker value={row.model_p50} format={min1} from={row.reactive} />
          </span>
          <div className="pb-0.5">
            <h1 className="text-[24px] font-semibold leading-[1.1] tracking-[-0.02em] sm:text-[34px]">
              минут теряет машина<br />вместо <Ticker value={row.reactive} format={min1} />
            </h1>
            <p className="mt-2 text-ink-2">Неделя {day(bt.week[0])}–{day(bt.week[1])}, которую модель не видела</p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2.5">
          <div className="flex items-center gap-2.5 text-[13px] text-ink-2">Машина едет
            <Segmented label="Машина едет" value={lead} options={LEADS.map((h): [number, string] => [h, leadLabel(h)])} onChange={setLead} />
          </div>
          <div className="flex items-center gap-2.5 text-[13px] text-ink-2">Кузов за
            <Segmented label="Кузов набирается за" value={fill} options={FILLS.map((h): [number, string] => [h, `${h} ч`])} onChange={setFill} />
          </div>
        </div>
      </motion.section>

      <motion.section variants={item} className="panel grid gap-8 p-5 sm:p-6 lg:grid-cols-[1fr_220px]">
        <div>
          <h2 className="text-[18px] font-semibold leading-6 tracking-tight">
            Потери на машину <span className="ml-1.5 text-[13px] font-normal tracking-normal text-muted">простой у ворот и ожидание груза, мин</span>
          </h2>
          <div className="mt-5 space-y-3.5">
            {RULES.map(([k, name], i) => (
              <div key={k} className="grid grid-cols-[minmax(120px,230px)_1fr_52px] items-center gap-4 text-[13px] sm:text-[15px]">
                <span className={k === "model_p50" ? "font-semibold" : "text-ink-2"}>{name}</span>
                <Bar share={row[k] / max} model={k === "model_p50"} delay={delay(i)} />
                <span className={`num text-right ${k === "model_p50" ? "font-semibold" : "text-ink-2"}`}><Ticker value={row[k]} format={min1} /></span>
              </div>
            ))}
          </div>
        </div>
        <div className="flex items-center gap-5 border-line lg:flex-col lg:items-start lg:justify-center lg:border-l lg:pl-8">
          {/* Every combination of lead time (rows) and body size (columns): a filled dot is one the model wins. */}
          <div className="grid grid-cols-3 gap-2" role="img" aria-label={`Модель лучше в ${won} из ${bt.sensitivity.length} сочетаний`}>
            {LEADS.flatMap((l) => FILLS.map((fh) => {
              const s = bt.sensitivity.find((q) => q.lead_h === l && q.fill_hours === fh);
              const on = l === lead && fh === fill;
              return (
                <motion.button key={`${l}-${fh}`} onClick={() => { setLead(l); setFill(fh); }}
                  title={`Едет ${leadLabel(l)}, кузов за ${fh} ч${s ? `: ${dec1(s.model_p50)} мин` : ""}`}
                  initial={{ scale: 0 }} animate={{ scale: 1 }} whileHover={{ scale: 1.15 }}
                  transition={{ type: "spring", stiffness: 500, damping: 28, delay: first ? 0.5 + (LEADS.indexOf(l) * 3 + FILLS.indexOf(fh)) * 0.04 : 0 }}
                  className={`size-5 rounded-full ${s && wins(s) ? "bg-signal" : "bg-line"} ${on ? "ring-2 ring-ink ring-offset-2 ring-offset-surface" : ""}`} />
              );
            }))}
          </div>
          <div>
            <div className="text-[26px] font-semibold leading-8 tracking-[-0.02em]"><span className="num">{won}</span> из <span className="num">{bt.sensitivity.length}</span></div>
            <div className="text-[13px] text-ink-2">сценариев выигрывает модель</div>
          </div>
        </div>
      </motion.section>

      <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
        <Tile title="Кто кого ждёт" value={<>{int(bt.per_day.model_p50.idle_h + bt.per_day.model_p50.late_h)} ч</>} note="в день с моделью, по всей сети">
          <div className="space-y-2.5">
            {hours.map((h, i) => (
              <div key={h.k} className="grid grid-cols-[1fr] gap-1" title={`${h.name}: машины ждут ${int(h.idle)} ч, груз ждёт ${int(h.late)} ч`}>
                <span className={`text-xs ${h.k === "model_p50" ? "font-semibold text-ink" : "text-ink-2"}`}>{h.name}</span>
                <span className="flex h-2.5 gap-0.5">
                  <motion.i className="block rounded-[3px] bg-forecast" initial={{ width: 0 }} animate={{ width: `${(h.idle / maxHours) * 100}%` }}
                    transition={{ ...flow, delay: delay(i) }} />
                  <motion.i className="block rounded-[3px] bg-ink-2/45" initial={{ width: 0 }} animate={{ width: `${(h.late / maxHours) * 100}%` }}
                    transition={{ ...flow, delay: delay(i) + 0.05 }} />
                </span>
              </div>
            ))}
            <div className="flex gap-4 pt-1 text-[11px] text-muted">
              <span className="flex items-center gap-1.5"><i className="size-2 rounded-[2px] bg-forecast" />машина ждёт груз</span>
              <span className="flex items-center gap-1.5"><i className="size-2 rounded-[2px] bg-ink-2/45" />груз ждёт машину</span>
            </div>
          </div>
        </Tile>

        <Tile title="Недогруз" value={<>{int(bt.per_day.model_p50.partial)} {" "}<span className="text-[15px] font-normal tracking-normal text-ink-2">машин в день</span></>}
          note={others.length ? `против ${int(Math.min(...others))}–${int(Math.max(...others))} у простых прогнозов` : "уехали неполными"}>
          <div className="space-y-2.5">
            {partial.map((p, i) => (
              <div key={p.k} className="grid grid-cols-[1fr_28px] items-end gap-x-3 gap-y-1">
                <span className={`col-span-2 text-xs ${p.k === "model_p50" ? "font-semibold text-ink" : "text-ink-2"}`}>{p.name}</span>
                <Bar thin share={p.v / maxPartial} model={p.k === "model_p50"} delay={delay(i)} />
                <span className="num -mb-0.5 text-right text-xs text-ink-2">{int(p.v)}</span>
              </div>
            ))}
          </div>
        </Tile>

        <Tile title="Коридоры честные" value={<>{pct(bt.intervals["80%"])}</>} note="фактов попало в коридор 80 %">
          <div className="space-y-4">
            {bands.map((b, i) => {
              const target = parseInt(b) / 100;
              return (
                <div key={b} className="grid grid-cols-[34px_1fr_48px] items-center gap-3 text-xs" title={`Обещано ${b}, сбылось ${pct(bt.intervals[b])}`}>
                  <span className="num text-ink-2">{b}</span>
                  <span className="relative h-2.5 rounded-[3px] bg-line">
                    <motion.i className="absolute inset-y-0 left-0 rounded-[3px] bg-forecast" initial={{ width: 0 }}
                      animate={{ width: `${bt.intervals[b] * 100}%` }} transition={{ ...flow, delay: delay(i) }} />
                    <i className="absolute -inset-y-1 w-0.5 rounded-full bg-ink" style={{ left: `${target * 100}%` }} />
                  </span>
                  <span className="num text-right text-ink">{pct(bt.intervals[b])}</span>
                </div>
              );
            })}
            <div className="flex items-center gap-1.5 text-[11px] text-muted"><i className="h-3 w-0.5 rounded-full bg-ink" />обещано</div>
          </div>
        </Tile>

        <Tile title="Самоконтроль"
          value={board.alerts.length ? <>{board.alerts.length} тревог</> : <span className="flex items-center gap-2"><Check size={22} className="text-good" />{board.monitor.length} дней без тревог</span>}
          note={`перекос суммы не больше ${pct(maxBias)}`}>
          <CoverageDots days={board.monitor} />
        </Tile>
      </div>

      <motion.section variants={item}>
        <Disclosure title="Таблицы проверки">
          <div className="grid gap-8 lg:grid-cols-2">
            <div>
              <h3 className="text-sm font-semibold">Машины за день</h3>
              <table className="num mt-3 w-full text-[13px]">
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
              <table className="num mt-3 w-full text-[13px]">
                <thead className="text-left text-[11px] text-muted">
                  <tr><th className="pb-2 font-normal">День</th><th className="pb-2 text-right font-normal">WAPE</th>
                    <th className="pb-2 text-right font-normal">Перекос суммы</th><th className="pb-2 text-right font-normal">В коридоре 80 %</th></tr>
                </thead>
                <tbody>
                  {board.monitor.map((m) => (
                    <tr key={m.day} className="border-t border-line">
                      <td className="py-1.5">{m.day}</td>
                      <td className="py-1.5 text-right">{m.wape.toFixed(3)}</td>
                      <td className="py-1.5 text-right">{signedPct(m.bias)}</td>
                      <td className="py-1.5 text-right">{pct(m.cov)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </Disclosure>
      </motion.section>
    </motion.div>
  );
}

// Charts of the console, drawn by hand in SVG so every mark can move: lines draw in, curves morph
// between routes, bars re-flow when the scenario changes. Rules: docs of the "Ночная смена" system.
import { motion, useReducedMotion } from "motion/react";
import { Truck } from "lucide-react";
import { useId, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { vol } from "../format";
import { ease, flow, useFirstRender } from "../motion";

type Pt = [number, number];
/** Below this width there is no room for the axes: draw nothing rather than marks of negative size. */
const MIN_W = 140;
const f = (v: number) => v.toFixed(1);
const tickNum = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 2 });
/** Rough width of a 10px mono label, enough to keep labels apart. */
const textW = (s: string) => s.length * 6.3;

/** Width of the element in px; charts draw at real size so text never scales. */
export function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [w, setW] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    setW(el.clientWidth);
    const ro = new ResizeObserver(([e]) => setW(Math.round(e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

/** Monotone cubic through the points: smooth, and never overshoots the data. Same point count = morphable. */
function curve(p: Pt[]): string {
  const n = p.length;
  if (n < 2) return "";
  const dx: number[] = [], m: number[] = [], t: number[] = new Array(n);
  for (let i = 0; i < n - 1; i++) {
    dx[i] = p[i + 1][0] - p[i][0];
    m[i] = (p[i + 1][1] - p[i][1]) / dx[i];
  }
  t[0] = m[0];
  t[n - 1] = m[n - 2];
  for (let i = 1; i < n - 1; i++) t[i] = m[i - 1] * m[i] <= 0 ? 0 : (2 * m[i - 1] * m[i]) / (m[i - 1] + m[i]);
  let d = `M${f(p[0][0])} ${f(p[0][1])}`;
  for (let i = 0; i < n - 1; i++) {
    const h = dx[i] / 3;
    d += ` C${f(p[i][0] + h)} ${f(p[i][1] + t[i] * h)} ${f(p[i + 1][0] - h)} ${f(p[i + 1][1] - t[i + 1] * h)} ${f(p[i + 1][0])} ${f(p[i + 1][1])}`;
  }
  return d;
}
/** Closed shape between two curves that share their x positions. */
const between = (top: Pt[], bottom: Pt[]) => `${curve(top)} L${curve([...bottom].reverse()).slice(1)} Z`;

/** Round axis values covering [min, max]. */
function niceTicks(min: number, max: number, count = 4): number[] {
  const raw = (max - min || 1) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].find((s) => s * mag >= raw)! * mag;
  const out: number[] = [];
  for (let v = Math.floor(min / step) * step; v < max + step - 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}

function Tooltip({ x, y, width, children }: { x: number; y: number; width: number; children: ReactNode }) {
  const flip = x > width / 2;
  return (
    <div className="pointer-events-none absolute z-10 whitespace-nowrap rounded-[14px] border border-line bg-raised px-3 py-2 text-xs shadow-[var(--shadow-pop)]"
      style={{ left: x + (flip ? -12 : 12), top: y, transform: flip ? "translateX(-100%)" : undefined }}>
      {children}
    </div>
  );
}
function TipRow({ k, v, swatch }: { k: string; v: string; swatch?: string }) {
  return (
    <div className="flex items-center justify-between gap-5 text-ink-2">
      <span className="flex items-center gap-1.5">
        {swatch && <i className="inline-block size-2 rounded-full" style={{ background: swatch }} />}
        {k}
      </span>
      <span className="num text-ink">{v}</span>
    </div>
  );
}

interface FanProps {
  times: string[]; // one label per point: history, then forecast
  hist: number[]; // fact, the last point is "now"
  fc: number[];
  lo: number[];
  hi: number[];
  zero?: boolean; // y axis from zero (a single route) or zoomed to the data (the whole network)
  height?: number;
  peak?: boolean; // mark the forecast maximum
  calls?: number[]; // routes to call a truck for at step k = 0.. from now; drawn as bars on the same time axis
  active?: (k: number) => boolean; // bars the queue below is filtered to
  onSlot?: (k: number) => void;
  ghost?: number[]; // the bars as the forecast planned them, before the dispatcher called anything
  enRoute?: { k: number; count: number }; // trucks called from the console and the step they arrive at
  baseFc?: number[]; // the forecast without an uploaded file: dashed, the gap to `fc` shaded unless shadeGap is off
  shadeGap?: boolean; // off where the band is wide: two fills over each other read as one mess
  histBase?: number[]; // the stored history where an uploaded file changed it
  mark?: { from: number; label: string; short?: string }; // a stretch of the history to point at, from this index up to "now"
  after?: number[]; // what was really shipped after "now", when the chart is about the past
  ghostLabel?: string; // what the ghost bars were: "было по плану" on the shift, "исходный" on an upload
}

/** Fact (ink) up to "now", then the forecast (sky) with its 10-90 % band; optionally the wave of truck calls beneath. */
export function Fan({ times, hist, fc, lo, hi, zero, height = 170, peak, calls, active, onSlot, ghost, enRoute, baseFc, shadeGap = true, histBase, mark, after, ghostLabel = "было по плану" }: FanProps) {
  const [ref, W] = useWidth<HTMLDivElement>();
  const reduced = useReducedMotion();
  const first = useFirstRender();
  const intro = true; // a chart draws itself in every time it appears: on load, on a new result, on coming back to its tab
  const clip = useId().replace(/:/g, "");
  const [hover, setHover] = useState<number | null>(null);
  const [overBars, setOverBars] = useState(false); // only the bars are clickable, so only they get the pointer cursor

  const L = 44, R = 18, top = 26, gap = 30, barsH = 60; // top + height + 24 = 260 at height 210: same box as Dock
  const plotB = top + height;
  const barsB = calls ? plotB + gap + barsH : plotB;
  const roadY = barsB + 14; // under the bars: the road of the trucks already called
  const H = barsB + (calls ? 20 : 0) + 24;
  const h0 = hist.length - 1;
  const n = h0 + fc.length;
  const slot = (W - L - R) / n;
  const x = (i: number) => L + slot * i;
  const seen = [...hist, ...(baseFc ?? []), ...(histBase ?? []), ...(after ?? [])];
  const tk = zero ? niceTicks(0, Math.max(...hi, ...seen), 3) : niceTicks(Math.min(...lo, ...seen), Math.max(...hi, ...seen), 3);
  const [d0, d1] = [tk[0], tk[tk.length - 1]];
  const y = (v: number) => plotB - ((v - d0) / (d1 - d0)) * height;
  // One unit for the whole axis, written once above it.
  const [div, unit] = d1 >= 1e6 ? [1e6, "млн товаров в час"] : d1 >= 1e3 ? [1e3, "тыс. товаров в час"] : [1, "товаров в час"];
  const unitY = top - 12;

  const nowPt: Pt = [x(h0), y(hist[h0])];
  const fact = hist.map((v, i): Pt => [x(i), y(v)]);
  const line = [nowPt, ...fc.map((v, k): Pt => [x(h0 + k + 1), y(v)])];
  const upper = [nowPt, ...hi.map((v, k): Pt => [x(h0 + k + 1), y(v)])];
  const lower = [nowPt, ...lo.map((v, k): Pt => [x(h0 + k + 1), y(v)])];
  const baseLine = baseFc ? [nowPt, ...baseFc.map((v, k): Pt => [x(h0 + k + 1), y(v)])] : null;
  const histChanged = !!histBase && histBase.some((v, i) => Math.abs(v - hist[i]) > 0.5);
  const pk = fc.indexOf(Math.max(...fc));
  const labelStep = Math.max(1, Math.ceil(58 / slot));
  const maxCall = calls ? Math.max(1, ...calls, ...(ghost ?? [])) : 1;
  // The old value of a bar goes over its outline when there is room; when it is close to the new one, it is printed
  // struck through right before the new value instead.
  const wasAbove = (i: number) => !!ghost && !!calls && ghost[i] > calls[i] && (calls[i] === 0 || (barsH * (ghost[i] - calls[i])) / maxCall >= 14);
  const wasInline = (i: number) => !!ghost && !!calls && ghost[i] !== calls[i] && !wasAbove(i);
  const barW = Math.min(22, Math.max(6, slot - 6));
  const activeKs = calls && active ? calls.map((_, k) => k).filter((k) => active(k)) : [];
  const t = (tr: object) => (reduced ? { duration: 0 } : tr);
  const k = hover == null ? -1 : hover - h0;
  const canPick = !!calls && !!onSlot && overBars && k >= 0 && k < calls.length && calls[k] > 0; // a bar under the cursor

  // Direct labels instead of a legend. "факт" sits just left of now, above the line unless it would hit "сейчас".
  const near = fact.filter(([px]) => px >= nowPt[0] - 48).map(([, py]) => py);
  const factY = Math.min(...near) - 8 >= top + 8 ? Math.min(...near) - 8 : Math.min(plotB - 4, Math.max(...near) + 16);
  // Forecast and baseline ends: the higher label goes above its line, the lower one below.
  const endX = W - R;
  const fcY = y(fc[fc.length - 1]);
  const bY = baseFc ? y(baseFc[baseFc.length - 1]) : 0;
  const fcAbove = baseFc ? fcY <= bY : !(peak && x(h0 + pk + 1) > endX - 70); // the peak label takes the room above
  const place = (py: number, above: boolean) => (above ? Math.max(top - 2, py - 8) : Math.min(plotB + 8, py + 15));
  // The mark label goes inside its stretch when, left of it, it would run into the unit or off the chart;
  // there it sits a row higher, over the stretch, clear of the "сейчас" label.
  // On a narrow chart the full label fits neither side, so the short one is used.
  const markLabel = !mark ? "" : x(mark.from) - 6 - textW(mark.label) >= textW(unit) + 10 || x(mark.from) + 6 + textW(mark.label) <= W - R
    ? mark.label : mark.short ?? mark.label;
  const markIn = !!mark && x(mark.from) - 6 - textW(markLabel) < textW(unit) + 10;

  return (
    <div ref={ref} className="relative select-none" style={{ height: H }}>
      {W > MIN_W && (
        <svg width={W} height={H} className="block overflow-visible" role="img"
          aria-label={calls ? "Нагрузка склада и число вызовов по времени" : "Отгрузки маршрута: факт и прогноз"}>
          <defs>
            <clipPath id={clip}>
              <motion.rect x={nowPt[0] - 2} y={0} height={H} initial={intro ? { width: 0 } : false} animate={{ width: W - nowPt[0] + 24 }}
                transition={t({ duration: 0.7, delay: 0.55, ease })} />
            </clipPath>
            <clipPath id={`${clip}b`}><rect x={0} y={0} width={W} height={barsB} /></clipPath>
          </defs>

          {tk.map((v) => (
            <g key={v}>
              <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke="var(--line)" />
              <text className="tick" x={L - 8} y={y(v) + 4} textAnchor="end">{tickNum.format(v / div)}</text>
            </g>
          ))}
          <text className="tick" x={0} y={unitY}>{unit}</text>

          {mark && mark.from < h0 && (
            <g>
              <rect x={x(mark.from)} y={top - 26} width={nowPt[0] - x(mark.from)} height={plotB - top + 26} fill="var(--alert-soft)" />
              <line x1={x(mark.from)} x2={x(mark.from)} y1={top - 26} y2={plotB} stroke="var(--alert)" strokeWidth={1.5} />
              <text className="mark" x={markIn ? x(mark.from) + 6 : x(mark.from) - 6} y={markIn ? top - 31 : top - 15} textAnchor={markIn ? "start" : "end"} style={{ fill: "var(--alert)", fontWeight: 600 }}>{markLabel}</text>
            </g>
          )}

          <g clipPath={`url(#${clip})`}>
            <motion.path fill="var(--forecast-soft)" initial={false} animate={{ d: between(upper, lower) }} transition={t(flow)} />
            {baseLine && (
              <>
                {shadeGap && <motion.path fill="var(--alert-soft)" initial={false} animate={{ d: between(baseLine, line) }} transition={t(flow)} />}
                <motion.path fill="none" stroke="var(--ink-2)" strokeWidth={1.75} strokeDasharray="5 5" strokeLinecap="round"
                  initial={false} animate={{ d: curve(baseLine) }} transition={t(flow)} />
              </>
            )}
            {after && (
              <motion.path fill="none" stroke="var(--ink)" strokeWidth={2} strokeDasharray="1 5" strokeLinecap="round" initial={false}
                animate={{ d: curve([nowPt, ...after.map((v, k): Pt => [x(h0 + k + 1), y(v)])]) }} transition={t(flow)} />
            )}
            <motion.path fill="none" stroke="var(--forecast)" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round"
              style={{ filter: "var(--glow-forecast)" }} initial={false} animate={{ d: curve(line) }} transition={t(flow)} />
            {peak && (
              <g>
                <circle cx={x(h0 + pk + 1)} cy={y(fc[pk])} r={5} fill="var(--forecast)" stroke="var(--ground)" strokeWidth={2} />
                {/* Centred over the dot; next to "now" it moves right so it never runs into the "сейчас" label. */}
                {x(h0 + pk + 1) - nowPt[0] < 56 ? (
                  <text className="mark" x={x(h0 + pk + 1) + 10} y={y(hi[pk]) - 10}>пик {times[h0 + pk + 1]}</text>
                ) : (
                  <text className="mark" x={Math.min(x(h0 + pk + 1), W - R - 34)} y={y(hi[pk]) - 10} textAnchor="middle">пик {times[h0 + pk + 1]}</text>
                )}
              </g>
            )}
            <motion.g initial={false} animate={{ x: endX, y: place(fcY, fcAbove) }} transition={t(flow)}>
              <text className="mark" textAnchor="end" style={{ fill: "var(--forecast-ink)" }}>прогноз</text>
            </motion.g>
            {baseFc && (
              <motion.g initial={false} animate={{ x: endX, y: place(bY, !fcAbove) }} transition={t(flow)}>
                <text className="mark" textAnchor="end">исходный</text>
              </motion.g>
            )}
          </g>

          {histChanged && (
            <path fill="none" stroke="var(--ink-2)" strokeWidth={1.5} strokeDasharray="4 4" d={curve(histBase!.map((v, i): Pt => [x(i), y(v)]))} />
          )}
          <motion.path fill="none" stroke="var(--fact)" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"
            initial={intro ? { pathLength: 0, d: curve(fact) } : false} animate={{ pathLength: 1, d: curve(fact) }}
            transition={t({ pathLength: { duration: 0.7, ease }, d: flow })} />
          {W >= 360 && (
            <motion.g initial={false} animate={{ x: nowPt[0] - 10, y: factY }} transition={t(flow)}>
              <text className="mark" textAnchor="end">факт</text>
            </motion.g>
          )}

          <line x1={nowPt[0]} x2={nowPt[0]} y1={top - 12} y2={barsB} stroke="var(--muted)" strokeDasharray="2 4" />
          <text className="mark" x={nowPt[0] - 8} y={top - 6} textAnchor="end">сейчас</text>
          <circle className="beacon" cx={nowPt[0]} cy={nowPt[1]} r={5} fill="none" stroke="var(--ink)" strokeWidth={1.5} />
          <circle cx={nowPt[0]} cy={nowPt[1]} r={5} fill="var(--ink)" stroke="var(--ground)" strokeWidth={2} />

          {calls && (
            <g>
              <line x1={L} x2={W - R + 12} y1={barsB} y2={barsB} stroke="var(--line)" />
              <text className="tick" x={nowPt[0] - barW / 2 - 10} y={barsB - 8} textAnchor="end">вызвать машин</text>
              <g clipPath={`url(#${clip}b)`}>
                {calls.map((c, i) => {
                  const h = c ? Math.max(3, (barsH * c) / maxCall) : 0;
                  const tone = i === 0 ? "now" : i <= 2 ? "soon" : "later";
                  return (
                    <motion.rect key={i} x={x(h0 + i) - barW / 2} width={barW} rx={Math.min(5, barW / 2)}
                      fill={tone === "later" ? "var(--ink-2)" : "var(--signal)"}
                      style={{ filter: tone === "now" ? "var(--glow-signal)" : undefined }}
                      initial={intro ? { y: barsB, height: 6, opacity: 0 } : false}
                      animate={{ y: barsB - h, height: h + 6, opacity: tone === "now" ? 1 : tone === "soon" ? (k === i ? 0.7 : 0.5) : k === i ? 0.5 : 0.3 }}
                      // A tween, not a spring: a spring overshoots below zero height when a bar collapses.
                      transition={t({ duration: 0.45, ease, delay: intro && first ? 0.7 + i * 0.045 : 0 })} />
                  );
                })}
                {/* What the plan was before the calls: a dotted outline where the bar has shrunk since. */}
                {ghost?.map((g, i) =>
                  g > calls[i] ? (
                    <motion.rect key={`ghost${i}`} x={x(h0 + i) - barW / 2 + 0.5} width={barW - 1} rx={Math.min(5, barW / 2)}
                      y={barsB - Math.max(3, (barsH * g) / maxCall)} height={Math.max(3, (barsH * g) / maxCall) + 6}
                      fill="none" stroke="var(--ink-2)" strokeDasharray="2 3" initial={{ opacity: 0 }} animate={{ opacity: 0.7 }} />
                  ) : null,
                )}
              </g>
              {ghost?.map((g, i) =>
                // The old value is printed only where it has room above the current one; otherwise the tooltip carries it.
                wasAbove(i) && (slot >= 30 || i === 0) ? (
                  <text key={`was${i}`} className="tick" x={x(h0 + i)} y={barsB - Math.max(3, (barsH * g) / maxCall) - 6}
                    textAnchor="middle" textDecoration="line-through">{g}</text>
                ) : null,
              )}
              {enRoute && enRoute.count > 0 && (
                <g>
                  <line x1={nowPt[0]} x2={x(h0 + enRoute.k)} y1={roadY} y2={roadY} stroke="var(--ink)" strokeOpacity={0.3}
                    strokeWidth={2} strokeLinecap="round" />
                  {/* The trucks drive from "now" to their arrival each time the count changes. */}
                  <motion.g key={enRoute.count} initial={{ x: nowPt[0], opacity: 0 }} animate={{ x: x(h0 + enRoute.k), opacity: 1 }}
                    transition={t({ x: { duration: 1.2, ease }, opacity: { duration: 0.2 } })}>
                    <circle cx={0} cy={roadY} r={10} fill="var(--ground)" />
                    <Truck x={-8} y={roadY - 8} width={16} height={16} strokeWidth={1.9} color="var(--ink)" />
                    {/* Right of the truck while it fits; otherwise the short form, or to the left of the truck. */}
                    {(() => {
                      const full = `едут ${enRoute.count}, будут в ${times[h0 + enRoute.k]}`, short = `едут ${enRoute.count}`;
                      const room = W - 4 - (x(h0 + enRoute.k) + 14), w = textW;
                      const label = w(full) <= room ? full : short;
                      return w(label) <= room
                        ? <text className="mark" x={14} y={roadY + 4} style={{ fill: "var(--ink)" }}>{label}</text>
                        : <text className="mark" x={-14} y={roadY + 4} textAnchor="end" style={{ fill: "var(--ink)", paintOrder: "stroke", stroke: "var(--ground)", strokeWidth: 4 }}>{label}</text>;
                    })()}
                  </motion.g>
                </g>
              )}
              {calls.map((c, i) => {
                const h = c ? Math.max(3, (barsH * c) / maxCall) : 0;
                const on = activeKs.includes(i);
                if (!c || (slot < 30 && i !== 0 && k !== i)) return null;
                return (
                  <motion.text key={i} x={x(h0 + i)} textAnchor="middle" className="mark"
                    style={{ fill: on || k === i ? "var(--ink)" : "var(--muted)", fontWeight: on ? 600 : 400 }}
                    initial={intro ? { y: barsB - 6, opacity: 0 } : false} animate={{ y: barsB - h - 6, opacity: 1 }}
                    transition={t({ ...flow, delay: intro && first ? 0.7 + i * 0.045 : 0 })}>
                    {wasInline(i) && (slot >= 44 || i === 0) && (
                      <tspan textDecoration="line-through" style={{ fill: "var(--muted)", fontWeight: 400 }}>{ghost![i]} </tspan>
                    )}
                    {c}
                  </motion.text>
                );
              })}
              {activeKs.length > 0 && (
                <motion.rect y={barsB + 3} height={2} rx={1} fill="var(--ink)" initial={false}
                  animate={{ x: x(h0 + activeKs[0]) - barW / 2, width: x(h0 + activeKs[activeKs.length - 1]) - x(h0 + activeKs[0]) + barW }}
                  // A tween: a spring shrinking the underline from many bars to one overshoots into a negative width.
                  transition={t({ duration: 0.28, ease })} />
              )}
            </g>
          )}

          {times.map((label, i) =>
            (i - h0) % labelStep === 0 ? (
              <text key={i} className="tick" x={x(i)} y={H - 4} textAnchor="middle"
                style={i === h0 ? { fill: "var(--ink)", fontWeight: 600 } : undefined}>{label}</text>
            ) : null,
          )}

          {hover != null && (
            <g pointerEvents="none">
              <line x1={x(hover)} x2={x(hover)} y1={top} y2={barsB} stroke="var(--ink-2)" strokeOpacity={0.5} />
              <circle cx={x(hover)} cy={y(hover <= h0 ? hist[hover] : fc[hover - h0 - 1])} r={5}
                fill={hover <= h0 ? "var(--ink)" : "var(--forecast)"} stroke="var(--ground)" strokeWidth={2} />
            </g>
          )}
          <rect x={L - slot / 2} y={top} width={W - L - R + slot} height={barsB - top} fill="transparent"
            style={{ cursor: canPick ? "pointer" : "default" }}
            onPointerMove={(e) => {
              const box = e.currentTarget.ownerSVGElement!.getBoundingClientRect();
              setHover(Math.max(0, Math.min(n, Math.round((e.clientX - box.left - L) / slot))));
              setOverBars(e.clientY - box.top > plotB + 6);
            }}
            onPointerLeave={() => { setHover(null); setOverBars(false); }}
            onClick={() => canPick && onSlot!(k)} />
        </svg>
      )}
      {hover != null && W > MIN_W && (
        <Tooltip x={x(hover)} y={top} width={W}>
          <div className="num mb-1 font-semibold text-ink">{times[hover]}</div>
          {hover <= h0 ? (
            <>
              <TipRow k="факт" v={vol(hist[hover])} swatch="var(--ink)" />
              {histChanged && histBase![hover] !== hist[hover] && <TipRow k="в истории" v={vol(histBase![hover])} />}
            </>
          ) : (
            <>
              <TipRow k="прогноз" v={vol(fc[hover - h0 - 1])} swatch="var(--forecast)" />
              {baseFc && <TipRow k="исходный" v={vol(baseFc[hover - h0 - 1])} />}
              {after && <TipRow k="было на деле" v={vol(after[hover - h0 - 1])} swatch="var(--ink)" />}
              <TipRow k="диапазон" v={`${vol(lo[hover - h0 - 1])} … ${vol(hi[hover - h0 - 1])}`} />
            </>
          )}
          {calls && k >= 0 && k < calls.length && <TipRow k="вызвать" v={String(calls[k])} swatch="var(--signal)" />}
          {ghost && k >= 0 && k < ghost.length && ghost[k] !== calls?.[k] && <TipRow k={ghostLabel} v={String(ghost[k])} />}
          {enRoute && enRoute.count > 0 && k === enRoute.k && <TipRow k="приедут" v={String(enRoute.count)} swatch="var(--ink)" />}
        </Tooltip>
      )}
    </div>
  );
}

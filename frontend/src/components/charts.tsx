// Charts of the console, drawn by hand in SVG so every mark can move: lines draw in, curves morph
// between routes, bars re-flow when the scenario changes. Rules: docs of the "Ночная смена" system.
import { motion, useReducedMotion } from "motion/react";
import { Truck } from "lucide-react";
import { useId, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { dec2, vol, volShort } from "../format";
import { ease, flow, useFirstRender } from "../motion";

type Pt = [number, number];
const f = (v: number) => v.toFixed(1);

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
/** Closed shape from a curve down to a baseline. */
const under = (top: Pt[], base: number) => `${curve(top)} L${f(top[top.length - 1][0])} ${f(base)} L${f(top[0][0])} ${f(base)} Z`;

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
    <div className="pointer-events-none absolute z-10 whitespace-nowrap rounded-[10px] border border-line bg-raised px-3 py-2 text-xs shadow-[var(--shadow-pop)]"
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
}

/** Fact (ink) up to "now", then the forecast (sky) with its 10-90 % band; optionally the wave of truck calls beneath. */
export function Fan({ times, hist, fc, lo, hi, zero, height = 170, peak, calls, active, onSlot }: FanProps) {
  const [ref, W] = useWidth<HTMLDivElement>();
  const reduced = useReducedMotion();
  const first = useFirstRender();
  const clip = useId().replace(/:/g, "");
  const [hover, setHover] = useState<number | null>(null);

  const L = 44, R = 18, top = 26, gap = 30, barsH = 60;
  const plotB = top + height;
  const barsB = calls ? plotB + gap + barsH : plotB;
  const H = barsB + 24;
  const h0 = hist.length - 1;
  const n = h0 + fc.length;
  const slot = (W - L - R) / n;
  const x = (i: number) => L + slot * i;
  const tk = zero ? niceTicks(0, Math.max(...hi, ...hist), 3) : niceTicks(Math.min(...lo, ...hist), Math.max(...hi, ...hist), 3);
  const [d0, d1] = [tk[0], tk[tk.length - 1]];
  const y = (v: number) => plotB - ((v - d0) / (d1 - d0)) * height;

  const nowPt: Pt = [x(h0), y(hist[h0])];
  const fact = hist.map((v, i): Pt => [x(i), y(v)]);
  const line = [nowPt, ...fc.map((v, k): Pt => [x(h0 + k + 1), y(v)])];
  const upper = [nowPt, ...hi.map((v, k): Pt => [x(h0 + k + 1), y(v)])];
  const lower = [nowPt, ...lo.map((v, k): Pt => [x(h0 + k + 1), y(v)])];
  const pk = fc.indexOf(Math.max(...fc));
  const labelStep = Math.max(1, Math.ceil(58 / slot));
  const maxCall = calls ? Math.max(1, ...calls) : 1;
  const barW = Math.min(22, Math.max(6, slot - 6));
  const activeKs = calls && active ? calls.map((_, k) => k).filter((k) => active(k)) : [];
  const t = (tr: object) => (reduced ? { duration: 0 } : tr);
  const k = hover == null ? -1 : hover - h0;

  return (
    <div ref={ref} className="relative select-none">
      {W > 0 && (
        <svg width={W} height={H} className="block overflow-visible" role="img"
          aria-label={calls ? "Нагрузка склада и число вызовов по времени" : "Отгрузки маршрута: факт и прогноз"}>
          <defs>
            <clipPath id={clip}>
              <motion.rect x={nowPt[0] - 2} y={0} height={H} initial={{ width: 0 }} animate={{ width: W - nowPt[0] + 24 }}
                transition={t({ duration: 0.7, delay: 0.55, ease })} />
            </clipPath>
            <clipPath id={`${clip}b`}><rect x={0} y={0} width={W} height={barsB} /></clipPath>
          </defs>

          {tk.map((v) => (
            <g key={v}>
              <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke="var(--line)" />
              <text className="tick" x={L - 8} y={y(v) + 4} textAnchor="end">{volShort(v)}</text>
            </g>
          ))}

          <g clipPath={`url(#${clip})`}>
            <motion.path fill="var(--forecast-soft)" initial={false} animate={{ d: between(upper, lower) }} transition={t(flow)} />
            <motion.path fill="none" stroke="var(--forecast)" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round"
              style={{ filter: "var(--glow-forecast)" }} initial={false} animate={{ d: curve(line) }} transition={t(flow)} />
            {peak && (
              <g>
                <circle cx={x(h0 + pk + 1)} cy={y(fc[pk])} r={5} fill="var(--forecast)" stroke="var(--ground)" strokeWidth={2} />
                <text className="mark" x={Math.min(x(h0 + pk + 1), W - R - 34)} y={y(hi[pk]) - 10} textAnchor="middle">пик {times[h0 + pk + 1]}</text>
              </g>
            )}
          </g>

          <motion.path fill="none" stroke="var(--ink)" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"
            initial={{ pathLength: 0, d: curve(fact) }} animate={{ pathLength: 1, d: curve(fact) }}
            transition={t({ pathLength: { duration: 0.7, ease }, d: flow })} />

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
                      initial={{ y: barsB, height: 6, opacity: 0 }}
                      animate={{ y: barsB - h, height: h + 6, opacity: tone === "now" ? 1 : tone === "soon" ? (k === i ? 0.7 : 0.5) : k === i ? 0.5 : 0.3 }}
                      transition={t({ ...flow, delay: first ? 0.7 + i * 0.045 : 0 })} />
                  );
                })}
              </g>
              {calls.map((c, i) => {
                const h = c ? Math.max(3, (barsH * c) / maxCall) : 0;
                const on = activeKs.includes(i);
                if (!c || (slot < 30 && i !== 0 && k !== i)) return null;
                return (
                  <motion.text key={i} x={x(h0 + i)} textAnchor="middle" className="mark"
                    style={{ fill: on || k === i ? "var(--ink)" : "var(--muted)", fontWeight: on ? 600 : 400 }}
                    initial={{ y: barsB - 6, opacity: 0 }} animate={{ y: barsB - h - 6, opacity: 1 }}
                    transition={t({ ...flow, delay: first ? 0.7 + i * 0.045 : 0 })}>
                    {c}
                  </motion.text>
                );
              })}
              {activeKs.length > 0 && (
                <motion.rect y={barsB + 3} height={2} rx={1} fill="var(--ink)" initial={false}
                  animate={{ x: x(h0 + activeKs[0]) - barW / 2, width: x(h0 + activeKs[activeKs.length - 1]) - x(h0 + activeKs[0]) + barW }}
                  transition={t({ type: "spring", stiffness: 500, damping: 40 })} />
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
            style={{ cursor: calls && onSlot && k >= 0 && k < calls.length ? "pointer" : "crosshair" }}
            onPointerMove={(e) => {
              const box = e.currentTarget.ownerSVGElement!.getBoundingClientRect();
              setHover(Math.max(0, Math.min(n, Math.round((e.clientX - box.left - L) / slot))));
            }}
            onPointerLeave={() => setHover(null)}
            onClick={() => calls && onSlot && k >= 0 && k < calls.length && onSlot(k)} />
        </svg>
      )}
      {hover != null && W > 0 && (
        <Tooltip x={x(hover)} y={top} width={W}>
          <div className="num mb-1 font-semibold text-ink">{times[hover]}</div>
          {hover <= h0 ? (
            <TipRow k="факт" v={vol(hist[hover])} swatch="var(--ink)" />
          ) : (
            <>
              <TipRow k="прогноз" v={vol(fc[hover - h0 - 1])} swatch="var(--forecast)" />
              <TipRow k="коридор" v={`${volShort(lo[hover - h0 - 1])} – ${volShort(hi[hover - h0 - 1])}`} />
            </>
          )}
          {calls && k >= 0 && k < calls.length && <TipRow k="вызвать" v={String(calls[k])} swatch="var(--signal)" />}
        </Tooltip>
      )}
    </div>
  );
}

interface DockProps {
  times: string[]; // now and the 8 steps after it
  line: number[]; // truckloads at the dock by the inflow the call rule decides on
  hi: number[]; // the cautious scenario, never below the line
  need: number; // level at which one more truck is needed
  callK?: number; // step of the call; undefined when no call is needed within the horizon
  arriveK?: number;
  runKey: string; // changes when the truck should drive again: route, call or lead time
}

/** Truckloads at the dock over the horizon: the curve climbs to the "one more truck" level, the truck is called a lead time before. */
export function Dock({ times, line, hi, need, callK, arriveK, runKey }: DockProps) {
  const [ref, W] = useWidth<HTMLDivElement>();
  const reduced = useReducedMotion();
  const grad = useId().replace(/:/g, "");
  const [hover, setHover] = useState<number | null>(null);

  const L = 28, R = 18, top = 40, height = 170, road = 20;
  const B = top + height;
  const H = B + road + 30;
  const n = line.length - 1;
  const slot = (W - L - R) / n;
  const x = (k: number) => L + slot * k;
  const max = Math.ceil(Math.max(need + 0.4, ...hi) * 1.05);
  const y = (v: number) => B - (v / max) * height;
  const step = max > 8 ? 2 : 1;
  const pts = line.map((v, k): Pt => [x(k), y(v)]);
  const cautious = hi.map((v, k): Pt => [x(k), y(v)]);
  const t = (tr: object) => (reduced ? { duration: 0 } : tr);
  const called = callK != null && arriveK != null;
  const roadY = B + road;
  const anchor = (px: number) => (px > W - 120 ? { x: px - 6, textAnchor: "end" as const } : { x: px + 6, textAnchor: "start" as const });

  return (
    <div ref={ref} className="relative select-none">
      {W > 0 && (
        <svg width={W} height={H} className="block overflow-visible" role="img" aria-label="Груз у ворот по времени, в кузовах">
          <defs>
            <linearGradient id={grad} x1="0" x2="0" y1="0" y2="1">
              <stop offset="0" stopColor="var(--forecast)" stopOpacity={0.3} />
              <stop offset="1" stopColor="var(--forecast)" stopOpacity={0} />
            </linearGradient>
          </defs>
          {Array.from({ length: Math.floor(max / step) + 1 }, (_, i) => i * step).map((v) => (
            <g key={v}>
              <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke="var(--line)" />
              <text className="tick" x={L - 8} y={y(v) + 4} textAnchor="end">{v}</text>
            </g>
          ))}

          <motion.path fill="var(--forecast-soft)" initial={false} animate={{ d: between(cautious, pts) }} transition={t(flow)} />
          <motion.path fill={`url(#${grad})`} initial={false} animate={{ d: under(pts, B) }} transition={t(flow)} />
          <motion.path fill="none" stroke="var(--forecast)" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round"
            style={{ filter: "var(--glow-forecast)" }}
            initial={{ pathLength: 0, d: curve(pts) }} animate={{ pathLength: 1, d: curve(pts) }}
            transition={t({ pathLength: { duration: 0.8, ease }, d: flow })} />

          {/* The label sits where the curve is not: above the level while cargo is still short of it, below once it is over. */}
          <motion.g initial={false} animate={{ y: y(need) }} transition={t(flow)}>
            <line x1={L} x2={W - R} stroke="var(--ink-2)" strokeDasharray="6 4" />
            <text className="mark" x={L + 10} y={line[0] < need ? -7 : 15}>нужна ещё машина</text>
          </motion.g>

          {called && (
            <g>
              <motion.g initial={false} animate={{ x: x(callK) }} transition={t(flow)}>
                <line y1={top - 26} y2={roadY} stroke="var(--signal)" strokeWidth={2} style={{ filter: "var(--glow-signal)" }} />
              </motion.g>
              <text className="mark" style={{ fill: "var(--signal-ink)" }} y={top - 18} {...anchor(x(callK))}>
                вызов {callK === 0 ? "сейчас" : times[callK]}
              </text>
              <motion.g initial={false} animate={{ x: x(arriveK) }} transition={t(flow)}>
                <line y1={top - 10} y2={roadY} stroke="var(--signal)" strokeWidth={1.5} strokeDasharray="2 4" />
              </motion.g>
              <text className="mark" style={{ fill: "var(--signal-ink)" }} y={top - 2} {...anchor(x(arriveK))}>
                прибытие {times[arriveK]}
              </text>
              <motion.circle r={5.5} fill="var(--signal)" stroke="var(--surface)" strokeWidth={2} initial={false}
                animate={{ cx: x(arriveK), cy: y(line[arriveK]) }} transition={t(flow)} />
              {/* The road: its length is the lead time, and the truck drives it once per plan. */}
              <motion.line y1={roadY} y2={roadY} stroke="var(--signal)" strokeOpacity={0.35} strokeWidth={2} strokeLinecap="round"
                initial={false} animate={{ x1: x(callK), x2: x(arriveK) }} transition={t(flow)} />
              <motion.g key={runKey} initial={{ x: x(callK), opacity: 0 }} animate={{ x: x(arriveK), opacity: 1 }}
                transition={t({ x: { duration: 1.5, delay: 0.35, ease }, opacity: { duration: 0.2, delay: 0.3 } })}>
                <circle cx={0} cy={roadY} r={11} fill="var(--surface)" />
                <Truck x={-9} y={roadY - 9} width={18} height={18} strokeWidth={1.9} color="var(--signal)" />
              </motion.g>
            </g>
          )}

          {times.map((label, k) =>
            slot >= 54 || k % 2 === 0 ? (
              <text key={k} className="tick" x={x(k)} y={H - 4} textAnchor="middle"
                style={k === 0 ? { fill: "var(--ink)", fontWeight: 600 } : undefined}>{label}</text>
            ) : null,
          )}

          {hover != null && (
            <g pointerEvents="none">
              <line x1={x(hover)} x2={x(hover)} y1={top} y2={B} stroke="var(--ink-2)" strokeOpacity={0.5} />
              <circle cx={x(hover)} cy={y(line[hover])} r={5} fill="var(--forecast)" stroke="var(--surface)" strokeWidth={2} />
            </g>
          )}
          <rect x={L - slot / 2} y={top} width={W - L - R + slot} height={height} fill="transparent" style={{ cursor: "crosshair" }}
            onPointerMove={(e) => {
              const box = e.currentTarget.ownerSVGElement!.getBoundingClientRect();
              setHover(Math.max(0, Math.min(n, Math.round((e.clientX - box.left - L) / slot))));
            }}
            onPointerLeave={() => setHover(null)} />
        </svg>
      )}
      {hover != null && W > 0 && (
        <Tooltip x={x(hover)} y={top + 8} width={W}>
          <div className="num mb-1 font-semibold text-ink">{times[hover]}</div>
          <TipRow k="у ворот" v={`${dec2(line[hover])} кузова`} swatch="var(--forecast)" />
          {hi[hover] - line[hover] > 0.005 && <TipRow k="осторожно до" v={dec2(hi[hover])} />}
        </Tooltip>
      )}
    </div>
  );
}

/** A truck whose body fills with the cargo waiting at the dock: one full body = one truckload. */
export function TruckGauge({ loads }: { loads: number }) {
  const full = loads >= 1;
  return (
    <svg width={50} height={24} viewBox="0 0 58 28" aria-hidden className="shrink-0">
      <rect x={1} y={3} width={38} height={18} rx={4} fill="none" stroke="var(--ink-2)" strokeWidth={1.5} />
      <clipPath id="truck-body"><rect x={3.5} y={5.5} width={33} height={13} rx={2} /></clipPath>
      <g clipPath="url(#truck-body)">
        <motion.rect x={3.5} y={5.5} height={13} fill={full ? "var(--signal)" : "var(--forecast)"} initial={false}
          animate={{ width: 33 * Math.min(1, loads) }} transition={flow} />
      </g>
      <path d="M41.5 8.5h7.2l6.3 6v6.5h-13.5z" fill="none" stroke="var(--ink-2)" strokeWidth={1.5} strokeLinejoin="round" />
      <circle cx={11} cy={23} r={3.2} fill="var(--surface)" stroke="var(--ink-2)" strokeWidth={1.5} />
      <circle cx={47} cy={23} r={3.2} fill="var(--surface)" stroke="var(--ink-2)" strokeWidth={1.5} />
    </svg>
  );
}

/** Width of the forecast band as a three-step meter: more bars = a more certain forecast. */
export function Meter({ level }: { level: 1 | 2 | 3 }) {
  return (
    <span className="inline-flex items-end gap-[3px]" aria-hidden>
      {[1, 2, 3].map((i) => (
        <motion.i key={i} className="w-[5px] rounded-[2px]" style={{ height: 4 + i * 4 }} initial={false}
          animate={{ backgroundColor: i <= level ? "var(--ink)" : "var(--line)" }} />
      ))}
    </span>
  );
}

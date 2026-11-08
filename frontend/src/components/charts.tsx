import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Board } from "../data";
import { dec2, hhmm, vol, volShort } from "../format";
import { clock } from "../plan";

const axis = { stroke: "var(--muted)", fontSize: 11, tickLine: false, axisLine: false } as const;

function Tip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  const rows: [string, string][] = [];
  if (p.fact != null) rows.push(["факт", vol(p.fact)]);
  if (p.fc != null && p.fact == null) rows.push(["прогноз", vol(p.fc)]);
  if (p.band) rows.push(["диапазон", `${vol(p.band[0])} – ${vol(p.band[1])}`]);
  return (
    <div className="rounded-lg border border-line bg-card px-3 py-2 text-xs shadow-lg">
      <div className="num mb-1 font-semibold">{label}</div>
      {rows.map(([k, v]) => (
        <div key={k} className="num flex justify-between gap-4 text-ink-2">
          <span>{k}</span>
          <span className="text-ink">{v}</span>
        </div>
      ))}
    </div>
  );
}

export interface FanPoint {
  t: string;
  fact?: number;
  fc?: number;
  band?: [number, number];
}

/** Chart points: history up to "now", then the forecast with its band, joined at "now". */
export function fanSeries(board: Board, hist: number[], fc: number[], lo: number[], hi: number[]) {
  const h0 = hist.length - 1;
  const data: FanPoint[] = hist.map((v, i) => ({ t: hhmm(clock(board, i - h0)), fact: v }));
  data[h0] = { ...data[h0], fc: hist[h0], band: [hist[h0], hist[h0]] };
  fc.forEach((v, k) => data.push({ t: hhmm(clock(board, k + 1)), fc: v, band: [lo[k], hi[k]] }));
  return { data, nowLabel: data[h0].t };
}

interface FanProps {
  data: FanPoint[];
  nowLabel: string;
  peak?: FanPoint;
  compact?: boolean; // small, no grid: the warehouse load next to the answer
  animate?: boolean;
}

/** History (ink) + forecast (accent) with its band; "now" marked; optional peak dot. */
export function Fan({ data, nowLabel, peak, compact, animate = true }: FanProps) {
  return (
    <ResponsiveContainer width="100%" height={compact ? 150 : 200}>
      <ComposedChart data={data} margin={{ top: 16, right: 20, bottom: 0, left: 0 }}>
        {!compact && <CartesianGrid stroke="var(--line)" vertical={false} />}
        <XAxis dataKey="t" {...axis} interval={compact ? 7 : 3} />
        <YAxis {...axis} width={48} tickFormatter={volShort} domain={[0, "auto"]} />
        <Tooltip content={<Tip />} cursor={{ stroke: "var(--muted)", strokeDasharray: "3 3" }} />
        <Area dataKey="band" stroke="none" fill="var(--accent-soft)" isAnimationActive={false} />
        <ReferenceLine x={nowLabel} stroke="var(--muted)" strokeDasharray="3 4" label={{ value: "сейчас", position: "insideTopLeft", fill: "var(--muted)", fontSize: 11 }} />
        <Line dataKey="fact" stroke="var(--ink)" strokeWidth={2} dot={false} isAnimationActive={false} />
        <Line dataKey="fc" stroke="var(--accent)" strokeWidth={2.5} dot={false}
          isAnimationActive={animate} animationDuration={450} animationEasing="ease-out" />
        {peak && (
          <ReferenceDot x={peak.t} y={peak.fc} r={5} fill="var(--accent)" stroke="var(--card)" strokeWidth={2}
            label={{ value: `пик ${peak.t}`, position: "top", fill: "var(--ink-2)", fontSize: 11 }} />
        )}
      </ComposedChart>
    </ResponsiveContainer>
  );
}

interface DockPoint {
  t: string;
  p50: number;
  band: [number, number];
}

/** Truckloads at the dock by forecast, cautious band, the "one more truck" level, call and arrival. */
export function Dock({ data, need, call, arrive, animate = true }: { data: DockPoint[]; need: number; call?: string; arrive?: string; animate?: boolean }) {
  const max = Math.max(need + 0.5, ...data.map((d) => d.band[1])) * 1.08;
  return (
    <ResponsiveContainer width="100%" height={210}>
      <ComposedChart data={data} margin={{ top: 18, right: 20, bottom: 0, left: 0 }}>
        <CartesianGrid stroke="var(--line)" vertical={false} />
        <XAxis dataKey="t" {...axis} interval={1} />
        <YAxis {...axis} width={32} domain={[0, Math.ceil(max)]} allowDecimals={false} />
        <Tooltip
          cursor={{ stroke: "var(--muted)", strokeDasharray: "3 3" }}
          content={({ active, payload, label }: any) =>
            active && payload?.length ? (
              <div className="rounded-lg border border-line bg-card px-3 py-2 text-xs shadow-lg">
                <div className="num mb-1 font-semibold">{label}</div>
                <div className="num text-ink-2">по прогнозу <b className="text-ink">{dec2(payload[0].payload.p50)}</b> кузова</div>
                <div className="num text-ink-2">осторожно до <b className="text-ink">{dec2(payload[0].payload.band[1])}</b></div>
              </div>
            ) : null
          }
        />
        <Area dataKey="band" stroke="none" fill="var(--accent-soft)" isAnimationActive={false} />
        <ReferenceLine y={need} stroke="var(--ink-2)" strokeDasharray="6 4"
          label={{ value: "нужна ещё машина", position: "insideBottomRight", fill: "var(--ink-2)", fontSize: 11 }} />
        {call && <ReferenceLine x={call} stroke="var(--signal)" strokeWidth={2} label={{ value: `вызов ${call}`, position: "insideTopLeft", fill: "var(--signal)", fontSize: 11 }} />}
        {arrive && <ReferenceLine x={arrive} stroke="var(--signal)" strokeDasharray="3 4" label={{ value: "прибытие", position: "insideBottomLeft", fill: "var(--signal)", fontSize: 11 }} />}
        <Line dataKey="p50" stroke="var(--accent)" strokeWidth={2.5} dot={false}
          isAnimationActive={animate} animationDuration={450} animationEasing="ease-out" />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

import { useLayoutEffect, useRef, useState } from 'react';
import { motion, useReducedMotion } from 'motion/react';
import { flow, hourly, number, pastHourly, time, type Route } from './model';
import Icon from './Icon';
import Numeric from './Numeric';
import { transition } from './motion';
export default function Forecast({ route, metric, point, onMetric, onPoint, unavailable, info }: {
    route: Route;
    metric: 'hourly' | 'inflow';
    point: number;
    onMetric: (v: 'hourly' | 'inflow') => void;
    onPoint: (v: number) => void;
    unavailable: boolean;
    info: () => void;
}) {
    const container = useRef<HTMLDivElement>(null), [size, setSize] = useState({ width: 0, height: 220 });
    const width = size.width;
    const reduced = useReducedMotion();
    useLayoutEffect(() => { const el = container.current; if (!el)
        return; const measure = () => { const width = Math.max(270, Math.floor(el.clientWidth)), height = Math.max(210, Math.floor(el.clientHeight)); setSize(previous => previous.width === width && previous.height === height ? previous : { width, height }); }; measure(); const observer = new ResizeObserver(measure); observer.observe(el); return () => observer.disconnect(); }, [unavailable]);
    const values = (metric === 'hourly' ? hourly : flow).map(v => v * route.k), past = (metric === 'hourly' ? pastHourly : [0, 0, 0, 0]).map(v => v * route.k);
    const max = metric === 'hourly' ? 150 : 450, left = 34, right = 14, top = 32, height = size.height, bottom = 32;
    const x = (i: number) => left + (width - left - right) / 11 * i, y = (v: number) => height - bottom - (height - top - bottom) * v / max;
    const path = (points: number[][]) => points.map(([a, b], i) => `${i ? 'L' : 'M'}${a.toFixed(1)},${b.toFixed(1)}`).join(' ');
    const band = [...values.map((v, i) => [x(i + 4), y(v * 1.2)]), ...values.map((v, i) => [x(i + 4), y(v * .8)]).reverse()];
    const t = reduced ? { duration: 0 } : transition;
    return <section className="forecast" aria-label={'Прогноз маршрута ' + route.id}><div className="block-heading"><div className="forecast-title"><h3>Прогноз отгрузок</h3><button className="icon-btn info-button" aria-label="Как читать прогноз" onClick={info}><Icon name="info"/></button></div><div className="metric-switch" role="group" aria-label="Показатель графика">{([['hourly', 'За час'], ['inflow', 'Накопление']] as const).map(([id, label]) => <button key={id} data-metric={id} aria-pressed={metric === id} onClick={() => onMetric(id)}>{label}</button>)}</div></div>{unavailable ? <div className="unavailable">Нет свежих данных для прогноза</div> : <><div className="plot-summary"><span>{time((point + 1) * 30)}</span><strong><Numeric value={values[point]}/></strong><small>ед.</small></div><div ref={container} className="plot-area">{width > 0 && <svg className="forecast-svg" width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={values.map((v, i) => `${time((i + 1) * 30)}: ${number(v)} тысяч`).join('; ')}><rect className="forecast-zone" x={x(3)} y={top} width={width - right - x(3)} height={height - top - bottom} rx="3"/>{[0, .5, 1].map(f => <g key={f}><line className="grid-line" x1={left} y1={y(f * max)} x2={width - right} y2={y(f * max)}/><text x={left - 9} y={y(f * max) + 4} textAnchor="end">{number(f * max)}</text></g>)}<text x={left} y="16">тыс. ед.</text><text x={x(3) + 8} y="16">10:30 · прогноз</text><motion.path initial={false} className="chart-band" animate={{ d: path(band) + 'Z' }} transition={t}/><motion.path initial={false} className="past-line" animate={{ d: path(past.map((v, i) => [x(i), y(v)])) }} transition={t}/><motion.path initial={false} className="forecast-line" animate={{ d: path([[x(3), y(past[3])], ...values.map((v, i) => [x(i + 4), y(v)])]) }} transition={t}/><motion.line initial={false} className="point-guide" y1={top} y2={height - bottom} animate={{ x1: x(point + 4), x2: x(point + 4) }} transition={t}/>{values.map((v, i) => <motion.circle key={i} initial={false} className="plot-dot" animate={{ cx: x(i + 4), cy: y(v), r: point === i ? 4.5 : 2.5 }} transition={t}/>)}{[0, 4, 7, 11].map(i => <text key={i} x={x(i)} y={height - 9} textAnchor={i === 0 ? 'start' : i === 11 ? 'end' : 'middle'}>{time((i - 3) * 30)}</text>)}</svg>}</div><div className="forecast-points" aria-label="Точки прогноза">{values.map((v, i) => <button key={i} data-point={i} aria-pressed={point === i} onClick={() => onPoint(i)}><span>{time((i + 1) * 30)}</span><strong><Numeric value={v} format="number"/></strong></button>)}</div><div className="legend"><span><i className="past"/>История</span><span><i className="future"/>Прогноз</span><span><i className="band"/>Диапазон</span></div></>}</section>;
}

export type DataStatus = 'ready' | 'loading' | 'error' | 'partial';
export type Filter = 'all' | 'now' | 'soon' | 'later';
export type Route = {
    id: number;
    backlog: number;
    capacity: number;
    k: number;
};
export const flow = [25, 52, 85, 120, 160, 202, 239, 270];
export const hourly = [48, 52, 60, 68, 75, 82, 79, 68];
export const pastHourly = [47, 44, 46, 48];
export const routes: Route[] = [
    { id: 471, backlog: 74, capacity: 120, k: 1 },
    { id: 104, backlog: 92, capacity: 100, k: .65 },
    { id: 208, backlog: 24, capacity: 90, k: .9 },
    { id: 629, backlog: 18, capacity: 110, k: 1.3 },
    { id: 315, backlog: 3, capacity: 75, k: .7 },
    { id: 862, backlog: 15, capacity: 100, k: .5 },
];
const formatter = new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 1 });
export const number = (v: number) => formatter.format(v);
export const amount = (v: number) => `${number(v)} тыс.`;
export const time = (m: number) => `${Math.floor((630 + m) / 60)}:${String((630 + m) % 60).padStart(2, '0')}`;
export function scenario(r: Route, lead: number, unavailable = false) {
    const step = flow.findIndex(v => r.backlog + v * r.k >= r.capacity);
    const ready = unavailable || step < 0 ? null : (step + 1) * 30;
    const call = ready === null ? null : Math.max(0, ready - lead);
    const arrival = call === null ? null : call + lead;
    const incoming = arrival === null ? null : (flow[arrival / 30 - 1] ?? flow.at(-1)!) * r.k;
    const total = incoming === null ? null : r.backlog + incoming;
    const kind = unavailable ? 'stale' : call === 0 ? 'now' : call !== null && call <= 60 ? 'soon' : 'later';
    return { ready, call, arrival, incoming, total, kind, label: unavailable ? 'Нет данных' : call === 0 ? 'Сейчас' : call === null ? 'Не требуется' : time(call) };
}

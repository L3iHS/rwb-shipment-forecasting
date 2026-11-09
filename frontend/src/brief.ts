// Shift brief and the offline answers of the assistant, built only from tools.ts.
import type { Board } from "./data";
import { dec1, dec2, int, leadLabel, plural, routesWord, vol } from "./format";
import { evidence, findRoutes, networkSummary, routeStatus, whatIf, type Action } from "./tools";
import type { Risk } from "./plan";

type Part = string | { label: string; action: Action };
export type Line = Part[];

export function brief(board: Board, calls: number[]): Line[] {
  const n = networkSummary(board, calls);
  const top = findRoutes(board, calls, "call_now", 3).routes;
  const low = findRoutes(board, calls, "low_confidence", 0).count;
  const ev = evidence(board).loss_per_truck_min;
  const lines: Line[] = [
    ["Вызвать машины сейчас: ", { label: `${int(n.call_now)} ${routesWord(n.call_now)}`, action: { type: "filter", value: "now" } },
      `, ещё ${int(n.call_within_hour)} в течение часа. До конца горизонта нужно ${int(n.new_trucks)} новых машин, с запасом до ${int(n.new_trucks_cautious)}.`],
    [`Пик отгрузок сети в ${n.peak_at}: ${vol(n.peak_volume)} за час.`],
  ];
  if (top.length)
    lines.push(["Больше всего груза у ворот: ",
      ...top.flatMap((t, i): Part[] => [{ label: `№ ${t.route_id}`, action: { type: "open_route", id: t.route_id } }, ` (${dec2(t.backlog_loads)})${i < top.length - 1 ? ", " : "."}`])]);
  if (n.waiting_without_truck)
    lines.push(["Груз ждёт без машины: ", { label: `${n.waiting_without_truck} ${routesWord(n.waiting_without_truck)}`, action: { type: "filter", value: "jam" } }, "."]);
  if (low) lines.push([`У ${low} ${plural(low, "маршрута", "маршрутов", "маршрутов")} с вызовом в ближайший час прогноз неуверенный: держите резервную машину.`]);
  lines.push([`На проверочной неделе модель сокращает потери до ${dec1(ev.model)} мин на машину против ${dec1(ev.reactive)} при вызове по полному кузову.`]);
  return lines;
}

/** Offline answer for common questions; null when the question needs the language model. */
export function localAnswer(board: Board, calls: number[], q: string, leadH: number, risk: Risk): { text: string; actions: Action[] } | null {
  const s = q.toLowerCase();
  const num = s.match(/\d+/)?.[0];
  if (num && /маршрут|№|почему|вызыва/.test(s)) {
    const r = routeStatus(board, +num, leadH, risk);
    if (!r) return { text: `Маршрута № ${num} нет: номера от 0 до ${board.routes.length - 1}.`, actions: [] };
    const when = r.call_now ? "вызывать сейчас" : r.call_at ? `вызывать в ${r.call_at}` : "новый вызов в горизонте не нужен";
    return {
      text: `Маршрут № ${r.route_id}: ${when}. У ворот ${dec2(r.backlog_loads)} кузова, вызвано машин: ${r.trucks_called}. ` +
        `За 4 часа ожидается ${vol(r.forecast_4h)} товаров, с запасом до ${vol(r.forecast_4h_p90)}. Кузов вмещает ${vol(r.capacity)}. ` +
        `Машина едет ${leadLabel(leadH)}, поэтому вызов приходится на момент, когда к её прибытию наберётся полный кузов.`,
      actions: [{ type: "open_route", id: r.route_id }],
    };
  }
  if (/без машин|ждёт|ждет/.test(s)) {
    const f = findRoutes(board, calls, "waiting_without_truck", 5);
    return { text: `Груз ждёт без машины на ${f.count} ${plural(f.count, "маршруте", "маршрутах", "маршрутах")}: ${f.routes.map((r) => `№ ${r.route_id} (${dec2(r.backlog_loads)} кузова)`).join(", ")}. Им нужен вызов сейчас.`,
      actions: [{ type: "filter", value: "jam" }] };
  }
  const lead = s.match(/(\d+(?:[.,]\d+)?)\s*(ч|час|мин)/);
  if (lead || /что если|подач/.test(s)) {
    const amount = lead ? +lead[1].replace(",", ".") : 2;
    const h = lead?.[2] === "мин" ? amount / 60 : amount;
    const askedH = [0.5, 1, 2].includes(h) ? h : 2;
    const w = whatIf(board, askedH, risk);
    return { text: `Если машина едет ${leadLabel(askedH)}, вызвать сейчас нужно на ${w.call_now} ${plural(w.call_now, "маршруте", "маршрутах", "маршрутах")} вместо ${w.call_now_base}; план меняется у ${w.changed} ${plural(w.changed, "маршрута", "маршрутов", "маршрутов")}. Пересчитал очередь.`,
      actions: [{ type: "scenario", lead_h: askedH, risk }] };
  }
  if (/пик|нагруз|сколько машин|сводк/.test(s)) {
    const n = networkSummary(board, calls);
    return { text: `Пик сети в ${n.peak_at}: ${vol(n.peak_volume)} за час. Вызвать сейчас: ${n.call_now} ${routesWord(n.call_now)}, в течение часа ещё ${n.call_within_hour}. Новых машин до конца горизонта: ${int(n.new_trucks)}.`,
      actions: [{ type: "filter", value: "now" }] };
  }
  return null;
}

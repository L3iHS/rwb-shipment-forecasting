// Offline answers of the assistant; every number comes from tools.ts.
import type { Board } from "./data";
import { dec1, pct, vol } from "./format";
import { findRoutes, modelQuality, networkForecast, routeForecast, type Action, type Order } from "./tools";

const signed = (v: number) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${dec1(Math.abs(v) * 100)}%`;
const list = (board: Board, order: Order) =>
  findRoutes(board, order, 3).routes.map((r) => `№ ${r.route_id} (${r.error_wape != null ? pct(r.error_wape) : vol(r.volume_4h)})`).join(", ");

/** Offline answer for common questions; null when the question needs the model. */
export function localAnswer(board: Board, q: string): { text: string; actions: Action[] } | null {
  const s = q.toLowerCase();
  const num = s.match(/\d+/)?.[0];
  if (num && /маршрут|№/.test(s)) {
    const r = routeForecast(board, +num);
    if (!r) return { text: `Маршрута № ${num} нет: номера от 0 до ${board.routes.length - 1}.`, actions: [] };
    const check = r.error_wape != null ? ` На проверке ошибка ${pct(r.error_wape)}${r.bias != null ? `, сдвиг ${signed(r.bias)}` : ""}.` : "";
    return {
      text: `№ ${r.route_id}: за 4 ч ${vol(r.volume_4h)}, пик в ${r.peak_at}, ${vol(r.peak_per_hour)} в час. Диапазон ±${Math.round(r.band_80 * 100)}%.${check}`,
      actions: [{ type: "open_route", id: r.route_id }],
    };
  }
  if (/хуже|неточн|ошибается/.test(s)) {
    const top = findRoutes(board, "least_accurate", 1).routes[0];
    return { text: `Больше всего ошибается прогноз у ${list(board, "least_accurate")}.`,
      actions: top ? [{ type: "open_route", id: top.route_id }] : [] };
  }
  if (/лучше|точнее/.test(s) && !/модел/.test(s)) {
    const top = findRoutes(board, "most_accurate", 1).routes[0];
    return { text: `Точнее всего прогноз у ${list(board, "most_accurate")}.`, actions: top ? [{ type: "open_route", id: top.route_id }] : [] };
  }
  if (/крупн|больш/.test(s)) {
    const f = findRoutes(board, "largest", 3);
    return { text: `Больше всего отгрузят ${f.routes.map((r) => `№ ${r.route_id} (${vol(r.volume_4h)})`).join(", ")}.`,
      actions: f.routes.length ? [{ type: "open_route", id: f.routes[0].route_id }] : [] };
  }
  if (/точн|качеств|метрик|провер|довер/.test(s)) {
    const m = modelQuality(board);
    const wape = m.per_day.map((d) => d.wape), bias = m.per_day.map((d) => d.bias);
    return { text: `Ошибка ${m.metric_case_from_1030.toFixed(3).replace(".", ",")} по метрике кейса, факт попал в диапазон прогноза в ${pct(m.band_80_coverage ?? 0)}. ` +
      `По дням ошибка ${pct(Math.min(...wape))}…${pct(Math.max(...wape))}, сдвиг суммы от ${signed(Math.min(...bias))} до ${signed(Math.max(...bias))}.`, actions: [] };
  }
  if (/машин|простой|вызов/.test(s)) {
    return { text: "Машин, их вместимости и времени подачи в данных нет, поэтому здесь только прогноз отгрузок.", actions: [] };
  }
  if (/пик|сколько|отгруз|сводк|прогноз|всего/.test(s)) {
    const n = networkForecast(board);
    return { text: `До ${n.horizon_end} отгрузят ${vol(n.volume_4h)} товаров, пик в ${n.peak_at}: ${vol(n.peak_per_hour)} в час. Диапазон по всем маршрутам ±${Math.round(n.band_80 * 100)}%.`, actions: [] };
  }
  return null;
}

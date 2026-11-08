const nf1 = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 1 });
const nf2 = new Intl.NumberFormat("ru-RU", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

export const int = (v: number) => Math.round(v).toLocaleString("ru-RU");
export const dec1 = (v: number) => nf1.format(v);
export const dec2 = (v: number) => nf2.format(v);
export const pct = (v: number) => nf1.format(v * 100) + "%";
export const signedPct = (v: number) => (v > 0 ? "+" : v < 0 ? "−" : "") + nf1.format(Math.abs(v) * 100) + "%";

/** Volume of goods: 12,3 млн / 450 тыс. */
export function vol(v: number): string {
  const a = Math.abs(v);
  if (a >= 1e9) return nf2.format(v / 1e9) + " млрд";
  if (a >= 1e6) return nf1.format(v / 1e6) + " млн";
  if (a >= 1e3) return int(v / 1e3) + " тыс.";
  return int(v);
}

/** Short axis label: 12м, 450т. */
export function volShort(v: number): string {
  const a = Math.abs(v);
  if (a >= 1e9) return nf1.format(v / 1e9) + "млрд";
  if (a >= 1e6) return nf1.format(v / 1e6) + "м";
  if (a >= 1e3) return Math.round(v / 1e3) + "т";
  return int(v);
}

export function plural(n: number, one: string, few: string, many: string): string {
  const m10 = n % 10, m100 = n % 100;
  if (m10 === 1 && m100 !== 11) return one;
  if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few;
  return many;
}
export const routesWord = (n: number) => plural(n, "маршрут", "маршрута", "маршрутов");

export const hhmm = (d: Date) => d.toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" });

/** Lead time label: 30 мин, 1 ч, 2 ч. */
export const leadLabel = (h: number) => (h < 1 ? `${h * 60} мин` : `${nf1.format(h)} ч`);

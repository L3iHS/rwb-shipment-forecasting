import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import type { Board, Route } from "../data";
import { dec2, hhmm, leadLabel, vol } from "../format";
import { base, flow } from "../motion";
import { clock, inbound, loadsAt, spread, type Risk } from "../plan";
import { Dock, Fan, fanSeries } from "./charts";
import { Disclosure } from "./Disclosure";

function confidence(r: Route, cuts: [number, number]) {
  const s = spread(r);
  if (s <= cuts[0]) return { label: "надёжный", dot: "bg-good", note: "узкий коридор прогноза" };
  if (s <= cuts[1]) return { label: "обычный", dot: "bg-warn", note: "обычный коридор прогноза" };
  return { label: "неуверенный", dot: "bg-critical", note: "широкий коридор: держите резервную машину" };
}

function advice(board: Board, r: Route, call: number, lead: number, leadH: number) {
  const n = inbound(r);
  if (call === 0)
    return { strong: "Вызвать машину сейчас.", text: `К ${hhmm(clock(board, lead))}, когда она доедет, у ворот наберётся полный кузов${n ? `, считая ${n} уже вызванн${n === 1 ? "ую" : "ых"}` : ""}.` };
  if (call > 0)
    return { strong: `Вызвать в ${hhmm(clock(board, call))}.`, text: `Полный кузов наберётся примерно к ${hhmm(clock(board, call + lead))}, а машина едет ${leadLabel(leadH)}. Раньше вызывать — машина будет стоять.` };
  return { strong: `Новый вызов до ${hhmm(clock(board, 8))} не нужен.`, text: n ? "Вызванных машин хватает на ожидаемый груз." : "Груза на полный кузов за это время не наберётся." };
}

interface Props {
  board: Board;
  r: Route;
  call: number;
  lead: number;
  leadH: number;
  risk: Risk;
  cuts: [number, number];
}

export function RouteCard({ board, r, call, lead, leadH, risk, cuts }: Props) {
  const reduced = useReducedMotion();
  const fan = fanSeries(board, r.h, r.p50, r.p10, r.p90);
  const dock = [{ t: hhmm(clock(board, 0)), p50: r.bl, band: [r.bl, r.bl] as [number, number] }].concat(
    // The line is the inflow the call rule decides on (cr, or c90 when cautious), so it meets the level at arrival.
    (risk === "p90" ? r.c90 : r.cr).map((c, k) => {
      const line = r.bl + c / r.cap;
      return { t: hhmm(clock(board, k + 1)), p50: line, band: [line, Math.max(line, loadsAt(r, k, "p90"))] as [number, number] };
    }),
  );
  const a = advice(board, r, call, lead, leadH);
  const conf = confidence(r, cuts);
  return (
    <motion.section layout transition={flow} className="relative overflow-hidden rounded-xl border border-line bg-card p-5">
      <AnimatePresence mode="popLayout" initial={false}>
        <motion.div key={r.id} layout="position" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, transition: { duration: 0.12 } }} transition={base}>
          <div className="eyebrow">Маршрут № {r.id}</div>
          <AnimatePresence mode="wait" initial={false}>
            <motion.div key={a.strong} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}>
              <h2 className={`mt-1 text-xl font-semibold tracking-tight ${call === 0 ? "text-signal" : ""}`}>{a.strong}</h2>
              <p className="mt-1 text-sm text-ink-2">{a.text}</p>
            </motion.div>
          </AnimatePresence>

          <dl className="mt-4 flex flex-wrap gap-x-6 gap-y-1.5 text-sm">
            <div className="flex gap-1.5"><dt className="text-ink-2">У ворот</dt><dd className="num font-medium">{dec2(r.bl)} кузова</dd></div>
            <div className="flex gap-1.5"><dt className="text-ink-2">Едут</dt><dd className="num font-medium">{inbound(r)}</dd></div>
            <div className="flex gap-1.5"><dt className="text-ink-2">Кузов</dt><dd className="num font-medium">{vol(r.cap)}</dd></div>
            <div className="flex items-center gap-1.5" title={conf.note}>
              <dt className="text-ink-2">Прогноз</dt>
              <dd className="flex items-center gap-1.5 font-medium"><span className={`size-2 rounded-full ${conf.dot}`} />{conf.label}</dd>
            </div>
          </dl>

          <div className="mt-5">
            <h3 className="text-sm font-semibold">Груз у ворот, кузовов</h3>
            <Dock data={dock} need={inbound(r) + 1} animate={!reduced}
              call={call >= 0 ? dock[call].t : undefined}
              arrive={call >= 0 && call + lead <= 8 ? dock[call + lead].t : undefined} />
          </div>

        </motion.div>
      </AnimatePresence>
      <div className="mt-4">
        <Disclosure title="Прогноз отгрузок">
          <Fan data={fan.data} nowLabel={fan.nowLabel} animate={!reduced} />
        </Disclosure>
      </div>
    </motion.section>
  );
}

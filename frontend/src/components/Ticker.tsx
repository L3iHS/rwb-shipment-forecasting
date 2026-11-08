import { animate, useReducedMotion } from "motion/react";
import { useEffect, useRef } from "react";
import { int } from "../format";
import { ease } from "../motion";

/** A number that counts to its new value instead of jumping. */
export function Ticker({ value, format = int }: { value: number; format?: (v: number) => string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const shown = useRef(value); // the number currently on screen, mid-count included
  const reduced = useReducedMotion();
  useEffect(() => {
    const from = shown.current;
    if (from === value || !ref.current) return;
    if (reduced) {
      shown.current = value;
      ref.current.textContent = format(value);
      return;
    }
    const c = animate(from, value, {
      duration: 0.5,
      ease,
      onUpdate: (v) => {
        shown.current = v;
        if (ref.current) ref.current.textContent = format(v);
      },
    });
    return () => c.stop();
  }, [value, format, reduced]);
  return <span ref={ref}>{format(value)}</span>;
}

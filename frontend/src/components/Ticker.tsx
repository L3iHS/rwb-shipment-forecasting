import { animate, useReducedMotion } from "motion/react";
import { useEffect, useRef } from "react";
import { int } from "../format";
import { ease } from "../motion";

/** A number that counts to its new value instead of jumping; with `from` it also counts up when it first appears. */
export function Ticker({ value, format = int, from }: { value: number; format?: (v: number) => string; from?: number }) {
  const ref = useRef<HTMLSpanElement>(null);
  const reduced = useReducedMotion();
  const shown = useRef(from != null && !reduced ? from : value); // the number currently on screen, mid-count included
  useEffect(() => {
    const start = shown.current;
    if (start === value || !ref.current) return;
    if (reduced) {
      shown.current = value;
      ref.current.textContent = format(value);
      return;
    }
    const c = animate(start, value, {
      duration: from != null && start === from ? 0.9 : 0.5,
      ease,
      onUpdate: (v) => {
        shown.current = v;
        if (ref.current) ref.current.textContent = format(v);
      },
    });
    return () => c.stop();
  }, [value, format, reduced, from]);
  return <span ref={ref}>{format(shown.current)}</span>;
}

import { useEffect, useRef } from "react";
// Motion tokens: one place for durations and springs. Motion explains state changes, it does not decorate.
export const ease = [0.22, 1, 0.36, 1] as const;
export const fast = { duration: 0.14, ease };
export const base = { duration: 0.24, ease };
export const snap = { type: "spring", stiffness: 500, damping: 40, mass: 0.6 } as const; // pills, highlights
export const flow = { type: "spring", stiffness: 260, damping: 32 } as const; // widths, bars, card height
export const reorder = { type: "spring", stiffness: 400, damping: 42 } as const; // list rows
export const panel = { type: "spring", stiffness: 320, damping: 34 } as const; // assistant drawer

/** True only during the component's first render: entrance staggers run once, not on every update. */
export function useFirstRender(): boolean {
  const first = useRef(true);
  useEffect(() => {
    first.current = false;
  }, []);
  return first.current;
}

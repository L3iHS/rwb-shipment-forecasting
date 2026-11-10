import { createContext, useContext, useEffect, useRef, useState } from "react";
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

const played = new Set<string>();

/** Which dashboard a component belongs to. Each dashboard plays its own entrance: the console's forecast once, and
 *  every result of an uploaded file once. Without the scope the first dashboard would use up the entrance of the second. */
export const IntroScope = createContext("page");

/** True only for the first mount of a component with this key inside its dashboard: the entrance of the sections plays
 *  once, not on every tab switch. The key is marked in an effect, so a StrictMode double render still sees the first value. */
export function useIntro(key: string): boolean {
  const id = `${useContext(IntroScope)}/${key}`;
  const [intro] = useState(() => !played.has(id));
  useEffect(() => {
    played.add(id);
  }, [id]);
  return intro;
}

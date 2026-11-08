import { motion } from "motion/react";
import { useId } from "react";
import { snap } from "../motion";

interface Props<T> {
  value: T;
  options: [T, string][];
  onChange: (v: T) => void;
}

/** A row of mutually exclusive buttons; the selected pill slides between them. */
export function Segmented<T extends string | number>({ value, options, onChange }: Props<T>) {
  const id = useId();
  return (
    <div className="flex rounded-lg bg-line/60 p-0.5">
      {options.map(([v, label]) => (
        <button key={String(v)} onClick={() => onChange(v)} aria-pressed={value === v}
          className="relative rounded-md px-2.5 py-1 text-xs text-ink-2 transition-colors duration-150 aria-pressed:text-ink">
          {value === v && <motion.span layoutId={`seg-${id}`} transition={snap} className="absolute inset-0 rounded-md bg-card shadow-sm" />}
          <span className="relative">{label}</span>
        </button>
      ))}
    </div>
  );
}

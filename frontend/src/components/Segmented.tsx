import { motion } from "motion/react";
import { useId } from "react";
import { snap } from "../motion";

interface Props<T> {
  value: T;
  options: [T, string][];
  onChange: (v: T) => void;
  label?: string;
}

/** A row of mutually exclusive buttons; the selected pill slides between them. */
export function Segmented<T extends string | number>({ value, options, onChange, label }: Props<T>) {
  const id = useId();
  return (
    <div className="flex rounded-[10px] bg-raised p-[3px]" role="group" aria-label={label}>
      {options.map(([v, text]) => (
        <button key={String(v)} onClick={() => onChange(v)} aria-pressed={value === v}
          className="relative rounded-md px-2.5 py-1.5 text-[14px] font-medium leading-[18px] text-ink-2 transition-colors duration-150 hover:text-ink aria-pressed:text-ink">
          {value === v && (
            <motion.span layoutId={`seg-${id}`} transition={snap} className="absolute inset-0 rounded-md bg-surface ring-1 ring-inset ring-line" />
          )}
          <span className="relative">{text}</span>
        </button>
      ))}
    </div>
  );
}

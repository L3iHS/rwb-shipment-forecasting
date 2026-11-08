import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { ChevronDown } from "lucide-react";
import { useState, type ReactNode } from "react";
import { flow, snap } from "../motion";

/** A collapsed section; its content (charts included) mounts only when opened. */
export function Disclosure({ title, hint, children }: { title: string; hint?: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const reduced = useReducedMotion();
  const enter = reduced ? { duration: 0 } : { height: flow, opacity: { duration: 0.2, delay: 0.05 } };
  const leave = reduced ? { duration: 0 } : { height: flow, opacity: { duration: 0.12 } };
  return (
    <div className="border-t border-line pt-3">
      <button aria-expanded={open} onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 text-left text-sm font-medium hover:text-ink">
        <motion.span animate={{ rotate: open ? 180 : 0 }} transition={snap} className="text-muted">
          <ChevronDown size={16} />
        </motion.span>
        {title}
        {hint && <span className="ml-auto text-xs font-normal text-muted">{hint}</span>}
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div key="body" className="overflow-hidden"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1, transition: enter }}
            exit={{ height: 0, opacity: 0, transition: leave }}>
            <div className="pt-3">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

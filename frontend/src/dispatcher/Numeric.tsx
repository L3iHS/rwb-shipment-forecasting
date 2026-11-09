import { useEffect, useRef } from 'react';
import { animate, useReducedMotion } from 'motion/react';
import { amount, number } from './model';
import { transition } from './motion';
export default function Numeric({ value, format = 'amount' }: {
    value: number;
    format?: 'amount' | 'number' | 'percent';
}) {
    const ref = useRef<HTMLSpanElement>(null), current = useRef(value), reduced = useReducedMotion();
    const formatValue = (v: number) => format === 'amount' ? amount(v) : format === 'percent' ? number(v) + '%' : number(v);
    useEffect(() => {
        const update = (v: number) => { current.current = v; if (ref.current)
            ref.current.textContent = formatValue(v); };
        if (reduced) {
            update(value);
            return;
        }
        const controls = animate(current.current, value, { ...transition, onUpdate: update, onComplete: () => update(value) });
        return () => controls.stop();
    }, [value, format, reduced]);
    return <span ref={ref} className="numeric" data-number={value} aria-label={formatValue(value)}>{formatValue(current.current)}</span>;
}

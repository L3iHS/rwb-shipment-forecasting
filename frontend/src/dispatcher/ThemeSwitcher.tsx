import { useEffect, useRef, useState, type KeyboardEvent } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'motion/react';
import Icon from './Icon';
import { followSystemTheme, parseThemePreference, readThemePreference, themeStorageKey, type ThemePreference } from './theme';

const choices = [
    { value: 'light', label: 'Светлая', icon: 'sun' },
    { value: 'dark', label: 'Тёмная', icon: 'moon' },
    { value: 'system', label: 'Системная', icon: 'monitor' },
] as const;

export default function ThemeSwitcher() {
    const [preference, setPreference] = useState<ThemePreference>(readThemePreference);
    const [open, setOpen] = useState(false);
    const root = useRef<HTMLDivElement>(null), trigger = useRef<HTMLButtonElement>(null);
    const menu = useRef<HTMLDivElement>(null);
    const reduced = useReducedMotion();
    const current = choices.find(choice => choice.value === preference)!;

    useEffect(() => followSystemTheme(preference), [preference]);
    useEffect(() => {
        const sync = (event: StorageEvent) => {
            if (event.key === themeStorageKey || event.key === null)
                setPreference(parseThemePreference(event.newValue));
        };
        window.addEventListener('storage', sync);
        return () => window.removeEventListener('storage', sync);
    }, []);
    useEffect(() => {
        if (!open) return;
        menu.current?.querySelector<HTMLButtonElement>('[aria-checked="true"]')?.focus();
        const outside = (event: PointerEvent) => {
            if (!root.current?.contains(event.target as Node)) setOpen(false);
        };
        document.addEventListener('pointerdown', outside);
        return () => document.removeEventListener('pointerdown', outside);
    }, [open]);

    const choose = (value: ThemePreference) => {
        setPreference(value);
        try { window.localStorage.setItem(themeStorageKey, value); } catch { /* Still works for this session. */ }
        setOpen(false);
        trigger.current?.focus();
    };
    const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
        if (event.key === 'Escape' && open) {
            event.preventDefault();
            setOpen(false);
            trigger.current?.focus();
            return;
        }
        if (!open && ['ArrowDown', 'ArrowUp'].includes(event.key)) {
            event.preventDefault();
            setOpen(true);
            return;
        }
        const buttons = Array.from(menu.current?.querySelectorAll<HTMLButtonElement>('[role="menuitemradio"]') ?? []);
        const index = buttons.indexOf(document.activeElement as HTMLButtonElement);
        const next = event.key === 'Home' ? 0 : event.key === 'End' ? 2
            : event.key === 'ArrowDown' ? (index + 1) % 3 : event.key === 'ArrowUp' ? (index + 2) % 3 : -1;
        if (open && next >= 0) { event.preventDefault(); buttons[next]?.focus(); }
    };

    return <div className="theme-switcher" ref={root} onKeyDown={onKeyDown}
        onBlur={event => { if (!event.currentTarget.contains(event.relatedTarget)) setOpen(false); }}>
        <button ref={trigger} className="theme-trigger" aria-label={`Оформление: ${current.label}`}
            title={`Оформление: ${current.label}`} aria-haspopup="menu" aria-expanded={open}
            aria-controls={open ? 'theme-menu' : undefined} onClick={() => setOpen(value => !value)}>
            <Icon name={current.icon}/><span>{current.label}</span><Icon name="chevron"/>
        </button>
        <AnimatePresence>{open && <motion.div id="theme-menu" ref={menu} className="theme-menu"
            role="menu" aria-label="Оформление" initial={{ opacity: 0, y: reduced ? 0 : -4 }}
            animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            transition={{ duration: reduced ? 0 : .16 }}>
            <div className="theme-menu-label">Оформление</div>
            {choices.map(choice => <button key={choice.value} type="button" role="menuitemradio"
                aria-checked={preference === choice.value} tabIndex={-1} onClick={() => choose(choice.value)}>
                <Icon name={choice.icon}/><span>{choice.label}</span>
                <i className="theme-choice-dot" aria-hidden="true"/>
            </button>)}
        </motion.div>}</AnimatePresence>
    </div>;
}

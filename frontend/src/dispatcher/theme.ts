export type ThemePreference = 'light' | 'dark' | 'system';
export const themeStorageKey = 'rwb-theme';
export const systemThemeQuery = '(prefers-color-scheme: dark)';

export function parseThemePreference(value: string | null): ThemePreference {
    return value === 'light' || value === 'dark' ? value : 'system';
}

export function readThemePreference(): ThemePreference {
    try {
        return parseThemePreference(window.localStorage.getItem(themeStorageKey));
    } catch {
        return 'system';
    }
}

export function resolveTheme(preference: ThemePreference, systemDark: boolean) {
    return preference === 'system' ? (systemDark ? 'dark' : 'light') : preference;
}

export function applyTheme(preference: ThemePreference, systemDark: boolean) {
    const resolved = resolveTheme(preference, systemDark);
    document.documentElement.dataset.theme = resolved;
    document.documentElement.style.colorScheme = resolved;
}

export function followSystemTheme(preference: ThemePreference) {
    const media = window.matchMedia(systemThemeQuery);
    const update = () => applyTheme(preference, media.matches);
    update();
    media.addEventListener('change', update);
    return () => media.removeEventListener('change', update);
}

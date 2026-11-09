import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { test } from 'node:test';
import ts from 'typescript';

const source = readFileSync(new URL('../src/dispatcher/theme.ts', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText;
const bootstrap = readFileSync(new URL('../index.html', import.meta.url), 'utf8').match(/<script>([\s\S]*?)<\/script>/)[1];

function environment({ saved = null, dark = false, blocked = false } = {}) {
  const listeners = new Set();
  const media = {
    matches: dark,
    addEventListener: (_, listener) => listeners.add(listener),
    removeEventListener: (_, listener) => listeners.delete(listener),
  };
  const localStorage = { getItem: () => { if (blocked) throw new Error('Storage blocked'); return saved; } };
  const document = { documentElement: { dataset: {}, style: {} } };
  const context = { exports: {}, document, localStorage, matchMedia: () => media, window: { localStorage, matchMedia: () => media } };
  vm.runInNewContext(compiled, context);
  return {
    api: context.exports, context, root: document.documentElement, listeners,
    systemChange(value) { media.matches = value; for (const listener of listeners) listener(); },
  };
}

test('system choice follows live OS changes and removes its listener', () => {
  const env = environment();
  const unsubscribe = env.api.followSystemTheme('system');
  assert.equal(env.root.dataset.theme, 'light');
  env.systemChange(true);
  assert.equal(env.root.dataset.theme, 'dark');
  assert.equal(env.root.style.colorScheme, 'dark');
  unsubscribe();
  assert.equal(env.listeners.size, 0);
  env.systemChange(false);
  assert.equal(env.root.dataset.theme, 'dark');
});

test('explicit choices override the OS and switching back to system uses the current OS', () => {
  for (const preference of ['light', 'dark']) {
    const env = environment({ dark: true });
    const unsubscribe = env.api.followSystemTheme(preference);
    env.systemChange(false);
    assert.equal(env.root.dataset.theme, preference);
    unsubscribe();
    env.api.followSystemTheme('system');
    assert.equal(env.root.dataset.theme, 'light');
  }
});

test('missing, invalid or inaccessible storage falls back to system', () => {
  for (const options of [{}, { saved: 'invalid' }, { blocked: true }])
    assert.equal(environment(options).api.readThemePreference(), 'system');
  for (const saved of ['light', 'dark', 'system'])
    assert.equal(environment({ saved }).api.readThemePreference(), saved);
});

test('pre-paint bootstrap agrees with runtime, including blocked storage', () => {
  for (const saved of [null, 'light', 'dark', 'system', 'invalid']) {
    for (const dark of [false, true]) {
      for (const blocked of [false, true]) {
        const env = environment({ saved, dark, blocked });
        vm.runInNewContext(bootstrap, env.context);
        const expected = env.api.resolveTheme(env.api.readThemePreference(), dark);
        assert.equal(env.root.dataset.theme, expected);
        assert.equal(env.root.style.colorScheme, expected);
      }
    }
  }
});

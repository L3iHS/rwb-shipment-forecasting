"""Collect DEV scores from experiment logs into results.md. Usage: python collect_results.py"""
import re, glob, os
here = os.path.dirname(os.path.abspath(__file__))
rows = {}
for path in sorted(glob.glob(os.path.join(here, 'logs', '*.log'))):
    for line in open(path):
        m = re.match(r'RESULT (\{.*\}) r=(\d+): score ([\d.]+) std ([\d.]+) wape ([\d.]+) bias ([+-][\d.]+) sat ([\d.]+)', line)
        if m:
            cfg, r, sc, sd, w, b, sat = m.groups()
            tag = re.search(r'"tag":"([^"]+)"', cfg).group(1)
            if tag not in rows or float(sc) < rows[tag][1]:
                rows[tag] = (cfg, float(sc), sd, w, b, sat, f'{r} деревьев')
            continue
        m = re.match(r'RESULT (\S+) dev\(14\) score ([\d.]+) wape ([\d.]+) bias ([+-][\d.]+)', line)
        if m:
            tag, sc, w, b = m.groups()
            rows[tag] = ('N-HiTS, см. logs/' + os.path.basename(path), float(sc), '', w, b, '', '')
out = ['# Результаты на DEV (14 срезов в 10:30, 11.10–24.10)', '',
       'Метрика платформы WAPE + |Relative Bias|, среднее по срезам; меньше — лучше. Для бустингов показано лучшее число деревьев.',
       'Собрано скриптом `collect_results.py` из `logs/`.', '',
       '| Тег | DEV | std | WAPE | bias | субботы | деревья | конфиг |', '|---|---:|---:|---:|---:|---:|---|---|']
for tag, (cfg, sc, sd, w, b, sat, r) in sorted(rows.items(), key=lambda kv: kv[1][1]):
    if tag.startswith('N_smoke'): continue
    out.append(f'| {tag} | {sc:.4f} | {sd} | {w} | {b} | {sat} | {r} | `{cfg}` |')
open(os.path.join(here, 'results.md'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out[:20]))

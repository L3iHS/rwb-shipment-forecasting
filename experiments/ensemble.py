"""Blend saved fold predictions. ensemble.py tagA[:row] tagB[:row] ...   (row = which prediction row, default last)
Dev = 14 folds 11.10-24.10. Weights on a coarse simplex grid + global k, chosen on dev only.
With --holdout tagA_H ... (same order) the chosen weights/k are applied once to the 7 holdout folds."""
import sys, os, itertools, numpy as np
d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'preds')
n = 8000
def load(spec, nfold, first=True):
    tag, _, row = spec.partition(':')
    a = np.load(os.path.join(d, tag + '.npy'))
    Y, P = a[0], a[int(row) if row else -1]
    k = len(Y) // n
    sl = slice(0, nfold * n) if (first or k == nfold) else slice((k - nfold) * n, k * n)
    return Y[sl], P[sl]
def score(t, p): w = np.abs(p - t).sum() / t.sum(); b = p.sum() / t.sum() - 1; return w + abs(b)
def cv(Y, P, nf): return np.mean([score(Y[q*n:(q+1)*n], P[q*n:(q+1)*n]) for q in range(nf)])
args = sys.argv[1:]
hold = args[args.index('--holdout') + 1:] if '--holdout' in args else []
dev = args[:args.index('--holdout')] if hold else args
Ys, Ps = zip(*[load(s, 14) for s in dev])
for Yx in Ys: assert np.allclose(Yx, Ys[0]), 'fold mismatch'
Y = Ys[0]
for s, P in zip(dev, Ps): print(f'{s:28s} dev {cv(Y, P, 14):.4f}')
m = len(Ps); grid = np.arange(0, 1.0001, 0.1)
best = (9, None, None)
for w in itertools.product(grid, repeat=m):
    if abs(sum(w) - 1) > 1e-9: continue
    B = sum(wi * Pi for wi, Pi in zip(w, Ps))
    for k in np.arange(0.97, 1.0501, 0.005):
        s = cv(Y, k * B, 14)
        if s < best[0]: best = (s, w, k)
eq = sum(Ps) / m
keq = min(np.arange(0.97, 1.0501, 0.005), key=lambda k: cv(Y, k * eq, 14))
print(f'equal weights k=1: {cv(Y, eq, 14):.4f}   equal + k={keq:.3f}: {cv(Y, keq*eq, 14):.4f}')
print(f'best dev: {best[0]:.4f} weights {np.round(best[1],2)} k={best[2]:.3f}')
if hold:
    Hs, HP = zip(*[load(s, 7, first=False) for s in hold])
    for Yx in Hs: assert np.allclose(Yx, Hs[0]), 'holdout fold mismatch'
    HY = Hs[0]
    for s, P in zip(hold, HP): print(f'{s:28s} holdout {cv(HY, P, 7):.4f}')
    B = sum(wi * Pi for wi, Pi in zip(best[1], HP))
    print(f'HOLDOUT chosen blend: {cv(HY, best[2]*B, 7):.4f}   equal+k: {cv(HY, keq*sum(HP)/m, 7):.4f}')

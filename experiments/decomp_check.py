import sys, numpy as np, pandas as pd
tr = pd.read_parquet(sys.argv[1]).sort_values(['timestamp','route_id'])
T = tr.timestamp.nunique(); R = tr.route_id.nunique()
y = tr.target_1h.to_numpy(float).reshape(T, R)
st = tr[[f'status_{i}' for i in range(1,7)]].to_numpy(float).reshape(T, R, 6)
print('integer target?', np.allclose(y, np.round(y)))
# reconstruct half-hour flow x with x_t + x_{t-1} = y_t, anchored at a zero of y
neg_share, anchored = [], 0
X = np.full((T, R), np.nan)
for r in range(R):
    z = np.where(y[:, r] == 0)[0]
    if len(z) == 0: continue
    anchored += 1
    a = z[0]; x = np.empty(T); x[a] = 0.0
    for t in range(a+1, T): x[t] = y[t, r] - x[t-1]
    for t in range(a-1, -1, -1): x[t] = y[t+1, r] - x[t+1]
    X[:, r] = x
    neg_share.append((x < 0).mean())
print('routes with zero anchor', anchored, 'of', R)
ns = np.array(neg_share); print('share negative x per route: median %.4f  max %.4f  routes with any neg %d' % (np.median(ns), ns.max(), (ns > 0).sum()))
# consistency: at other zeros x should be 0
bad = 0; tot = 0
for r in range(R):
    z = np.where(y[:, r] == 0)[0]
    if len(z) < 2: continue
    tot += len(z); bad += (np.abs(X[z, r]) > 1e-6).sum()
print('zeros where reconstructed x != 0:', bad, 'of', tot)
# compare x with statuses (which status is the half-hour shipped flow?)
ok = ~np.isnan(X[:, 0])
for k in range(6):
    for lag in range(0, 4):
        a = X[lag:, :].ravel(); b = st[:T-lag, :, k].ravel()
        m = ~np.isnan(a)
        print(f'corr x_t vs status_{k+1}_(t-{lag}) = {np.corrcoef(a[m], b[m])[0,1]:.3f}', end='  ')
    print()

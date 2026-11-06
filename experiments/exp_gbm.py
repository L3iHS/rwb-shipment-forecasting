"""LightGBM direct multi-horizon experiments. Usage: exp_gbm.py train.parquet '<json config>' [test.parquet out.csv]"""
import sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
t0 = time.time()
CFG = dict(tag=None, h1_struct=False, algo='lgb', level='mean', clip=False, feats='v1', train_days=42, origin_slots=[16, 26], objective='l2',
           rounds=[200, 300, 400, 600], lr=0.05, leaves=63, min_leaf=200, route_cat=True, per_h=False,
           n_val=14, seeds=[42], tweedie_p=1.3, val_end_offset=7)
CFG.update(json.loads(sys.argv[2]))
tr = pd.read_parquet(sys.argv[1]).sort_values(['timestamp', 'route_id'])
times = pd.DatetimeIndex(tr.timestamp.unique()); T = len(times); R = 1000; H = 8; D = 48
y_raw = tr.target_1h.to_numpy(np.float64).reshape(T, R)
st = tr[[f'status_{i}' for i in range(1, 7)]].to_numpy(np.float64).reshape(T, R, 6)
slot = (times.hour * 2 + times.minute // 30).to_numpy(); dow = times.day_of_week.to_numpy()

# causal clipping of spikes: cap each value by 4x the route's trailing 7-day median (computed per day)
if CFG['clip']:
    y = y_raw.copy()
    day_start = np.where(slot == 0)[0]
    for s in day_start:
        if s < 7 * D: continue
        cap = 4 * np.median(y_raw[s - 7 * D:s], 0) + 1
        e = min(s + D, T)
        y[s:e] = np.minimum(y_raw[s:e], cap)
else:
    y = y_raw
cs = np.vstack([np.zeros((1, R)), np.cumsum(y, 0)])
scs = np.concatenate([np.zeros((1, R, 6)), np.cumsum(st, 0)])
def rmean(i, n): return (cs[i + 1] - cs[i + 1 - n]) / n
def smean(i, n): return (scs[i + 1] - scs[i + 1 - n]) / n
gtot = y.sum(1); gcs = np.concatenate([[0], np.cumsum(gtot)])
def gmean(i, n): return (gcs[i + 1] - gcs[i + 1 - n]) / n
# half-hour flow x_t with y_t = x_t + x_{t-1}: alternating sums give lo <= x_t <= hi
KX = 16
A = np.zeros((T, R)); lo = np.zeros((T, R)); hi = y_raw.copy()
for j in range(KX):
    sh = np.zeros((T, R)); sh[j:] = y_raw[:T - j]
    A = A + ((-1) ** j) * sh
    if j % 2 == 0: hi = np.minimum(hi, A)      # odd number of terms -> upper bound
    else: lo = np.maximum(lo, A)               # even number of terms -> lower bound
feas = lo <= hi
XH = np.where(feas, (lo + hi) / 2, y_raw / 2); XW = np.where(feas, hi - lo, y_raw)
XH[:KX] = y_raw[:KX] / 2
gst = st.sum(1)
gst_cs = np.vstack([np.zeros((1, 6)), np.cumsum(gst, 0)])
def gs_sum(i, n, cols): return (gst_cs[i + 1, cols] - gst_cs[i + 1 - n, cols]).sum()
def g_td(i, n, cols):   # last n steps vs the same window on the previous 7 days
    return np.log((gs_sum(i, n, cols) + 1) / (np.mean([gs_sum(i - k * D, n, cols) for k in range(1, 8)]) + 1))
_lvl_cache = {}
def level(i):
    if CFG['level'] == 'mean': return rmean(i, 7 * D) + 1.0
    if CFG['level'] == 'median':
        return np.median(y[i - 7 * D + 1:i + 1], 0) + 1.0
    if CFG['level'] == 'mean14': return rmean(i, 14 * D) + 1.0
    raise ValueError

def feats(i):
    lvl = level(i)
    F = {}
    for j in range(8): F[f'lag{j}'] = y[i - j] / lvl
    for n, name in ((6, 'm3h'), (D, 'm1d'), (3 * D, 'm3d'), (7 * D, 'm7d'), (14 * D, 'm14d'), (28 * D, 'm28d')):
        F[name] = rmean(i, n) / lvl
    F['sd1d'] = y[i - D + 1:i + 1].std(0) / lvl
    F['loglvl'] = np.log(lvl)
    F['zero_share1d'] = (y[i - D + 1:i + 1] == 0).mean(0)
    nl = 3 if CFG['feats'] == 'v1' else 8
    for k in range(6):
        for j in range(nl): F[f's{k+1}_l{j}'] = st[i - j, :, k] / lvl
        F[f's{k+1}_m1d'] = smean(i, D)[:, k] / lvl
        if CFG['feats'] != 'v1':
            F[f's{k+1}_m3h'] = smean(i, 6)[:, k] / lvl
            F[f's{k+1}_ratio'] = smean(i, 6)[:, k] / (smean(i, 7 * D)[:, k] + 1)
    F['cur_sum'] = st[i, :, 0:3].sum(1) / lvl; F['prev_sum'] = st[i, :, 3:6].sum(1) / lvl
    if CFG['feats'] != 'v1':
        for n in (6, 12, 24):
            # backlog proxy: statuses flowing in over n steps minus half-hour shipments (y/2) over n steps
            F[f'backlog3_{n}'] = (smean(i, n)[:, 2] - rmean(i, n) / 2) * n / lvl
            F[f'backlog12_{n}'] = (smean(i, n)[:, 0:2].sum(1) - rmean(i, n)) * n / lvl
    if CFG['feats'] == 'v3':
        for j in range(4): F[f'x{j}'] = XH[i - j] / lvl
        F['xshare'] = XH[i] / (y_raw[i] + 1); F['xwidth'] = XW[i] / lvl
        F['bl1_16'] = smean(i, 16)[:, 0] / (smean(i, 7 * D)[:, 0] + 1) - rmean(i, 16) / (rmean(i, 7 * D) + 1)
        v5 = gst[:, 4]; F['gp5_d1d'] = (v5[i] - v5[i - D]) / (v5[i - 7 * D + 1:i + 1].mean() + 1)
        F['g_ytd6'] = np.log((gtot[i - 5:i + 1].sum() + 1) / (np.mean([gtot[i - 5 - k * D:i + 1 - k * D].sum() for k in range(1, 8)]) + 1))
        F['g_scur_td6'] = g_td(i, 6, [0, 1, 2]); F['g_sprev_td6'] = g_td(i, 6, [3, 4, 5])
    F['g_last'] = gtot[i] / gmean(i, 7 * D); F['g_m3h'] = gmean(i, 6) / gmean(i, 7 * D); F['g_m1d'] = gmean(i, D) / gmean(i, 7 * D)
    base = pd.DataFrame(F)
    out = []
    for h in range(1, H + 1):
        b = base.copy(); tgt = i + h
        b['h'] = h; b['slot_t'] = (slot[i] + h) % 48
        b['dow_t'] = dow[i] if slot[i] + h < 48 else (dow[i] + 1) % 7
        b['slot_o'] = slot[i]
        b['yday'] = y[tgt - D] / lvl; b['y2day'] = y[tgt - 2 * D] / lvl; b['yweek'] = y[tgt - 7 * D] / lvl
        b['slotmean7'] = np.mean([y[tgt - k * D] for k in range(1, 8)], 0) / lvl
        if CFG['feats'] != 'v1':
            b['slotmed7'] = np.median([y[tgt - k * D] for k in range(1, 8)], 0) / lvl
            b['slotmean14'] = np.mean([y[tgt - k * D] for k in range(1, 15)], 0) / lvl
            b['dowslot4'] = np.mean([y[tgt - k * 7 * D] for k in range(1, 5)], 0) / lvl
            b['g_slot7'] = np.mean([gtot[tgt - k * D] for k in range(1, 8)]) / gmean(i, 7 * D)
            b['g_dowslot4'] = np.mean([gtot[tgt - k * 7 * D] for k in range(1, 5)]) / gmean(i, 7 * D)
            b['yday_nb'] = (y[tgt - D - 1] + y[tgt - D + 1]) / 2 / lvl
        if CFG['feats'] == 'v3':
            b['xslot7'] = np.mean([XH[tgt - k * D] for k in range(1, 8)], 0) / lvl
            b['xslot7_prev'] = np.mean([XH[tgt - 1 - k * D] for k in range(1, 8)], 0) / lvl
            b['known'] = XH[i] / lvl if h == 1 else 0.0
            b['h1_naive'] = (XH[i] + np.mean([XH[i + 1 - k * D] for k in range(1, 8)], 0)) / lvl if h == 1 else 0.0
        b['route_id'] = np.arange(R)
        out.append(b)
    return pd.concat(out, ignore_index=True), lvl

def build(origins, with_y=True):
    Xs, Ys, Ls = [], [], []
    for i in origins:
        X, lvl = feats(i); Xs.append(X); Ls.append(np.tile(lvl, H))
        if with_y: Ys.append(np.concatenate([y_raw[i + h] for h in range(1, H + 1)]))
    return pd.concat(Xs, ignore_index=True), (np.concatenate(Ys) if with_y else None), np.concatenate(Ls)

def score(t, p):
    w = np.abs(p - t).sum() / t.sum(); b = p.sum() / t.sum() - 1; return w + abs(b), w, b

lo_s, hi_s = CFG['origin_slots']
def train_origins(end_idx):  # all labels <= end_idx
    lo = end_idx - H - CFG['train_days'] * D
    return [i for i in range(max(lo, 29 * D), end_idx - H + 1) if lo_s <= slot[i] <= hi_s]

def fit_predict(Xtr, Ytr, Ltr, Xte, Lte):
    preds = {r: [] for r in CFG['rounds']}
    obj = CFG['objective']
    p = dict(objective=obj, learning_rate=CFG['lr'], num_leaves=CFG['leaves'], min_data_in_leaf=CFG['min_leaf'],
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=10)
    if obj == 'tweedie': p['tweedie_variance_power'] = CFG['tweedie_p']
    cols = [c for c in Xtr.columns if CFG['route_cat'] or c != 'route_id']
    cat = ['route_id'] if CFG['route_cat'] else []
    groups = [None] if not CFG['per_h'] else list(range(1, H + 1))
    for seed in CFG['seeds']:
        p['seed'] = seed
        out = {r: np.zeros(len(Xte)) for r in CFG['rounds']}
        for g in groups:
            mtr = np.ones(len(Xtr), bool) if g is None else (Xtr.h.values == g)
            mte = np.ones(len(Xte), bool) if g is None else (Xte.h.values == g)
            Xa, ya, wa, Xb = Xtr.loc[mtr, cols], Ytr[mtr] / Ltr[mtr], Ltr[mtr], Xte.loc[mte, cols]
            if CFG['h1_struct']:   # h=1: half of the target (x_i) is already observed; learn only the unknown half
                o_tr = np.where(Xtr.h.values[mtr] == 1, Xtr.known.values[mtr], 0.0)
                ya = np.maximum(ya - o_tr, 0)
            o_te = np.where(Xte.h.values[mte] == 1, Xte.known.values[mte], 0.0) if CFG['h1_struct'] else 0.0
            if CFG['algo'] == 'lgb':
                m = lgb.train(p, lgb.Dataset(Xa, ya, weight=wa, categorical_feature=cat), num_boost_round=max(CFG['rounds']))
                pr = lambda r: m.predict(Xb, num_iteration=r)
            elif CFG['algo'] == 'xgb':
                import xgboost as xgb
                xo = {'l2': 'reg:squarederror', 'l1': 'reg:absoluteerror', 'tweedie': 'reg:tweedie', 'poisson': 'count:poisson'}[obj]
                xp = dict(objective=xo, eta=CFG['lr'], max_depth=8, min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
                          tree_method='hist', nthread=10, seed=seed, reg_lambda=1.0)
                if obj == 'tweedie': xp['tweedie_variance_power'] = CFG['tweedie_p']
                Xa2, Xb2 = Xa.drop(columns=['route_id'], errors='ignore'), Xb.drop(columns=['route_id'], errors='ignore')
                m = xgb.train(xp, xgb.DMatrix(Xa2, ya, weight=wa), num_boost_round=max(CFG['rounds']))
                db = xgb.DMatrix(Xb2); pr = lambda r: m.predict(db, iteration_range=(0, r))
            else:
                from catboost import CatBoostRegressor, Pool
                co = {'l2': 'RMSE', 'l1': 'MAE', 'tweedie': f"Tweedie:variance_power={CFG['tweedie_p']}", 'poisson': 'Poisson'}[obj]
                m = CatBoostRegressor(loss_function=co, iterations=max(CFG['rounds']), learning_rate=CFG['lr'] * 2, depth=8,
                                      thread_count=10, random_seed=seed, verbose=0, border_count=128)
                m.fit(Pool(Xa, ya, weight=wa, cat_features=cat)); pb = Pool(Xb, cat_features=cat)
                pt = 'Exponent' if obj in ('tweedie', 'poisson') else 'RawFormulaVal'; pr = lambda r: m.predict(pb, ntree_end=r, prediction_type=pt)
            for r in CFG['rounds']:
                out[r][mte] = (np.maximum(pr(r), 0) + o_te) * Lte[mte]
        for r in CFG['rounds']: preds[r].append(out[r])
    imp = pd.Series(m.feature_importance('gain'), index=cols) if CFG['algo']=='lgb' else pd.Series(dtype=float)
    return {r: np.mean(v, 0) for r, v in preds.items()}, imp

o1030 = [i for i in range(T) if slot[i] == 21 and i + H < T]
if len(sys.argv) > 3:  # final fit on everything, predict test
    last = T - 1
    Xtr, Ytr, Ltr = build(train_origins(last))
    Xte, _, Lte = build([last], with_y=False)
    preds, _ = fit_predict(Xtr, Ytr, Ltr, Xte, Lte)
    r = CFG['final_rounds']; pred = preds[r] * CFG.get('k', 1.0)
    fut = pd.DataFrame({'route_id': np.tile(np.arange(R), H),
                        'timestamp': np.repeat([times[last] + pd.Timedelta(minutes=30 * h) for h in range(1, H + 1)], R), 'y_pred': pred})
    test = pd.read_parquet(sys.argv[3])
    sub = test.merge(fut, on=['route_id', 'timestamp'], how='left')[['id', 'y_pred']].sort_values('id')
    assert len(sub) == 8000 and sub.y_pred.notna().all() and (sub.y_pred >= 0).all()
    sub.to_csv(sys.argv[4], index=False); print('written', sys.argv[4], 'sum', sub.y_pred.sum()); sys.exit()

val_o = o1030[len(o1030) - CFG['n_val'] - CFG['val_end_offset']: len(o1030) - CFG['val_end_offset']]
Xtr, Ytr, Ltr = build(train_origins(val_o[0]))
Xva, Yva, Lva = build(val_o)
preds, imp = fit_predict(Xtr, Ytr, Ltr, Xva, Lva)
n = R * H
name = sys.argv[2]
for r, pv in preds.items():
    sc = np.array([score(Yva[k * n:(k + 1) * n], pv[k * n:(k + 1) * n]) for k in range(len(val_o))])
    ks = {k: np.mean([score(Yva[q * n:(q + 1) * n], k * pv[q * n:(q + 1) * n])[0] for q in range(len(val_o))]) for k in (0.98, 0.99, 1.01, 1.02, 1.03)}
    sat = np.mean([s[0] for s, i in zip(sc, val_o) if dow[i] == 5])
    print(f'RESULT {name} r={r}: score {sc[:,0].mean():.4f} std {sc[:,0].std():.4f} wape {sc[:,1].mean():.4f} bias {sc[:,2].mean():+.4f} sat {sat:.4f} | ' + ' '.join(f'k{k}:{v:.4f}' for k, v in ks.items()))
import os, hashlib
tag = CFG.get('tag') or hashlib.md5(name.encode()).hexdigest()[:8]
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'preds'), exist_ok=True)
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'preds', f'{tag}.npy'), np.vstack([Yva, *[preds[r] for r in CFG['rounds']]]))
print('top imp', (imp / imp.sum()).sort_values(ascending=False).head(12).round(3).to_dict())
print(f'rows {len(Xtr)}  time {time.time()-t0:.0f}s')

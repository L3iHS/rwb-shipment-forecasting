"""N-HiTS via neuralforecast. Modes:
  cv:    exp_nhits.py train.parquet cv  '<json cfg>'   -> preds/<tag>.npy (21 daily 10:30 windows up to 31.10; first 14 = dev, last 7 = holdout)
  final: exp_nhits.py train.parquet final '<json cfg>' test.parquet out.npy -> predictions for test ordered h-major, route 0..999
"""
import sys, os, json, time, logging, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore'); logging.getLogger('pytorch_lightning').setLevel(logging.ERROR); logging.getLogger('lightning').setLevel(logging.ERROR)
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MAE, HuberLoss, MSE
t0 = time.time()
mode = sys.argv[2]
CFG = dict(tag='nhits', input_size=336, loss='mae', max_steps=1500, lr=1e-3, seed=1, batch_size=128, windows_batch_size=512,
           hist=['status_1', 'status_2', 'status_3', 'status_4', 'status_5', 'status_6'], scaler='robust',
           pools=[2, 2, 1], freq_ds=[24, 6, 1], mlp=512, dropout=0.0, start='2025-08-15', xh=False)
CFG.update(json.loads(sys.argv[3]))
tr = pd.read_parquet(sys.argv[1])
tr = tr[tr.timestamp >= CFG['start']]
df = tr.rename(columns={'route_id': 'unique_id', 'timestamp': 'ds', 'target_1h': 'y'})[['unique_id', 'ds', 'y', *CFG['hist']]].copy()
def add_cal(d):
    hh = d.ds.dt.hour + d.ds.dt.minute / 60
    d['tod_s'] = np.sin(2 * np.pi * hh / 24); d['tod_c'] = np.cos(2 * np.pi * hh / 24)
    dw = d.ds.dt.dayofweek
    d['dow_s'] = np.sin(2 * np.pi * dw / 7); d['dow_c'] = np.cos(2 * np.pi * dw / 7)
    d['wkend'] = (dw >= 5).astype(float)
    return d
if CFG['xh']:  # reconstructed half-hour flow x_t (y_t = x_t + x_{t-1}) from a 16-step alternating-sum window
    piv = tr.pivot(index='timestamp', columns='route_id', values='target_1h').sort_index()
    yv = piv.to_numpy(np.float64); T_, R_ = yv.shape
    A = np.zeros_like(yv); lo = np.zeros_like(yv); hi = yv.copy()
    for j in range(16):
        sh = np.zeros_like(yv); sh[j:] = yv[:T_ - j]; A = A + ((-1) ** j) * sh
        if j % 2 == 0: hi = np.minimum(hi, A)
        else: lo = np.maximum(lo, A)
    xh = np.where(lo <= hi, (lo + hi) / 2, yv / 2); xh[:16] = yv[:16] / 2
    xdf = pd.DataFrame(xh, index=piv.index, columns=piv.columns).stack().rename('xh').reset_index()
    xdf.columns = ['ds', 'unique_id', 'xh']
    df = df.merge(xdf, on=['unique_id', 'ds'], how='left')
    CFG['hist'] = CFG['hist'] + ['xh']
df = add_cal(df)
futr = ['tod_s', 'tod_c', 'dow_s', 'dow_c', 'wkend']
loss = {'mae': MAE(), 'huber': HuberLoss(), 'mse': MSE()}[CFG['loss']]
model = NHITS(h=8, input_size=CFG['input_size'], loss=loss, valid_loss=MAE(), max_steps=CFG['max_steps'], learning_rate=CFG['lr'],
              stack_types=['identity'] * 3, n_blocks=[1, 1, 1], mlp_units=[[CFG['mlp'], CFG['mlp']]] * 3,
              n_pool_kernel_size=CFG['pools'], n_freq_downsample=CFG['freq_ds'], dropout_prob_theta=CFG['dropout'],
              scaler_type=CFG['scaler'], hist_exog_list=CFG['hist'], futr_exog_list=futr,
              batch_size=CFG['batch_size'], windows_batch_size=CFG['windows_batch_size'], random_seed=CFG['seed'],
              enable_progress_bar=False, enable_model_summary=False, logger=False, accelerator='mps', devices=1)
nf = NeuralForecast(models=[model], freq='30min')
R, H = 1000, 8
outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'preds'); os.makedirs(outdir, exist_ok=True)
if mode == 'cv':
    d = df[df.ds <= '2025-10-31 14:30']
    cv = nf.cross_validation(df=d, n_windows=21, step_size=48, refit=False)
    cv = cv.sort_values(['cutoff', 'ds', 'unique_id'])
    cutoffs = sorted(cv.cutoff.unique()); assert len(cutoffs) == 21 and all(pd.Timestamp(c).hour == 10 and pd.Timestamp(c).minute == 30 for c in cutoffs), cutoffs
    Y = cv.y.to_numpy(); P = np.maximum(cv.NHITS.to_numpy(), 0)
    np.save(os.path.join(outdir, CFG['tag'] + '.npy'), np.vstack([Y, P]))
    def score(t, p): w = np.abs(p - t).sum() / t.sum(); b = p.sum() / t.sum() - 1; return w + abs(b), w, b
    n = R * H
    sc = np.array([score(Y[k*n:(k+1)*n], P[k*n:(k+1)*n]) for k in range(21)])
    print(f"RESULT {CFG['tag']} dev(14) score {sc[:14,0].mean():.4f} wape {sc[:14,1].mean():.4f} bias {sc[:14,2].mean():+.4f} | first cutoff {cutoffs[0]}")
else:
    nf.fit(df=df)
    test = pd.read_parquet(sys.argv[4]).rename(columns={'route_id': 'unique_id', 'timestamp': 'ds'})
    fut = add_cal(test[['unique_id', 'ds']].copy())
    p = nf.predict(futr_df=fut).sort_values(['ds', 'unique_id'])
    np.save(sys.argv[5], np.maximum(p.NHITS.to_numpy(), 0))
    print('final written', sys.argv[5])
print(f'time {time.time()-t0:.0f}s')

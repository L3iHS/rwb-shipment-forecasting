import sys, numpy as np, pandas as pd
tr = pd.read_parquet(sys.argv[1]).sort_values(['timestamp','route_id'])
T = tr.timestamp.nunique(); R = 1000
y_raw = tr.target_1h.to_numpy(np.float64).reshape(T, R)
KX=24; A=np.zeros((T,R)); lo=np.zeros((T,R)); hi=y_raw.copy()
for j in range(KX):
    sh=np.zeros((T,R)); sh[j:]=y_raw[:T-j]; A=A+((-1)**j)*sh
    if j%2==0: hi=np.minimum(hi,A)
    else: lo=np.maximum(lo,A)
feas=lo<=hi; XH=np.where(feas,(lo+hi)/2,y_raw/2)
t=slice(100,None)
rec=XH[t]+np.roll(XH,1,0)[t]
print('feasible share %.4f' % feas[t].mean())
print('relative abs err of x_t+x_{t-1} vs y: %.4f' % (np.abs(rec-y_raw[t]).sum()/y_raw[t].sum()))
print('median rel width (hi-lo)/y: %.3f' % np.median(((hi-lo)/(y_raw+1))[t][feas[t]]))
print('x share of y: mean %.3f' % (XH[t].sum()/y_raw[t].sum()))

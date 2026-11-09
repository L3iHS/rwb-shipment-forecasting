import sys, numpy as np, pandas as pd
tr = pd.read_parquet(sys.argv[1]).sort_values(['timestamp','route_id'])
times = pd.DatetimeIndex(tr.timestamp.unique()); T=len(times); R=1000
y = tr.target_1h.to_numpy(float).reshape(T, R)
H=8; D=48
def score(t,p):
    w=np.abs(p-t).sum()/t.sum(); b=p.sum()/t.sum()-1; return w+abs(b), w, b
# origins: every day at 10:30 with a full horizon, last 35 days
o_all=[i for i,ts in enumerate(times) if ts.hour==10 and ts.minute==30 and i+H<T]
origins=o_all[-35:]
def level(i, days, kind):
    h=y[i+1-days*D:i+1]
    return h.mean(0) if kind=='mean' else np.median(h,0)
def profile(i, days, sat_only=False):
    # ratio of slot mean to daily mean, global over routes, for the 8 target slots
    idx=np.arange(i+1-days*D, i+1)
    ts=times[idx]; tot=y[idx].sum(1)
    df=pd.DataFrame({'tot':tot,'slot':ts.hour*2+ts.minute//30,'day':ts.normalize(),'dow':ts.dayofweek})
    if sat_only: df=df[df.dow==5]
    df['rel']=df.tot/df.groupby('day').tot.transform('mean')
    p=df.groupby('slot').rel.mean()
    slots=[(times[i]+pd.Timedelta(minutes=30*(h+1))) for h in range(H)]
    return np.array([p[s.hour*2+s.minute//30] for s in slots])
rows=[]
cache={}
for i in origins:
    truth=y[i+1:i+1+H].T
    last=y[i]
    for days in (3,5,7,14):
        for kind in ('mean','median'):
            lv=level(i,days,kind)
            for pname in ('flat','prof28','prof14'):
                pr=np.ones(H) if pname=='flat' else profile(i,int(pname[4:]))
                base=lv[:,None]*pr[None,:]
                cache[(i,days,kind,pname)]=(base,truth,last)
                s,w,b=score(truth,base)
                rows.append(dict(origin=times[i],sat=times[i].dayofweek==5,days=days,kind=kind,prof=pname,score=s,wape=w,bias=b))
res=pd.DataFrame(rows)
g=res.groupby(['days','kind','prof']).agg(score=('score','mean'),std=('score','std'),wape=('wape','mean'),bias=('bias','mean')).sort_values('score')
print(g.head(12).round(4))
best=g.index[0]; print('best',best)
# step-wise blend with last value + multiplier k, fitted on first 25 origins, tested on last 10
fit_o, test_o = origins[:-10], origins[-10:]
def blend_pred(i,a,k):
    base,truth,last=cache[(i,)+best]
    lv_last=last[:,None]
    p=(a[None,:]*lv_last + (1-a[None,:])*base)*k
    return p,truth
# fit a_h by grid per step on fit origins (WAPE per step)
a=np.zeros(H)
for h in range(H):
    bestv=None
    for cand in np.linspace(0,0.8,17):
        tot=0
        for i in fit_o:
            base,truth,last=cache[(i,)+best]
            p=cand*last+(1-cand)*base[:,h]
            tot+=np.abs(p-truth[:,h]).sum()/truth[:,h].sum()
        if bestv is None or tot<bestv: bestv,a[h]=tot,cand
print('a per step',a)
for k in np.arange(0.97,1.071,0.01):
    sf=np.mean([score(*blend_pred(i,a,k)[::-1][::-1][1:2]+blend_pred(i,a,k)[0:1])[0] if False else score(blend_pred(i,a,k)[1],blend_pred(i,a,k)[0])[0] for i in fit_o])
    st=np.mean([score(blend_pred(i,a,k)[1],blend_pred(i,a,k)[0])[0] for i in test_o])
    ss=np.mean([score(blend_pred(i,a,k)[1],blend_pred(i,a,k)[0])[0] for i in origins if times[i].dayofweek==5])
    print(f'k={k:.2f} fit {sf:.4f} holdout10 {st:.4f} saturdays {ss:.4f}')
base_only=np.mean([score(cache[(i,)+best][1],cache[(i,)+best][0])[0] for i in test_o])
print('holdout10 base only', round(base_only,4))

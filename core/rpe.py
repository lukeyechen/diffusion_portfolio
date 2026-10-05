"""Frozen monthly five-asset RPE prototype, independent of market test outcomes."""
from functools import lru_cache
from pathlib import Path
import json
import base64
import numpy as np
import pandas as pd
import torch
from torch import nn
from .feasible_tuning import TradingComparison,TuningSettings,trace_calibration,historical_backtest,latest_portfolios
from .portfolio_rules import compute_weights
from .turnover_upgrade import solve_mv_turnover_aware,partial_rebalance
from .short_horizon_portfolio import project_long_only_capped,performance_metrics

MODEL=Path(__file__).resolve().parents[1]/'models/rpe/prototype.json'
RPE_NAMES=('RPE (no TC)','RPE (TC)','RPE predictive (no TC)','RPE predictive (TC)')

class PosteriorNet(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(53,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,20))
    def forward(self,z,t,c):
        f=t[:,None]*torch.tensor([1.,2.,4.,8.])[None]*np.pi
        return self.net(torch.cat([z,c,torch.sin(f),torch.cos(f)],1))

@lru_cache(maxsize=1)
def load_model():
    payload=json.loads(MODEL.read_text())
    with torch.random.fork_rng(devices=[]):
        model=PosteriorNet()
    state={k:torch.from_numpy(np.frombuffer(base64.b64decode(v['data']),dtype='<f4').copy().reshape(v['shape'])) for k,v in payload['state'].items()}
    model.load_state_dict(state); model.eval()
    return model,tuple(np.asarray(x) for x in payload['normalization'])

@torch.no_grad()
def posterior_moments(sample,*,draws=500,seed=731):
    x=np.asarray(sample,float)
    if x.shape!=(120,5) or not np.isfinite(x).all(): raise ValueError('The frozen RPE prototype requires 120 monthly observations for exactly five assets.')
    if not 100<=draws<=1000: raise ValueError('Use 100–1000 posterior draws.')
    mean=x.mean(0); s=(x-mean).T@(x-mean)/120
    l=np.linalg.cholesky(s); d=np.sqrt(np.diag(s)); tri=np.tril_indices(5); diagonal=np.flatnonzero(tri[0]==tri[1])
    condition=np.r_[mean/d,(s/d[:,None]/d[None,:])[tri],np.log(d/.075)]
    model,(zm,zs,cm,cs)=load_model(); condition=torch.tensor((condition-cm)/cs,dtype=torch.float32).repeat(draws,1)
    # Per-call generator avoids changing other sessions' random streams.
    generator=torch.Generator().manual_seed(int(seed)); z=torch.randn(draws,20,generator=generator)
    beta=torch.linspace(.0001,.2,50); alpha=1-beta; abar=torch.cumprod(alpha,0)
    for i in reversed(range(50)):
        eps=model(z,torch.full((draws,),i/49),condition)
        z=(z-beta[i]/torch.sqrt(1-abar[i])*eps)/torch.sqrt(alpha[i])
        if i: z+=torch.sqrt(beta[i]*(1-abar[i-1])/(1-abar[i]))*torch.randn(z.shape,generator=generator)
    decoded=z.numpy()*zs+zm; mus=mean+decoded[:,:5]*d
    chol=np.zeros((draws,5,5)); packed=decoded[:,5:].copy(); packed[:,diagonal]=np.exp(packed[:,diagonal]); chol[:,tri[0],tri[1]]=packed
    full=l[None]@chol; covs=full@full.swapaxes(-1,-2)
    if not np.isfinite(mus).all() or not np.isfinite(covs).all(): raise ValueError('Posterior is non-finite; these inputs are outside the prototype numerical range.')
    return mus.mean(0),covs.mean(0),np.cov(mus,rowvar=False)


def targets(mean,cov,uncertainty,trading,gamma,previous):
    result={}
    for label,h in [('RPE',cov),('RPE predictive',cov+uncertainty)]:
        result[label+' (no TC)']=compute_weights('Mean-Variance',mean,h,gamma=gamma,constraint_mode='Long-only',max_long_weight=trading.cap)
        name=label+' (TC)'; prev=previous.get(name,np.full(5,.2))
        target=solve_mv_turnover_aware(mean,h,prev,trading.turnover_penalty,gamma=gamma,max_long_weight=trading.cap)
        result[name]=project_long_only_capped(partial_rebalance(target,prev,trading.rebalance_alpha),trading.cap)
    return result


def compare(returns,*,replay_start,calibration_start,gamma=3.,cap=.4,cost_bps=25.,penalty_bps=25.,alpha=.5,draws=500,seed=731):
    if returns.shape[1]!=5 or len(returns)<122: raise ValueError('Supply at least 122 monthly return rows for exactly five assets; more history is needed for calibration.')
    dates=pd.DatetimeIndex(returns.index)
    if not dates.is_monotonic_increasing or not dates.is_unique or dates.to_period('M').duplicated().any(): raise ValueError('Returns must have one row per month, in date order.')
    if np.any(np.diff(dates.to_period('M').asi8)!=1): raise ValueError('Monthly history must be contiguous; missing months cannot be treated as consecutive returns.')
    values=returns.to_numpy(float)
    if not np.isfinite(values).all() or np.any(values<=-1): raise ValueError('Supply finite decimal returns above -100%.')
    trading=TradingComparison(inner_folds=4,validation_size=12,min_train_size=60,cap=cap,turnover_penalty=penalty_bps/10000,rebalance_alpha=alpha)
    trading.validate(120,5); base=TuningSettings(gamma=gamma)
    cal=trace_calibration(returns,base,[.001,.01,.05,.1,.25],cs=[.25,.5,1,2,4],window=120,calibration_start=calibration_start,evaluation_start=replay_start,periods_per_year=12,cost_bps=cost_bps,trading=trading)
    settings=TuningSettings(gamma=gamma,c=cal['selected_c'],epsilon=cal['selected_epsilon'])
    old=historical_backtest(returns,settings,window=120,oos_start=replay_start,cost_bps=cost_bps,trading=trading)
    rows=[]; previous={}
    for t in range(120,len(values)):
        if dates[t]<pd.Timestamp(replay_start): continue
        u,h,v=posterior_moments(values[t-120:t],draws=draws,seed=seed+t)
        for name,w in targets(u,h,v,trading,gamma,previous).items():
            prev=previous.get(name,np.full(5,.2)); turnover=np.abs(w-prev).sum()/2; gross=float(w@values[t]); net=gross-cost_bps/10000*turnover
            if net<=-1: raise ValueError('A strategy reaches nonpositive wealth.')
            previous[name]=w*(1+values[t])/(1+gross)
            rows.append({'Date':dates[t],'Method':name,'Gross return':gross,'Net return':net,'Turnover':turnover,**{f'Weight {a}':w[j] for j,a in enumerate(returns.columns)}})
    history=pd.concat([old,pd.DataFrame(rows)],ignore_index=True); history['Method']=history['Method'].replace({'Classical (main sample)':'Classical MV'})
    summary=[]
    for name,g in history.groupby('Method',sort=False):
        summary.append({'Method':name,**performance_metrics(g['Net return'].to_numpy(),gamma=gamma,periods_per_year=12),'Average turnover':g['Turnover'].mean()})
    allocations,_=latest_portfolios(returns,settings,old,window=120,trading=trading)
    u,h,v=posterior_moments(values[-120:],draws=draws,seed=seed+len(values)); allocations.update(targets(u,h,v,trading,gamma,previous))
    final=pd.DataFrame([{'Method':'Classical MV' if k=='Classical (main sample)' else k,**dict(zip(returns.columns,w))} for k,w in allocations.items()])
    return dict(history=history,summary=pd.DataFrame(summary),allocations=final,selected_c=settings.c,selected_epsilon=settings.epsilon,calibration_through=cal['calibration_through'])

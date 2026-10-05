from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import streamlit as st
from core.data import download_yahoo_monthly_returns,load_returns_csv
from core.diffusion_exposure import completed_yahoo_returns
from core.rpe import compare

st.title('RPE')
st.caption('Compare a frozen regularized posterior diffusion prototype with classical MV, trace tuning and old diffusion. All methods use the same monthly data, 120-observation window, constraints and transaction costs.')
st.info('Research prototype trained on independent simulated tasks. Supports exactly five assets and monthly returns. Its simulation results do not establish a live-market advantage.')

def remember(key,widget):
    st.session_state.setdefault('rpe_inputs',{})[key]=st.session_state[widget]

def field(kind,label,key,**kwargs):
    saved=st.session_state.setdefault('rpe_inputs',{}); widget='_rpe_'+key
    if widget not in st.session_state and key in saved: st.session_state[widget]=saved[key]
    value=getattr(st,kind)(label,key=widget,on_change=remember,args=(key,widget),**kwargs); saved[key]=value
    return value

@st.cache_data(ttl=3600,show_spinner=False)
def download(tickers):
    return completed_yahoo_returns(download_yahoo_monthly_returns(list(tickers),start='2000-01-01'),source='monthly').loc[:,list(tickers)].dropna()

@st.cache_data(show_spinner=False)
def run_comparison(data,settings):
    return compare(data,**settings)

shared=st.session_state.get('shared_current_window',{})
source=field('radio','Data source','source',options=['Portfolio saved data','Yahoo Finance','Upload CSV'])
data=None
if source=='Portfolio saved data':
    if isinstance(shared.get('full_returns'),pd.DataFrame) and shared.get('holding_period')=='1 month':
        data=shared['full_returns'].copy()
        st.caption('Using the Portfolio tab’s saved monthly history. Comparison settings below apply to every method.')
    else: st.info('Run Portfolio with five assets and a 1 month interval, or choose Yahoo Finance / Upload CSV.')
elif source=='Yahoo Finance':
    text=field('text_input','Five tickers','tickers',value='AAPL,MSFT,NVDA,GOOG,AMZN')
    tickers=tuple(dict.fromkeys(t.strip().upper() for t in text.split(',') if t.strip()))
    if st.button('Download / refresh RPE data'):
        try:
            if len(tickers)!=5: raise ValueError('Enter exactly five distinct tickers.')
            with st.spinner('Downloading complete monthly returns…'): st.session_state['rpe_download']=(tickers,download(tickers))
        except Exception as exc: st.error(str(exc))
    saved=st.session_state.get('rpe_download')
    if saved is not None and saved[0]==tickers: data=saved[1]
else:
    upload=st.file_uploader('CSV of monthly decimal returns (Date plus five asset columns)',type='csv')
    if upload is not None:
        try: st.session_state['rpe_upload']=load_returns_csv(upload)
        except Exception as exc: st.error(str(exc))
    data=st.session_state.get('rpe_upload')

if data is None: st.stop()
if data.shape[1]!=5: st.error('This trained prototype supports exactly five assets.'); st.stop()
st.caption(f'{len(data)} monthly rows; {data.index[0].date()} to {data.index[-1].date()}.')
today=pd.Timestamp(datetime.now(ZoneInfo('America/Havana')).date())
start=field('text_input','Backtest / strategy replay start date','start',value=str((today-pd.DateOffset(months=6)).date()))
calibration=field('text_input','Trace calibration starts','calibration',value=str(data.index[min(120,len(data)-1)].date()))
gamma=field('number_input','Risk aversion γ','gamma',min_value=.01,value=float(shared.get('gamma',3.)))
cap=field('number_input','Maximum weight per asset','cap',min_value=.2,max_value=1.,value=float(shared.get('max_long_weight',.4)))
cost=field('number_input','Trading cost (bps per turnover)','cost',min_value=0.,max_value=1000.,value=25.)
penalty=field('number_input','TC penalty (bps)','penalty',min_value=0.,max_value=1000.,value=25.)
alpha=field('number_input','TC rebalance step (%)','alpha',min_value=1.,max_value=100.,value=50.)
with st.expander('Prototype and comparison details'):
    st.write('The frozen model was trained on 16,000 five-asset Gaussian tasks, each containing 120 monthly returns. It uses a broad factor prior and 4,000 denoising updates. Training used no market backtest outcomes. Covariance draws remain positive definite through Cholesky decoding.')
    st.write('RPE uses posterior mean and average conditional covariance. RPE predictive also adds posterior covariance of the mean. Posterior draws never increase the historical information size. Every method pays trading costs; TC variants also use the penalty and partial rebalance. Classical MV stays unadjusted, as in Feasible Tuning.')
    st.write('Trace c and epsilon are selected jointly on data before the replay start and then frozen. Old diffusion retains its nested Best-T selection. All strategies start from equal weights. This prototype is not the PDF’s checkpoint or RPE–Jorion implementation.')
    draws=field('number_input','Posterior draws','draws',min_value=100,max_value=1000,value=500,step=100)
settings=dict(replay_start=start,calibration_start=calibration,gamma=gamma,cap=cap,cost_bps=cost,penalty_bps=penalty,alpha=alpha/100,draws=draws)
signature=(pd.util.hash_pandas_object(data,index=True).to_numpy().tobytes(),tuple(data.columns),tuple(settings.items()))
if st.button('Run RPE comparison',type='primary'):
    try:
        with st.spinner('Calibrating trace tuning and replaying all methods…'):
            result=run_comparison(data,settings)
        st.session_state['rpe_result']=(signature,result)
    except Exception as exc: st.error(f'Comparison failed: {exc}')
saved=st.session_state.get('rpe_result')
if saved is None: st.stop()
if saved[0]!=signature: st.info('Settings or data changed. Run the comparison to update results.'); st.stop()
result=saved[1]
st.success(f"Trace settings: c = {result['selected_c']:g}, ε = {result['selected_epsilon']:g}; calibration through {pd.Timestamp(result['calibration_through']).date()}.")
st.subheader('Final allocations')
st.dataframe(result['allocations'].style.format({a:'{:.2%}' for a in data.columns}),hide_index=True,width='stretch')
st.subheader('Historical results')
summary=result['summary']; percent=[c for c in summary.columns if c not in ['Method','Sharpe','Final $10,000']]
st.dataframe(summary.style.format({**{c:'{:.2%}' for c in percent},'Sharpe':'{:.3f}','Final $10,000':'${:,.2f}'}),hide_index=True,width='stretch')
wealth=result['history'].pivot(index='Date',columns='Method',values='Net return').add(1).cumprod(); st.line_chart(wealth)
st.download_button('Download historical comparison CSV',summary.to_csv(index=False),'rpe_comparison.csv','text/csv')

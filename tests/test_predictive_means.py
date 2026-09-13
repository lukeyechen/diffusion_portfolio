import numpy as np
import pandas as pd
import pytest

from core.predictive_means import regression_coefficients, regression_dataset, forecast_means
from core.short_horizon_portfolio import turnover_controlled_target, run_oos_comparison, replay_latest_recommendation


def returns(n=50):
    rng = np.random.default_rng(78)
    r = rng.normal(.001, .025, (n, 3))
    for i in range(1, n):
        r[i] += .3*r[i-1]
    return r


def test_exact_regression_matches_ols_and_terminal_moment_formula():
    rng = np.random.default_rng(23)
    x = rng.normal(size=(200, 3)) + 1
    y = .7 + x @ np.array([.3, -.4, .2]) + rng.normal(size=200)
    intercept, slope, _ = regression_coefficients(x, y, 0)
    expected = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
    np.testing.assert_allclose(np.r_[intercept, slope], expected, atol=1e-12)
    z = np.column_stack([x, y]); mu = z.mean(axis=0); A = np.cov(z, rowvar=False, bias=True)
    a = .13; C = (1-a)*np.eye(4)+a*A
    m = mu-a*A@np.linalg.solve(C, mu)
    S = A-a*a*np.linalg.matrix_power(A, 3)@np.linalg.matrix_power(np.linalg.inv(C), 2)
    b = np.linalg.solve(S[:3,:3], S[:3,3])
    intercept, slope, _ = regression_coefficients(x, y, a)
    np.testing.assert_allclose(slope, b, atol=1e-12)
    np.testing.assert_allclose(intercept, m[3]-m[:3]@b, atol=1e-12)


def test_pair_alignment_and_live_features():
    r = returns(); X, y, latest = regression_dataset(r)
    np.testing.assert_allclose(X[0,:,0], r[2]*100)
    np.testing.assert_allclose(y[0], r[3]*100)
    np.testing.assert_allclose(X[-1,:,0], r[-2]*100)
    np.testing.assert_allclose(y[-1], r[-1]*100)
    np.testing.assert_allclose(latest[:,0], r[-1]*100)
    np.testing.assert_allclose(latest[:,1], r[-3:].mean(axis=0)*100)


def test_validation_scores_are_manual_past_only_forecasts():
    r = returns(25); result = forecast_means(r)
    X, y, _ = regression_dataset(r); n=len(y)
    start=max(12, n-min(12,max(1,n//5)))
    for row in result['validation']:
        errors=[]
        for k in range(start,n):
            predictions=[]
            for j in range(3):
                intercept, slope, _ = regression_coefficients(X[:k,j],y[:k,j],row['b']/k)
                predictions.append((intercept+X[k,j]@slope)/100)
            errors.append(np.mean((np.array(predictions)-y[k]/100)**2))
        assert row['validation_mse'] == pytest.approx(np.mean(errors))
    assert result['a'] == result['b']/result['n_pairs']


def test_degenerate_data_fallback_and_invalid_input():
    r=np.full((20,3),.01); out=forecast_means(r)
    assert all('fallback' in s for s in out['status'])
    np.testing.assert_allclose(out['diffusion'], .01)
    with pytest.raises(ValueError): forecast_means(returns(10))
    with pytest.raises(ValueError): regression_coefficients(np.ones((20,3)),np.ones(20),1)
    r[0,0]=np.nan
    with pytest.raises(ValueError): forecast_means(r)


CFG=dict(lookback=22, inner_folds=2, validation_size=3, min_train_size=15, periods_per_year=52)
KW=dict(candidate_t=[0.0,.1], n_steps=10, max_long_weight=.6, rebalance_alpha=.5)


def test_mean_substitution_preserves_diffusion_covariance_and_caps():
    r=returns(22); prev=np.ones(3)/3
    original=turnover_controlled_target(r,prev,CFG,**KW)
    forecast=turnover_controlled_target(r,prev,CFG,mean_model='Diffusion forecast',**KW)
    np.testing.assert_array_equal(original['sigma'],forecast['sigma'])
    assert original['T']==forecast['T']
    np.testing.assert_array_equal(original['mu'],forecast['original_mu'])
    np.testing.assert_array_equal(forecast['mu'],forecast['forecast']['diffusion'])
    assert forecast['weights'].sum()==pytest.approx(1)
    assert np.all(forecast['weights']>=0) and np.max(forecast['weights'])<=.6+1e-9


def test_future_rows_cannot_change_earlier_forecasts_or_weights():
    df=pd.DataFrame(returns(26),index=pd.date_range('2020-01-01',periods=26,freq='W'),columns=list('ABC'))
    summary, detail, _, _=run_oos_comparison(df,CFG,include_forecasts=True,**KW)
    changed=df.copy();changed.iloc[-1]=[.4,-.3,.2]
    _, altered, _, _=run_oos_comparison(changed,CFG,include_forecasts=True,**KW)
    pd.testing.assert_frame_equal(detail.iloc[:-1],altered.iloc[:-1])
    cols=[c for c in detail if c.startswith('forecast_') and not c.startswith('forecast_mse')]
    pd.testing.assert_series_equal(detail[cols].iloc[-1],altered[cols].iloc[-1])
    assert len(summary)==8
    # Recommendation on a prefix is the portfolio actually held next in replay.
    rec=replay_latest_recommendation(df.iloc[:-1],CFG,mean_model='Diffusion forecast',**KW)
    name='Diffusion Forecast + Diffusion Risk'
    assert detail[f'return__{name}'].iloc[-1]==pytest.approx(rec['weights']@df.iloc[-1].to_numpy())


def test_utility_identity_and_mse_counterexample():
    mu=np.array([.03,.08]); C=np.array([[.04,.01],[.01,.09]]); S=np.array([[.06,.005],[.005,.07]])
    q0=np.array([.02,.09]);q1=np.array([.04,.05]);gamma=3
    H=np.linalg.inv(S); w0=H@q0/gamma;w1=H@q1/gamma;d=w1-w0
    U=lambda w: w@mu-gamma/2*w@C@w
    L=lambda q: (H@q-np.linalg.solve(C,mu))@C@(H@q-np.linalg.solve(C,mu))
    assert U(w1)-U(w0)==pytest.approx(d@(mu-gamma*C@w0)-gamma/2*d@C@d)
    assert U(w1)-U(w0)==pytest.approx((L(q0)-L(q1))/(2*gamma))
    C=np.diag([.01,1]); e0=np.array([0.,1]);e1=np.array([.2,0])
    assert e1@e1 < e0@e0
    assert e1@np.linalg.solve(C,e1) > e0@np.linalg.solve(C,e0)


def test_oos_rejects_reverse_chronology():
    df=pd.DataFrame(returns(26),index=pd.date_range('2020-01-01',periods=26,freq='W'))
    with pytest.raises(ValueError, match='oldest to newest'):
        run_oos_comparison(df.iloc[::-1],CFG,include_forecasts=True,**KW)


def test_drawdown_includes_initial_capital():
    from core.short_horizon_portfolio import performance_metrics
    result=performance_metrics(np.array([-.1,.05]),gamma=3,periods_per_year=52)
    assert result['Max drawdown']==pytest.approx(-.1)

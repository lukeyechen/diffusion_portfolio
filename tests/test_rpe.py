import numpy as np
import pandas as pd
import pytest
from core.rpe import posterior_moments, compare,RPE_NAMES


def test_posterior_is_reproducible_spd_and_preserves_global_rng():
    import torch
    x=np.random.default_rng(42).normal(.005,.06,(120,5))
    state=torch.random.get_rng_state().clone()
    a=posterior_moments(x,draws=100,seed=4)
    b=posterior_moments(x,draws=100,seed=4)
    assert torch.equal(state,torch.random.get_rng_state())
    # Inference must not change global random state.
    cached_state=torch.random.get_rng_state().clone()
    posterior_moments(x,draws=100,seed=4)
    assert torch.equal(cached_state,torch.random.get_rng_state())
    for first,second in zip(a,b): np.testing.assert_array_equal(first,second)
    assert np.linalg.eigvalsh(a[1]).min()>0
    assert np.linalg.eigvalsh(a[2]).min()>-1e-12


def test_rejects_wrong_history_shape():
    with pytest.raises(ValueError,match='120 monthly'): posterior_moments(np.ones((120,4)))


def test_comparison_has_paired_dates_costs_and_constrained_allocations():
    x=pd.DataFrame(np.random.default_rng(2).normal(.006,.05,(126,5)),columns=list('ABCDE'),index=pd.date_range('2010-01-31',periods=126,freq='ME'))
    result=compare(x,replay_start=str(x.index[124].date()),calibration_start=str(x.index[120].date()),draws=100)
    history=result['history']
    assert set(RPE_NAMES)<=set(history.Method)
    assert history.groupby('Method').size().eq(2).all()
    assert result['calibration_through']<x.index[124]
    np.testing.assert_allclose(history['Net return'],history['Gross return']-.0025*history.Turnover)
    assert np.isfinite(result['summary'].select_dtypes('number')).all().all()
    final=result['allocations'].drop(columns='Method').to_numpy()
    np.testing.assert_allclose(final.sum(1),1,atol=1e-6)
    assert final.min()>=-1e-8 and final.max()<=.400001
    # Each estimator receives only its trailing past block.
    expected=posterior_moments(x.iloc[4:124].to_numpy(),draws=100,seed=731+124)
    from core.rpe import targets
    from core.feasible_tuning import TradingComparison
    w=targets(*expected,TradingComparison(),3.,{})['RPE (no TC)']
    row=history[(history.Method=='RPE (no TC)')&(history.Date==x.index[124])].iloc[0]
    np.testing.assert_allclose(row[[f'Weight {a}' for a in x.columns]].to_numpy(float),w)

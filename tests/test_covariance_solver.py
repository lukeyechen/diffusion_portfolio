import numpy as np
import pytest
from core.covariance_solver import factor_covariance
from core.diffusion import generate_reverse_diffusion_samples, gaussian_score_diagnostics, forward_moments
from core.moments import sample_moments
from core.turnover_upgrade import exact_reverse_diffusion_moments


def test_well_conditioned_solve_matches_full_inverse_without_ridge():
    c=np.array([[2.,.2],[.2,1.]])
    solver=factor_covariance(c)
    assert solver.diagnostics['Ridge added']==0
    np.testing.assert_allclose(solver.solve(np.eye(2)),np.linalg.inv(c),rtol=1e-13)
    np.testing.assert_allclose(c@solver.solve(np.array([1.,2.])),[1.,2.])


@pytest.mark.parametrize('c',[np.ones((3,3)),np.diag([1e-14,1.]),np.zeros((2,2))])
def test_singular_and_ill_conditioned_covariances_are_stabilized(c):
    solver=factor_covariance(c,max_condition=1e5)
    assert solver.diagnostics['Ridge added']>0
    assert np.linalg.eigvalsh(solver.matrix).min()>0
    assert np.linalg.cond(solver.matrix)<=1e5*(1+1e-9)
    assert np.isfinite(solver.solve(np.eye(len(c)))).all()


@pytest.mark.parametrize('c,match',[(np.diag([-1.,2.]),'negative eigenvalue'),(np.array([[1.,.5],[0.,1.]]),'asymmetric'),(np.array([[np.nan]]),'NaN'),(np.array([[np.inf]]),'infinity')])
def test_invalid_matrices_are_rejected(c,match):
    with pytest.raises(ValueError,match=match): factor_covariance(c)


def test_small_roundoff_asymmetry_is_symmetrized():
    c=np.array([[1.,.2+1e-14],[.2,1.]])
    solver=factor_covariance(c)
    np.testing.assert_array_equal(solver.matrix,solver.matrix.T)


def test_duplicate_asset_integration_and_diagnostics():
    x=np.random.default_rng(1).normal(size=(120,1)); x=np.repeat(x,3,axis=1)
    mu,cov=exact_reverse_diffusion_moments(x,0.)
    assert np.linalg.eigvalsh(cov).min()>0
    assert np.isfinite(generate_reverse_diffusion_samples(x,20,.5,n_steps=20,seed=1)).all()
    rows=gaussian_score_diagnostics(*sample_moments(x),0.)
    assert rows[0]['Ridge added']>0
    assert rows[0]['Condition after']<=1e8*(1+1e-9)


def test_cholesky_paths_match_direct_inverse_reference():
    x=np.random.default_rng(3).normal(size=(120,3)); u,h=sample_moments(x)
    horizon=.4; beta=1.; steps=20; dt=horizon/steps; eye=np.eye(3)
    mean=np.zeros(3); covariance=(1-np.exp(-horizon))*eye
    rng=np.random.default_rng(77); paths=rng.normal(size=(25,3))*np.sqrt(1-np.exp(-horizon))
    for k in range(steps):
        mu,c=forward_moments(u,h,horizon-k*dt,beta); a=np.linalg.inv(c)
        f=eye-beta*(a-.5*eye)*dt
        mean=f@mean+beta*a@mu*dt
        covariance=f@covariance@f.T+beta*dt*eye
        paths=paths-beta*(paths@(a-.5*eye).T-a@mu)*dt+np.sqrt(beta*dt)*rng.normal(size=paths.shape)
    got_mean,got_cov=exact_reverse_diffusion_moments(x,horizon,n_steps=steps)
    np.testing.assert_allclose(got_mean,mean,rtol=1e-12,atol=1e-12)
    # Output covariance keeps the app's existing separate portfolio-input ridge.
    np.testing.assert_allclose(got_cov,covariance+1e-8*max(np.trace(covariance)/3,1)*eye,rtol=1e-12)
    got_paths=generate_reverse_diffusion_samples(x,25,horizon,n_steps=steps,seed=77)
    np.testing.assert_allclose(got_paths,paths,rtol=1e-12,atol=1e-12)

"""Validated SPD covariance factorization; no silent pseudoinverse fallback."""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve


@dataclass
class CovarianceFactor:
    matrix: np.ndarray
    factor: tuple
    diagnostics: dict

    def solve(self, rhs):
        value=cho_solve(self.factor,np.asarray(rhs,float),check_finite=True)
        if not np.isfinite(value).all():
            raise ValueError('Gaussian covariance solve produced non-finite values.')
        return value


def factor_covariance(covariance, *, eigenvalue_floor=1e-8, max_condition=1e8):
    """Apply the minimum ridge meeting a scaled eigenvalue floor and condition bound.

    Reject materially indefinite/asymmetric inputs rather than disguising an
    invalid model as a numerical problem. The floor is scaled by max(mean
    diagonal, 1), preserving the previous app's absolute ridge scale.
    """
    c=np.asarray(covariance,float)
    if c.ndim!=2 or c.shape[0]!=c.shape[1] or not c.size:
        raise ValueError('Gaussian covariance must be a nonempty square matrix.')
    if not np.isfinite(c).all(): raise ValueError('Gaussian covariance contains NaN or infinity; check return data.')
    if not np.isfinite(eigenvalue_floor) or eigenvalue_floor<=0 or not np.isfinite(max_condition) or max_condition<=1:
        raise ValueError('Require a positive eigenvalue floor and condition limit greater than one.')
    norm=max(float(np.linalg.norm(c,ord=np.inf)),np.finfo(float).tiny)
    if np.max(np.abs(c-c.T))>1e-10*norm:
        raise ValueError('Gaussian covariance is materially asymmetric; check its construction.')
    c=(c+c.T)/2
    eigenvalues=np.linalg.eigvalsh(c); low,high=float(eigenvalues[0]),float(eigenvalues[-1])
    tolerance=100*np.finfo(float).eps*max(abs(low),abs(high),np.finfo(float).tiny)
    if low < -tolerance:
        raise ValueError('Gaussian covariance has a materially negative eigenvalue; check the covariance model before regularizing.')
    floor=eigenvalue_floor*max(float(np.trace(c)/len(c)),1.)
    ridge=max(0.,floor-low,(high-max_condition*low)/(max_condition-1))
    if ridge: ridge=float(np.nextafter(ridge,np.inf))
    regularized=c+ridge*np.eye(len(c))
    try: factor=cho_factor(regularized,lower=True,check_finite=True)
    except np.linalg.LinAlgError as exc:
        raise ValueError('Cholesky factorization failed after covariance regularization; check duplicate assets, units and sample size.') from exc
    diagnostics={'Minimum eigenvalue before':low,'Condition before':float(high/low) if low>0 else np.inf,
                 'Ridge added':ridge,'Eigenvalue floor':floor,
                 'Condition after':float((high+ridge)/(low+ridge))}
    return CovarianceFactor(regularized,factor,diagnostics)

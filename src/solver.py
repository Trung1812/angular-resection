"""
ODR and OLS solvers for 2-D angular resection (bearing intersection)
=======================================================================

This Python module offers 2 estimation backends for the same
angular-resection problem:

Ordinary Least Squares (OLS) - assumes anchors are error-free and
  minimises squared bearing residuals only.  Suitable when anchor survey error
  is negligible versus compass noise.
Orthogonal / Total Least Squares (ODR) - accounts for both compass
  and anchor covariance with optional Huber robustness.

Usage
-----
```python
from odr_resection_solver import solve_resection_ols, solve_resection_odr

x_hat_ols = solve_resection_ols(theta, anchors, sigma_theta)
res = solve_resection_odr(theta, anchors, sigma_theta, Sigma)
print(res["position"], "±", np.sqrt(np.diag(res["cov"])))
```
"""
from __future__ import annotations

import numpy as np
from numpy.linalg import lstsq, norm, inv
from scipy.stats import chi2

__all__ = [
    "solve_resection_ols",
    "solve_resection_odr",
    "confidence_ellipse"
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _prepare_initial_guess(theta: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Linear LS intersection ignoring noise – good enough to start GN or IRLS."""
    nvec = np.column_stack((-np.sin(theta), np.cos(theta)))  # m×2 normals
    b = np.einsum("ij,ij->i", nvec, anchors)
    x0, *_ = lstsq(nvec, b, rcond=None)
    return x0  # (2,)

def _huber_weights(u: np.ndarray, delta: float) -> np.ndarray:
    """Return the weight factor ϕ(u)/u used in IRLS for the Huber loss."""
    absu = np.abs(u)
    w = np.where(absu <= delta, 1.0, delta / absu)
    return w

def _wrap_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angle to (‑π, π]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

# -----------------------------------------------------------------------------
# 1) OLS solver – bearing residuals only (anchors treated as exact)
# -----------------------------------------------------------------------------

def solve_resection_ols(
    theta: np.ndarray,
    anchors: np.ndarray,
    sigma_theta: float | np.ndarray = 1.0,
    *,
    init_guess: np.ndarray | None = None,
    max_iter: int = 50,
    tol: float = 1e-5,
):
    """Estimate position X by ordinary least squares on bearing residuals.

    Parameters
    ----------
    theta : (m,) array_like
        Measured bearings (rad) clockwise from north.
    anchors : (m, 2) array_like
        Anchor coordinates (x_i, y_i) in **same CRS**.
    sigma_theta : float or (m,) array_like, default 1.0
        1-sigma noise for each bearing (rad). Scalar is broadcast.
    init_guess : (2,) array_like, optional
        Starting point.  Default: quick LS intersection.
    max_iter : int, default 50
        Gauss-Newton iterations.
    tol : float, default 1e-5 (m)
        Convergence threshold on |ΔX|.

    Returns
    -------
    dict with keys
        position  : (2,) ndarray - Estimated (x̂, ŷ)
        cov       : (2,2) ndarray - Covariance ≈ (Jᵀ W J)⁻¹ · σ²
        residuals : (m,) ndarray - Final bearing residuals (rad)
        iterations: int  - Number of iterations executed
        converged : bool - True if |ΔX| < tol
    """
    theta = np.asarray(theta, dtype=float)
    anchors = np.asarray(anchors, dtype=float)
    m = theta.size
    if anchors.shape != (m, 2):
        raise ValueError("anchors must be (m,2) array")

    # Noise – diagonal weights
    if np.isscalar(sigma_theta):
        sigma_theta = np.full(m, sigma_theta, dtype=float)
    else:
        sigma_theta = np.asarray(sigma_theta, dtype=float)
        if sigma_theta.shape != (m,):
            raise ValueError("sigma_theta must be scalar or length m")
    W = np.diag(1.0 / sigma_theta**2)  # constant throughout iterations

    # Initial position
    if init_guess is None:
        X = _prepare_initial_guess(theta, anchors)
    else:
        X = np.asarray(init_guess, dtype=float)
        if X.shape != (2,):
            raise ValueError("init_guess must be length‑2 array")

    for k in range(max_iter):
        # Predicted bearings (model)
        dx = anchors[:, 0] - X[0]
        dy = anchors[:, 1] - X[1]
        g = np.arctan2(dy, dx)  # (m,)

        # Residuals wrapped to (‑π, π]
        v = _wrap_pi(theta - g)  # (m,)

        # Jacobian J (m×2)
        r2 = dx**2 + dy**2
        J = np.column_stack((dy / r2, -dx / r2))

        # Normal equations: (Jᵀ W J) Δ = Jᵀ W v
        JW = J.T @ W  # 2×m
        H = JW @ J    # 2×2
        gvec = JW @ v  # 2×
        try:
            delta = np.linalg.solve(H, gvec)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Normal matrix singular - check anchor geometry") from exc

        X_new = X + delta
        if norm(delta) < tol:
            X = X_new
            converged = True
            break
        X = X_new
    else:
        converged = False
        k = max_iter

    # Final residuals and covariance
    dx = anchors[:, 0] - X[0]
    dy = anchors[:, 1] - X[1]
    g = np.arctan2(dy, dx)
    v = _wrap_pi(theta - g)
    J = np.column_stack((dy / (dx**2 + dy**2), -dx / (dx**2 + dy**2)))
    cov = inv(J.T @ W @ J)

    return {
        "position": X,
        "cov": cov,
        "residuals": v,
        "iterations": k + 1,
        "converged": converged,
    }

# -----------------------------------------------------------------------------
# 2) ODR solver – dynamic weights + optional Huber
# -----------------------------------------------------------------------------

def solve_resection_odr(
    theta: np.ndarray,
    anchors: np.ndarray,
    sigma_theta: float | np.ndarray,
    Sigma: np.ndarray,
    *,
    init_guess: np.ndarray | None = None,
    huber_delta: float = 1.5,
    robust: bool = True,
    max_iter: int = 50,
    tol: float = 1e-5,
):
    """Estimate the unknown position X via Weighted ODR / TLS.

    Parameters
    ----------
    theta : (m,) array_like
        Measured bearings in **radians** clockwise from north.
    anchors : (m, 2) array_like
        Anchor coordinates (x_i, y_i) in the same projected CRS.
    sigma_theta : float or (m,) array_like
        bearing noise (rad).  A scalar is broadcast.
    Sigma : (m, 2, 2) array_like
        Covariance matrices \Sigma_i (m anchors).
    init_guess : (2,) array_like, optional
        Starting point (x, y).  Default uses linear LS intersection.
    huber_delta : float, default 1.5
        Huber clipping threshold (in \sigma units).
    robust : bool, default True
        If *False* use pure quadratic loss.
    max_iter : int, default 50
        Maximum IRLS iterations.
    tol : float, default 1e-5 (metres)
        Convergence threshold.

    Returns
    -------
    dict with keys
        position : (2,) ndarray - Estimated (x̂, ŷ)
        cov      : (2, 2) ndarray - Covariance of the estimate (≈ (J^T W J)^{-1})
        residuals: (m,) ndarray - Final orthogonal distances r_i
        iterations : int       - Number of iterations executed
        converged  : bool      - True if ||ΔX|| < tol
    """
    theta = np.asarray(theta, dtype=float)
    anchors = np.asarray(anchors, dtype=float)
    m = theta.size

    # Bearing noise theta sigma
    if np.isscalar(sigma_theta):
        sigma_theta = np.full(m, sigma_theta, dtype=float)
    else:
        sigma_theta = np.asarray(sigma_theta, dtype=float)
    assert sigma_theta.shape == (m,)

    # Anchor covariances
    Sigma = np.asarray(Sigma, dtype=float)
    assert Sigma.shape == (m, 2, 2)

    # Bearing unit normals n_i = [-sin theta, cos theta]
    nvec = np.stack((-np.sin(theta), np.cos(theta)), axis=1)  # m×2

    # Initial guess
    if init_guess is None:
        X = _prepare_initial_guess(theta, anchors)
    else:
        X = np.asarray(init_guess, dtype=float)
        assert X.shape == (2,)

    J_const = nvec  # Jacobian of orthogonal distance w.r.t X (constant)

    for k in range(max_iter):
        diff = X - anchors            # m×2
        D = norm(diff, axis=1)        # ranges
        proj_var = np.einsum("ij,ijk,ik->i", nvec, Sigma, nvec)  # nᵀΣn
        sigma_r2 = D**2 * sigma_theta**2 + proj_var
        sigma_r = np.sqrt(sigma_r2)

        # Residuals r_i = n_i * (X − A_i)
        r = nvec @ X - np.einsum("ij,ij->i", nvec, anchors)

        # Robust or quadratic weights
        if robust:
            u = r / sigma_r
            h = _huber_weights(u, huber_delta)
            w = h / sigma_r2              
        else:
            w = 1.0 / sigma_r2

        JW = J_const * w[:, None]        # weight each row of J
        H = J_const.T @ JW               
        g = J_const.T @ (w * r)          

        try:
            delta = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Normal matrix singular - poor anchor geometry") from exc

        X_new = X + delta
        if norm(delta) < tol:
            X = X_new
            converged = True
            break
        X = X_new
    else:
        converged = False
        k = max_iter

    # Final residuals & covariance
    diff = X - anchors
    D = norm(diff, axis=1)
    sigma_r2 = D**2 * sigma_theta**2 + np.einsum("ij,ijk,ik->i", nvec, Sigma, nvec)
    r = nvec @ X - np.einsum("ij,ij->i", nvec, anchors)
    jw = J_const * (1 / sigma_r2)[:, None]
    cov = inv(J_const.T @ jw)

    return {
        "position": X,
        "cov": cov,
        "residuals": r,
        "iterations": k + 1,
        "converged": converged,
    }

def confidence_ellipse(cov, alpha=0.95):
    """
    Return semi-axis lengths (a, b) and azimuth angle φ (rad, from x-axis)
    for the 2-D confidence ellipse defined by cov.
    """
    # eigen-decomposition
    vals, vecs = np.linalg.eigh(cov)          # lambda1 <= lambda2
    order = np.argsort(vals)[::-1]            # largest first
    lambda1, lambda2 = vals[order]
    v1 = vecs[:, order[0]]                    # eigen-vector for lambda1 (major axis)

    chi_sq = chi2.ppf(alpha, df=2)             
    a = np.sqrt(chi_sq * lambda1)             # semi-major
    b = np.sqrt(chi_sq * lambda2)             # semi-minor
    phi = np.arctan2(v1[1], v1[0])            # rotation (rad)

    return a, b, phi

# -----------------------------------------------------------------------------
# Self‑test block
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    true_X = np.array([10.0, 8.0])
    anchors = np.array([[0, 0], [25, 0], [25, 25], [0, 25]], dtype=float)
    m = anchors.shape[0]

    # Synthetic bearings with σθ = 0.5°
    sigma_theta = 0.5 * np.pi / 180
    theta_true = np.arctan2(anchors[:, 1] - true_X[1], anchors[:, 0] - true_X[0])
    theta_meas = theta_true + np.random.normal(0, sigma_theta, m)

    # OLS test ------------------------------------------------------------
    res_ols = solve_resection_ols(theta_meas, anchors, sigma_theta)
    print("OLS →", res_ols["position"].round(3), "converged:", res_ols["converged"])

    # ODR test (1 cm anchor cov) -----------------------------------------
    Sigma = np.repeat(np.eye(2)[None, :, :] * 0.01**2, m, axis=0)
    res_odr = solve_resection_odr(theta_meas, anchors, sigma_theta, Sigma)
    print("ODR →", res_odr["position"].round(3), "converged:", res_odr["converged"])

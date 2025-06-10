# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
from numpy.linalg import lstsq, norm, inv

__all__ = [
    "solve_resection_odr",
]

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _prepare_initial_guess(theta: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Linear least-squares intersection (ignores noise) for the start value."""
    nvec = np.column_stack((-np.sin(theta), np.cos(theta)))  # m×2
    b = np.einsum("ij,ij->i", nvec, anchors)
    x0, *_ = lstsq(nvec, b, rcond=None)
    return x0  # shape (2,)

def _huber_weights(u: np.ndarray, delta: float) -> np.ndarray:
    """Huber weight factor ϕ(u)/u for IRLS (|u|≤δ ⇒ 1;  else δ/|u|)."""
    absu = np.abs(u)
    w = np.where(absu <= delta, 1.0, delta / absu)
    return w

# -----------------------------------------------------------------------------
# Public API
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

    # Bearing noise σ_θ
    if np.isscalar(sigma_theta):
        sigma_theta = np.full(m, sigma_theta, dtype=float)
    else:
        sigma_theta = np.asarray(sigma_theta, dtype=float)
    assert sigma_theta.shape == (m,)

    # Anchor covariances Σ_i
    Sigma = np.asarray(Sigma, dtype=float)
    assert Sigma.shape == (m, 2, 2)

    # Bearing unit normals n_i
    nvec = np.stack((-np.sin(theta), np.cos(theta)), axis=1)  # m×2

    # Initial guess
    if init_guess is None:
        X = _prepare_initial_guess(theta, anchors)
    else:
        X = np.asarray(init_guess, dtype=float)
        assert X.shape == (2,)

    J = nvec  # Jacobian = n_i components, constant w.r.t X

    for k in range(max_iter):
        diff = X - anchors            # m×2
        D = norm(diff, axis=1)        # ranges
        proj_var = np.einsum("ij,ijk,ik->i", nvec, Sigma, nvec)  # nᵀΣn
        sigma_r2 = D**2 * sigma_theta**2 + proj_var
        sigma_r = np.sqrt(sigma_r2)

        # Residuals r_i = n_i·(X − A_i)
        r = nvec @ X - np.einsum("ij,ij->i", nvec, anchors)

        # Robust or quadratic weights
        if robust:
            u = r / sigma_r
            h = _huber_weights(u, huber_delta)
            w = h / sigma_r2              # w_i = ϕ(u)/σ_r²
        else:
            w = 1.0 / sigma_r2

        JW = J * w[:, None]              # weight each row of J
        H = J.T @ JW                     # Normal matrix (2×2)
        g = J.T @ (w * r)                # Gradient

        try:
            delta = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Normal matrix singular - poor anchor geometry?") from exc

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
    jw = J * (1 / sigma_r2)[:, None]
    cov = inv(J.T @ jw)

    return {
        "position": X,
        "cov": cov,
        "residuals": r,
        "iterations": k + 1,
        "converged": converged,
    }

if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # Quick self‑test with synthetic data
    # ---------------------------------------------------------------------
    np.random.seed(0)
    # True position
    true_X = np.array([10.0, 8.0])
    # Anchors at corners of a square (0,0), (20,0), (20,20), (0,20)
    anchors = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=float)
    m = anchors.shape[0]
    # Bearing measurements with 0.5° (0.0087 rad) noise
    theta_true = np.arctan2(anchors[:, 1] - true_X[1], anchors[:, 0] - true_X[0])
    sigma_theta = 0.5 * np.pi / 180
    theta_meas = theta_true + np.random.normal(0, sigma_theta, m)
    # Anchor covariance 1 cm isotropic
    Sigma = np.repeat(np.eye(2)[None, :, :] * 0.01**2, m, axis=0)

    result = solve_resection_odr(theta_meas, anchors, sigma_theta, Sigma)
    print("\n--- Synthetic test ---")
    print("True X:", true_X)
    print("Estimated X:", result["position"].round(4))
    print("Iterations:", result["iterations"], " Converged:", result["converged"])

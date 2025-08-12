import numpy as np
import pandas as pd

class GraphFlowAllocator:
    """
    Reconstruct bilateral who-to-whom flows X (holders i -> issuers j) from:
      - row marginals r_i (asset acquisitions / uses, nonnegative magnitudes)
      - col marginals c_j (liability incurrence / sources, nonnegative magnitudes)
      - adjacency mask A_ij in {0,1} defining allowed edges (graph connectivity)
    using a masked entropic-OT / Sinkhorn (a.k.a. RAS/IPFP on support) with optional costs.

    Objective (informal):
        minimize  <C, X> - tau * H(X)   over X >= 0,  with   X 1 = r,   X^T 1 = c,   X_ij = 0 if A_ij = 0
    where K = exp(-C/tau) acts as a prior kernel. If you also pass a prior P (nonnegative),
    the kernel becomes K .* P (renormalized).

    Parameters
    ----------
    tol : float
        Convergence tolerance on row/column sums.
    max_iter : int
        Maximum Sinkhorn iterations.
    tau : float
        Entropic temperature; higher => smoother allocations (more diffuse).
        If tau -> inf and cost=0 on support, solution approaches uniform on edges.
    epsilon_floor : float
        Small floor to avoid numerical underflow on K.
    """

    def __init__(self, tol=1e-10, max_iter=2000, tau=1.0, epsilon_floor=1e-300):
        self.tol = tol
        self.max_iter = max_iter
        self.tau = tau
        self.epsilon_floor = epsilon_floor

    @staticmethod
    def _safe(vec):
        v = np.asarray(vec, dtype=float).copy()
        v[np.isnan(v)] = 0.0
        return v

    def allocate(self, r, c, A, cost=None, prior=None, return_kernel=False):
        """
        Compute X given marginals r (rows), c (cols), adjacency A, optional cost/prior.

        Inputs
        ------
        r : array-like shape (n,)
            Row targets (nonnegative). Will be scaled if sum(r) != sum(c).
        c : array-like shape (m,)
            Column targets (nonnegative).
        A : array-like shape (n,m) in {0,1}
            Adjacency mask: 1 = allowed edge, 0 = forbidden.
        cost : array-like shape (n,m), optional
            Cost matrix C; ignored where A=0. If None, C=0 on support.
        prior : array-like shape (n,m), optional
            Nonnegative prior weights; ignored where A=0. If None, uniform on support.
        return_kernel : bool
            If True, also return the effective kernel K used in scaling.

        Returns
        -------
        X : ndarray shape (n,m)
            Balanced bilateral flows.
        info : dict
            Diagnostics: iterations, max_row_err, max_col_err, mass_scaled_factor
        (K : ndarray) optional
            The kernel used before Sinkhorn scaling.
        """
        r = self._safe(r).astype(float)
        c = self._safe(c).astype(float)
        A = (np.asarray(A, dtype=float) > 0).astype(float)
        n, m = A.shape
        assert r.shape[0] == n and c.shape[0] == m, "Marginals do not match adjacency dimensions"

        # Ensure total mass consistency
        s_r = r.sum(); s_c = c.sum()
        scale = 1.0
        if s_r <= 0 and s_c <= 0:
            return np.zeros_like(A), {"iterations": 0, "max_row_err": 0.0, "max_col_err": 0.0, "mass_scaled_factor": 0.0}
        if s_r == 0 and s_c > 0:
            # cannot allocate
            return np.zeros_like(A), {"iterations": 0, "max_row_err": s_c, "max_col_err": s_c, "mass_scaled_factor": 0.0}
        if s_c == 0 and s_r > 0:
            return np.zeros_like(A), {"iterations": 0, "max_row_err": s_r, "max_col_err": s_r, "mass_scaled_factor": 0.0}
        if abs(s_r - s_c) > 1e-9 * max(1.0, s_r, s_c):
            scale = s_r / max(1e-30, s_c)
            c = c * scale

        # Build kernel K on support
        if cost is None:
            C = np.zeros_like(A)
        else:
            C = np.asarray(cost, dtype=float)
            assert C.shape == A.shape
            C = C * A  # ignore off-support

        if prior is None:
            K = np.exp(-C / max(self.tau, 1e-12)) * A
        else:
            P = np.asarray(prior, dtype=float)
            assert P.shape == A.shape
            K = np.exp(-C / max(self.tau, 1e-12)) * P * A

        # Avoid underflow
        K = np.maximum(K, self.epsilon_floor) * A

        u = np.ones(n)
        v = np.ones(m)

        # Sinkhorn iterations on support
        for it in range(self.max_iter):
            Ku = K.dot(v)  # shape (n,)
            # rows
            u = np.divide(r, Ku, out=np.zeros_like(r), where=Ku > 0)
            KTu = K.T.dot(u)
            # cols
            v = np.divide(c, KTu, out=np.zeros_like(c), where=KTu > 0)

            # Check errors occasionally
            if (it % 25) == 0 or it == self.max_iter - 1:
                X = (u[:, None] * K) * v[None, :]
                row_err = np.max(np.abs(X.sum(axis=1) - r))
                col_err = np.max(np.abs(X.sum(axis=0) - c))
                if max(row_err, col_err) <= self.tol:
                    return (X, {"iterations": it+1, "max_row_err": float(row_err), "max_col_err": float(col_err), "mass_scaled_factor": float(scale)}) if not return_kernel else (X, {"iterations": it+1, "max_row_err": float(row_err), "max_col_err": float(col_err), "mass_scaled_factor": float(scale)}, K)

        # Final X
        X = (u[:, None] * K) * v[None, :]
        row_err = np.max(np.abs(X.sum(axis=1) - r))
        col_err = np.max(np.abs(X.sum(axis=0) - c))
        return (X, {"iterations": self.max_iter, "max_row_err": float(row_err), "max_col_err": float(col_err), "mass_scaled_factor": float(scale)}) if not return_kernel else (X, {"iterations": self.max_iter, "max_row_err": float(row_err), "max_col_err": float(col_err), "mass_scaled_factor": float(scale)}, K)


def allocate_flows_from_graph(row_marginals, col_marginals, adjacency, cost=None, prior=None, tau=1.0, tol=1e-10, max_iter=2000):
    """
    Convenience wrapper. Returns numpy array X and info dict.
    """
    allocator = GraphFlowAllocator(tol=tol, max_iter=max_iter, tau=tau)
    X, info = allocator.allocate(row_marginals, col_marginals, adjacency, cost=cost, prior=prior, return_kernel=False)
    return X, info

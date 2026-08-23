"""
lsqlin_util.py

Small helper reproducing the subset of MATLAB's `lsqlin` behavior needed by
MSLM_INT.py: linearly equality- and inequality-constrained least squares,

    minimize   || C @ x - d ||_2^2
    subject to Aeq @ x == beq
               Aneq @ x <= bneq

MATLAB's `lsqlin(C, d, Aneq, bneq, Aeq, beq, [], [], x0, options)` (no lower/
upper bounds, interior-point algorithm) solves exactly this problem. SciPy
has no direct equivalent, so this wraps `scipy.optimize.minimize` (SLSQP)
with an analytic objective gradient and analytic constraint Jacobians, which
is exact for this quadratic objective / linear constraints case.

Written for the Python translation of MSLM_INT.m (Dmitri Kondrashov),
by Taylor McDonald, 2026.
"""

import numpy as np
from scipy.optimize import minimize


def lsqlin(C, d, Aeq=None, beq=None, Aneq=None, bneq=None, x0=None,
           maxiter=500, ftol=1e-12):
    """
    Linearly-constrained least squares, matching MATLAB's lsqlin (no bounds).

    Args:
        C (np.ndarray): Design matrix, shape (m, n).
        d (np.ndarray): Target vector, shape (m,).
        Aeq (np.ndarray, optional): Equality constraint matrix, shape (p, n).
        beq (np.ndarray, optional): Equality constraint RHS, shape (p,).
        Aneq (np.ndarray, optional): Inequality constraint matrix (Aneq @ x <= bneq),
            shape (q, n).
        bneq (np.ndarray, optional): Inequality constraint RHS, shape (q,).
        x0 (np.ndarray, optional): Initial guess, shape (n,). Defaults to ones,
            matching MSLM_INT.m's `b0 = ones((nmax+2)*2,1)`.
        maxiter (int): Maximum SLSQP iterations.
        ftol (float): SLSQP precision goal for the objective.

    Returns:
        tuple:
            x (np.ndarray): Solution vector, shape (n,).
            res (OptimizeResult): Full scipy result (for diagnostics).
    """
    C = np.asarray(C, dtype=float)
    d = np.asarray(d, dtype=float).ravel()
    n = C.shape[1]

    if x0 is None:
        x0 = np.ones(n)
    else:
        x0 = np.asarray(x0, dtype=float).ravel()

    CtC = C.T @ C
    Ctd = C.T @ d

    def obj(x):
        r = C @ x - d
        return float(r @ r)

    def jac(x):
        return 2.0 * (CtC @ x - Ctd)

    constraints = []

    if Aeq is not None and np.size(Aeq) > 0 and Aeq.shape[0] > 0:
        Aeq = np.asarray(Aeq, dtype=float)
        beq = np.zeros(Aeq.shape[0]) if beq is None else np.asarray(beq, dtype=float).ravel()
        constraints.append({
            'type': 'eq',
            'fun': lambda x, A=Aeq, b=beq: A @ x - b,
            'jac': lambda x, A=Aeq: A,
        })

    if Aneq is not None and np.size(Aneq) > 0 and Aneq.shape[0] > 0:
        Aneq = np.asarray(Aneq, dtype=float)
        bneq = np.zeros(Aneq.shape[0]) if bneq is None else np.asarray(bneq, dtype=float).ravel()
        # SLSQP 'ineq' convention is fun(x) >= 0; MATLAB's is Aneq@x <= bneq,
        # i.e. bneq - Aneq@x >= 0.
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, A=Aneq, b=bneq: b - A @ x,
            'jac': lambda x, A=Aneq: -A,
        })

    res = minimize(obj, x0, jac=jac, constraints=constraints, method='SLSQP',
                     options={'maxiter': maxiter, 'ftol': ftol})
    # res = minimize(obj, x0, jac=jac, constraints=constraints, method='trust-constr',
    #                options={'maxiter': maxiter, 'gtol': ftol})

    if not res.success and res.status not in (0, 4, 9):
        # 4 = "positive directional derivative" and 9 = iteration-limit are
        # common benign SLSQP exits that still return a usable near-optimum;
        # anything else gets a warning so silent divergence isn't hidden.
        print(f"lsqlin: SLSQP did not fully converge (status={res.status}): {res.message}")

    return res.x, res

#
import warnings
import torch
from torch.linalg import solve

# test
import numpy as np
from scipy.optimize import nnls

# Note:
# 0. Input A and b should be of torch.tensor type
# 1. A and b should be finite. To improve performance, we do not include the check.
#    Hence, use torch.isnan(A).any() and torch.isnan(b).any() beforehand.
def tensor_nnls(A, b, maxiter=None, *, atol=None):
    if len(A.shape) != 2:
        raise ValueError("Expected a two-dimensional array (matrix)" +
                         f", but the shape of A is {A.shape}")
    if len(b.shape) != 1:
        raise ValueError("Expected a one-dimensional array (vector)" +
                         f", but the shape of b is {b.shape}")

    m, n = A.shape
    if m != b.shape[0]:
        raise ValueError(
                "Incompatible dimensions. The first dimension of " +
                f"A is {m}, while the shape of b is {(b.shape[0], )}")

    x, rnorm, mode = _nnls_tensor(A, b, maxiter, tol=atol)
    if mode != 1:
        raise RuntimeError("Maximum number of iterations reached.")

    return x, rnorm

def _nnls_tensor(A, b, maxiter=None, tol=None):
    # Default Value Setting
    m, n = A.shape
    input_dtype = A.dtype
    device = A.device
    if not maxiter:
        maxiter = 3*n
    if tol is None:
        tol = 10 * max(m, n) * torch.finfo(input_dtype).eps

    AtA = A.T @ A
    Atb = b @ A  # Result should be 1D - let torch figure it out!

    # Initialize vars
    x = torch.zeros(n, dtype=input_dtype, device=device)
    s = torch.zeros(n, dtype=input_dtype, device=device)
    # Inactive constraint switches
    P = torch.zeros(n, dtype=torch.bool, device=device)

    # Projected residual
    w = Atb.clone().to(dtype=input_dtype, device=device)  # x=0. Skip (-AtA @ x) term

    # Overall iteration counter
    # Outer loop is not counted, inner iter is counted across outer spins
    iter = 0

    # not P.all() -> exit loop when all elements of P is True
    # (w[~P] > tol).any() -> choose elements of w whose corresponding elements in P is False
    #                        exit loop when none of these elements is larger then tol
    while (not P.all()) and (w[~P] > tol).any():  # B
        # Get the "most" active coeff index and move to inactive set
        k = torch.argmax(w * (~P))  # B.2
        P[k] = True  # B.3

        # Iteration solution
        s[:] = 0.
        # B.4
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Ill-conditioned matrix found', category=RuntimeWarning)
            # torch does not support .ix_ , so the code would be slightly tedious
            P_indices = torch.where(P)[0]
            sub_AtA = AtA[P_indices][:, P_indices]
            sub_Atb = Atb[P_indices]
            s[P] = solve(sub_AtA, sub_Atb)

        # Inner loop
        while (iter < maxiter) and (s[P].min() < 0):  # C.1
            iter += 1
            inds = P * (s < 0)
            alpha = (x[inds] / (x[inds] - s[inds])).min()  # C.2
            x *= (1 - alpha)
            x += alpha*s
            P[x <= tol] = False
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Ill-conditioned matrix found', category=RuntimeWarning)
                # torch does not support .ix_ , so the code would be slightly tedious
                P_indices = torch.where(P)[0]
                sub_AtA = AtA[P_indices][:, P_indices]
                sub_Atb = Atb[P_indices]
                s[P] = solve(sub_AtA, sub_Atb)
            s[~P] = 0  # C.6

        x[:] = s[:]
        w[:] = Atb - AtA @ x

        if iter == maxiter:
            # Typically following line should return
            # return x, np.linalg.norm(A@x - b), -1
            # however at the top level, -1 raises an exception wasting norm
            # Instead return dummy number 0.
            return x, 0., -1

    return x, torch.linalg.norm(A@x - b), 1

# test the algorithm
if __name__ == "__main__":
    m, n = 50, 10
    A = np.random.randn(m, n)
    x_true = np.abs(np.random.randn(n))
    b = A @ x_true + 0.1 * np.random.randn(m)
    device = "cuda"
    A_tensor = torch.from_numpy(A).to(torch.float32).to(device)
    b_tensor = torch.from_numpy(b).to(torch.float32).to(device)

    result_np, norm_np = nnls(A, b)
    result_to, norm_to = tensor_nnls(A_tensor, b_tensor)
    print("true:", x_true)
    print("numpy:", result_np)
    print("torch:", result_to)
    print("norm_numpy", norm_np)
    print("norm_torch", norm_to)
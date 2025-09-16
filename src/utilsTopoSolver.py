import torch
import numpy as np
import pandas as pd
import numpy.linalg as la
import scipy.linalg as sla

import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

from src.data_generation import get_omp_coeff


def indicator_matrix(row):
    tmp = row.sigma.copy()
    tmp[row.idx] = 0
    return np.diag(tmp)


def indicator_matrix_rev(row):
    tmp = row.sigma.copy()
    tmp[row.idx] = 1
    return np.diag(tmp)


def compute_Luj(row, b2, J):
    Lu = b2 @ row.sigma @ b2.T
    Luj = np.array([la.matrix_power(Lu, i) for i in range(1, J + 1)])
    return Luj


def split_coeffs(h, s, k, sep=False):
    h_tmp = h.value.flatten()
    if sep:
        if s != 1 and k != 1:
            hH = h_tmp[np.arange(0, (s * (2 * k + 1)), (2 * k + 1))].reshape((s, 1))
            hS = h_tmp[
                np.hstack(
                    [[i, i + 1] for i in range(1, (s * (2 * k + 1)), (2 * k + 1))]
                )
            ].reshape((s, k))
            hI = h_tmp[
                np.hstack(
                    [[i, i + 1] for i in range((k + 1), (s * (2 * k + 1)), (2 * k + 1))]
                )
            ].reshape((s, k))
        else:
            hH = h_tmp[np.arange(0, (s * (2 * k)), (2 * k))]
            hS = h_tmp[
                np.hstack([[i, i + 1] for i in range(1, (s * (2 * k)), (2 * k))])
            ]
            hI = h_tmp[
                np.hstack([[i, i + 1] for i in range((k), (s * (2 * k)), (2 * k))])
            ]
        return [hH, hS, hI]
    if s != 1 and k != 1:
        hi = h_tmp[np.arange(0, (s * (k + 1)), (k + 1))].reshape((s, 1))
        h = h_tmp[
            np.hstack([[i, i + 1] for i in range(1, (s * (k + 1)), (k + 1))])
        ].reshape((s, k))
    else:
        hi = h_tmp[np.arange(0, (s * k), k)]
        h = h_tmp[np.hstack([[i, i + 1] for i in range(1, (s * k), k)])]
    return np.hstack([h, hi])


def rearrange_coeffs(h, J, P, sep=False):
    h_tmp = h.value.flatten()
    if sep:
        base_idxs = np.arange(0, (P * (2 * J + 1)), (2 * J + 1))
        hH = h_tmp[base_idxs].reshape((P, 1))
        hS = h_tmp[
            np.hstack([np.arange((i + 1), (i + 1 + J)) for i in base_idxs])
        ].reshape((P, J))
        hI = h_tmp[
            np.hstack([np.arange((i + 1 + J), (i + 1 + 2 * J)) for i in base_idxs])
        ].reshape((P, J))
        return [hH, hS, hI]
    base_idxs = np.arange(0, (P * (J + 1)), (J + 1))
    hi = h_tmp[base_idxs].reshape((P, 1))
    h = h_tmp[np.hstack([np.arange((i + 1), (i + 1 + J)) for i in base_idxs])].reshape(
        (P, J)
    )
    return np.hstack([h, hi])


def sparse_transform(D, K0, Y_te, Y_tr=None, fit_intercept=True):
    ep = np.finfo(float).eps  # to avoid some underflow problems
    dd = la.norm(D, axis=0) + ep
    W = np.diag(1.0 / dd)
    Domp = D @ W
    X_te = np.apply_along_axis(
        lambda x: get_omp_coeff(
            K0, Domp=Domp.real, col=x.real, fit_intercept=fit_intercept
        ),
        axis=0,
        arr=Y_te,
    )
    # Normalization
    X_te = W @ X_te

    if np.all(Y_tr == None):
        return X_te

    # Same for the training set
    X_tr = np.apply_along_axis(
        lambda x: get_omp_coeff(
            K0, Domp=Domp.real, col=x.real, fit_intercept=fit_intercept
        ),
        axis=0,
        arr=Y_tr,
    )
    X_tr = W @ X_tr

    return X_te, X_tr


def compute_vandermonde(L, k):
    def polynomial_exp(x, k):
        x = x ** np.arange(0, k + 1)
        return x

    eigenvalues, _ = sla.eig(L)
    idx = eigenvalues.argsort()
    tmp_df = pd.DataFrame({"Eigs": eigenvalues[idx]})
    tmp_df["Poly"] = tmp_df["Eigs"].apply(lambda x: polynomial_exp(x, k))
    B = np.vstack(tmp_df["Poly"].to_numpy())

    return B


def nmse(D, X, Y, m):
    return (1 / m) * np.sum(la.norm(Y - (D @ X), axis=0) ** 2 / la.norm(Y, axis=0) ** 2)


def nmsept(D, X, Y, m):
    """
    Compute the normalized mean squared error (NMSE) in PyTorch.

    Args:
    D (torch.Tensor): Dictionary tensor.
    X (torch.Tensor): Coefficients tensor.
    Y (torch.Tensor): Original signal tensor.
    m (int): Number of columns (data points).

    Returns:
    torch.Tensor: The NMSE value.
    """

    reconstruction_error = torch.norm(Y - torch.matmul(D, X), dim=0) ** 2
    original_norm = torch.norm(Y, dim=0) ** 2
    return (1 / m) * torch.sum(reconstruction_error / original_norm)


def partial_subder(hu, B2, Lu, k):
    ek = np.eye(1, B2.shape[1], k)
    Ek = ek.T @ ek
    LL = [la.matrix_power(Lu, i) for i in range(0, hu.shape[0])]
    der = np.zeros((Lu.shape))
    for j in range(1, hu.shape[0] + 1):
        for l in range(j):
            # print(f"LL[l] : {LL[l].shape}")
            # print(f"B2 : {B2.shape}")
            # tmp1 = LL[l] @ B2 @ Ek
            # print(f"tmp1 : {tmp1.shape}")
            # tmp2 = tmp1 @ B2.T
            # print(f"tmp2 : {tmp2.shape}")
            # tmp3 = tmp2 @ LL[j - l - 1]
            # print(f"tmp3 : {tmp3.shape}")
            # print(f"huj: {hu[0].shape}, {hu[j-1]}")
            # der += hu[j - 1] * tmp3
            der += hu[j - 1] * (LL[l] @ B2 @ Ek) @ B2.T @ LL[j - l - 1]

    return der


def partial_der(hu, B2, Lu, k):
    der = np.concatenate(
        [partial_subder(hu[i, :], B2, Lu, k) for i in range(hu.shape[0])]
    )
    print(f"Der: {der.shape}")
    return der


def compute_derivative(D, Y, X, hu, B2, Lu, k):
    print(f"First: {(Y - D @ X).T.shape}")
    par_der = partial_der(hu, B2, Lu, k)
    term1 = -2 * np.trace(Y.T @ par_der @ X)
    term2 = +2 * np.trace(X.T @ D @ par_der @ X)
    return term1 + term2


def compute_grad(D, Y, X, hu, B2, Lu):
    grad = np.concatenate(
        compute_derivative(D, Y, X, hu, B2, Lu, k) for k in range(B2.shape[1])
    )
    print(f"Grad: {grad.shape}")
    return grad


def generate_dictionarypt(h, P, *matrices):
    """
    Generate a dictionary matrix as a concatenation of sub-dictionary matrices. Each of the sub-dictionary
    matrices is generated from given coefficients and a Laplacian matrix (or matrices if discriminating between
    upper and lower Laplacian).

    Parameters:
    - h (torch.Tensor): Coefficients for linear combination.
    - P (int): Number of kernels (number of sub-dictionaries).
    - matrices (torch.Tensor): Laplacian matrices.

    Returns:
    - torch.Tensor: Generated dictionary matrix.
    """

    D = []
    # Check if upper and lower Laplacians are separately provided
    if len(matrices) == 1:
        Lj = matrices[0]
        M = Lj.shape[-1]
        J = Lj.shape[0]

        for i in range(P):
            h_tmp = h[i, :-1].reshape(J, 1, 1)
            tmp = torch.sum(h_tmp * Lj, dim=0) + h[i, -1] * torch.eye(
                M, device=Lj.device
            )
            D.append(tmp)

    elif len(matrices) == 2:
        Luj, Ldj = matrices
        M = Luj.shape[-1]
        J = Luj.shape[0]

        for i in range(P):
            hu = h[1][i].reshape(J, 1, 1)
            hd = h[2][i].reshape(J, 1, 1)
            hid = h[0][i]
            tmp = torch.sum(hu * Luj + hd * Ldj, dim=0) + hid * torch.eye(
                M, device=Luj.device
            )
            D.append(tmp)

    else:
        raise ValueError("Function accepts one vector and either 1 or 2 matrices.")

    D = torch.hstack(D)
    return D


def proximal_op(z, lambda_, mode="soft2"):
    k = lambda_
    if mode == "hard":
        np.clip(z, 0, 1, out=z)
    elif mode == "soft":
        for i in range(len(z)):
            if z[i] <= k:
                z[i] = 0
            elif z[i] > k and z[i] < (1 + k):
                z[i] -= k
            else:
                z[i] = 1
    elif mode == "soft2":
        for i in range(len(z)):
            if z[i] <= k:
                z[i] = 0
            elif z[i] > k and z[i] < 1:
                pass
            else:
                z[i] = 1
    elif mode == "soft3":
        l = 0.95
        for i in range(len(z)):
            if z[i] <= k:
                z[i] = 0
            elif z[i] > k and z[i] < l:
                pass
            else:
                z[i] = 1
    else:
        raise ValueError("Invalid mode. Choose from 'hard', 'soft', or 'soft2")
    return z

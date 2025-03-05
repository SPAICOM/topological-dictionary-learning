"This file contains the functions for generating synthetic data for dictionary learning experiments."

import numpy as np
import numpy.linalg as la
import pandas as pd
from scipy.sparse.linalg import eigs
from sklearn.linear_model import OrthogonalMatchingPursuit
from typing import Tuple, Union
from tqdm import tqdm

if __name__ == "__main__":
    from utils import (
        memoize_or_save,
    )
else:
    from topolearn.utils import (
        memoize_or_save,
    )


def compute_Lj_and_lambdaj(
    L: np.ndarray, J: int, separated: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute powers of the matrix L up to J and the maximum and minimum eigenvalues raised to the powers of 1 through J.

    Args:
        L : np.ndarray:
            The cell complex Laplacian matrix.
        J : int:
            The highest power to compute, i.e., the order of the Cell FIR filter
        separated (bool, optional): If True, compute separated eigenvalue ranges. Defaults to False.

    Returns:
        Lj : np.ndarray
            The horizontal concatenation of all the matrix powers of L from 1 to J.
        lambda_max_k : np.ndarray
            The vector of maximum eigenvectors for each power of L.
        lambda_min_k : np.ndarray
            The vector of minimum eigenvectors for each power of L.
    """
    try:
        lambdas, _ = eigs(L)
    except:
        L = np.array(L, dtype=float)
        lambdas, _ = eigs(L)
    lambdas[np.abs(lambdas) < np.finfo(float).eps] = 0
    lambda_max = np.max(lambdas).real
    lambda_min = np.min(lambdas).real
    Lj = np.array([la.matrix_power(L, i) for i in range(1, J + 1)])

    if separated:
        lambda_max_k = lambda_max ** np.arange(1, J + 1)
        lambda_min_k = lambda_min ** np.arange(1, J + 1)
    else:
        lambda_max_k = lambda_max ** np.array(list(np.arange(1, J + 1)) + [0])
        lambda_min_k = lambda_min ** np.array(list(np.arange(1, J + 1)) + [0])

    return Lj, lambda_max_k, lambda_min_k


def generate_coeffs(
    *arrays: np.ndarray, P: int, mult: int = 10
) -> Tuple[np.ndarray, float, float, float, float]:
    """Generate coefficients for synthetic data generation.

    Args:
        arrays : np.ndarray
          Variable number of arrays specifying eigenvalues.
        P : int
          Number of sub-dictionaries.
        mult : int
          Multiplier for coefficient generation. Defaults to 10.

    Returns:
        h : Union[list, np.ndarray
            The generated dictionary coefficients (a list of coefficients in the case of "separated" parameterization).
        c : float
            Dictionary frequency control parameter for constraints (b) and (c) in the paper.
        epsilon : float
            Dictionary frequency control parameter for constraint (c) in the paper.
        tmp_sum_min, tmp_sum_max : float
            Control parameters for dictionary initialization (used at inference time and not for generation).
    """

    # if passing four arguments (two for upper and two for lower laplacian eigevals)
    # it means that you are using dictionary_type="separated"
    if len(arrays) == 2:
        lambda_max_k, lambda_min_k = arrays
        J = lambda_max_k.shape[0]
        h = mult / np.max(lambda_max_k) * np.random.rand(P, J)
        tmp_max_vec = h @ lambda_max_k
        tmp_min_vec = h @ lambda_min_k
        c = np.max(tmp_max_vec)
        tmp_sum_max = np.sum(tmp_max_vec)
        tmp_sum_min = np.sum(tmp_min_vec)

        Delta_min = c - tmp_sum_min
        Delta_max = tmp_sum_max - c
        epsilon = (Delta_max - Delta_min) * np.random.rand() + Delta_min

    elif len(arrays) == 4:
        lambda_max_u_k, lambda_min_u_k, lambda_max_d_k, lambda_min_d_k = arrays
        J = lambda_max_u_k.shape[0]
        hI = mult / np.max(lambda_max_d_k) * np.random.rand(P, J)
        hS = mult / np.max(lambda_max_u_k) * np.random.rand(P, J)
        hH = (
            mult
            / np.min([np.max(lambda_max_u_k), np.max(lambda_max_d_k)])
            * np.random.rand(P, 1)
        )
        h = [hH, hS, hI]
        tmp_max_vec_S = (hS @ lambda_max_u_k).reshape(P, 1)
        tmp_min_vec_S = (hS @ lambda_min_u_k).reshape(P, 1)
        tmp_max_vec_I = (hI @ lambda_max_d_k).reshape(P, 1)
        tmp_min_vec_I = (hI @ lambda_min_d_k).reshape(P, 1)
        c = np.max(tmp_max_vec_I + tmp_max_vec_S + hH)
        tmp_sum_min = np.sum(tmp_min_vec_I + tmp_min_vec_S + hH)
        tmp_sum_max = np.sum(tmp_max_vec_I + tmp_max_vec_S + hH)
        Delta_min = c - tmp_sum_min
        Delta_max = tmp_sum_max - c
        epsilon = np.max([Delta_min, Delta_max])
    else:
        raise ValueError(
            "Function accepts either 2 or 4 arrays! In case of 4 arrays are provided,\
                        the first 2 refer to upper laplacian and the other two to lower laplacian."
        )

    return h, c, epsilon, tmp_sum_min, tmp_sum_max


def generate_dictionary(
    h: Union[list, np.ndarray], P: int, *matrices: np.ndarray
) -> np.ndarray:
    """Generate a dictionary matrix as a concatenation of sub-dictionary matrices. Each of the sub-dictionary
    matrices is generated from given coefficients and a Laplacian matrix (or matrices if discriminating between
    upper and lower Laplacian).

    Args:
        h : Union[list, np.ndarray]
            dictionary coefficients (a list of coefficients in the case of "separated" parameterization).
        P : int
            Number of kernels, i.e., number of sub-dictionaries.
        matrices : np.ndarray:
            Powers of Laplacian matrices.

    Returns:
        D : np.ndarray
            Generated dictionary matrix.
    """

    D = []
    # Check if upper and lower Laplacians are separately provided
    if len(matrices) == 1:
        Lj = matrices[0]
        M = Lj.shape[-1]
        J = Lj.shape[0]

        for i in range(0, P):
            h_tmp = h[i, :-1].reshape(J, 1, 1)
            tmp = np.sum(h_tmp * Lj, axis=0) + h[i, -1] * np.eye(M, M)
            D.append(tmp)
    elif len(matrices) == 2:
        Luj, Ldj = matrices
        M = Luj.shape[-1]
        J = Luj.shape[0]

        for i in range(0, P):
            hu = h[1][i].reshape(J, 1, 1)
            hd = h[2][i].reshape(J, 1, 1)
            hid = h[0][i]
            tmp = np.sum(hu * Luj + hd * Ldj, axis=0) + hid * np.eye(M, M)
            D.append(tmp)
    else:
        raise ValueError("Function accepts one vector and either 1 or 2 matrices.")
    D = np.hstack(tuple(D))
    return D


def create_ground_truth(
    Lu: np.ndarray,
    Ld: np.ndarray,
    m_train: int,
    m_test: int,
    P: int,
    J: int,
    K0: int,
    dictionary_type: str,
    sparsity_mode: str,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray
]:
    """Create ground truth data for testing dictionary learning algorithms.

    Args:
        Lu : np.ndarray
            Upper Laplacian matrix.
        Ld : np.ndarray
            Lower Laplacian matrix.
        m_train : int
            Number of training samples.
        m_test : int
            Number of testing samples.
        P : int
            Number of kernels (sub-dictionaries).
        J : int
            Maximum power of Laplacian matrices.
        K0 : int
            Maximum number of non-zero coefficients.
        dictionary_type : str
            Type of dictionary.
        sparsity_mode : str
            Mode of sparsity.

    Returns:
        D : np.ndarray
            Generated dictionary
        h : Union[list, np.ndarray
            The generated dictionary coefficients (a list of coefficients in the case of "separated" parameterization).
        train_Y : np.ndarray
            Compressible training topological synthetic signals.
        test_Y : np.ndarray
            Compressible test topological synthetic signals.
        epsilon : float
            Dictionary frequency control parameter for constraint (c) in the paper.
        c : float
            Dictionary frequency control parameter for constraints (b) and (c) in the paper.
        X_train : np.ndarray
            Ground-truth synthetic sparse representation of training data.
        X_test : np.ndarray
            Ground-truth synthetic sparse representation of test data.
    """

    if dictionary_type == "joint":
        Lj, lambda_max_k, lambda_min_k = compute_Lj_and_lambdaj(Lu + Ld, J)
        h, c, epsilon, _, _ = generate_coeffs(lambda_max_k, lambda_min_k, P=P)
        D = generate_dictionary(h, P, Lj)

    elif dictionary_type == "edge":
        Lj, lambda_max_k, lambda_min_k = compute_Lj_and_lambdaj(Ld, J)
        h, c, epsilon, _, _ = generate_coeffs(lambda_max_k, lambda_min_k, P=P)
        D = generate_dictionary(h, P, Lj)

    elif dictionary_type == "separated":
        Luj, lambda_max_u_k, lambda_min_u_k = compute_Lj_and_lambdaj(
            Lu, J, separated=True
        )
        Ldj, lambda_max_d_k, lambda_min_d_k = compute_Lj_and_lambdaj(
            Ld, J, separated=True
        )
        h, c, epsilon, _, _ = generate_coeffs(
            lambda_max_u_k, lambda_min_u_k, lambda_max_d_k, lambda_min_d_k, P=P
        )
        D = generate_dictionary(h, P, Luj, Ldj)

    M = D.shape[0]

    def create_column_vec(row, M, P):
        tmp = np.zeros(M * P)
        tmp[row["idxs"]] = row["non_zero_coeff"]
        return tmp

    m_total = m_train + m_test
    tmp = pd.DataFrame()

    if sparsity_mode == "random":
        tmp_K0 = np.random.choice(np.arange(1, K0 + 1), size=(m_total), replace=True)
    elif sparsity_mode == "max":
        tmp_K0 = np.full((m_total,), K0)
    # sparsity coefficient for each column
    tmp["K0"] = tmp_K0
    # for each column get K0 indexes
    tmp["idxs"] = tmp.K0.apply(lambda x: np.random.choice(M * P, x, replace=False))
    # for each of the K0 row indexes in each column, sample K0 values
    tmp["non_zero_coeff"] = tmp.K0.apply(lambda x: np.random.randn(x))
    # create the column vectors with the desired characteristics
    tmp["column_vec"] = tmp.apply(lambda x: create_column_vec(x, M=M, P=P), axis=1)
    # finally derive the sparse signal representation matrix
    X = np.column_stack(tmp["column_vec"].values)

    all_data = D @ X
    X_train = X[:, :m_train]
    X_test = X[:, m_train:]
    train_Y = all_data[:, :m_train]
    test_Y = all_data[:, m_train:]

    return D, h, train_Y, test_Y, epsilon, c, X_train, X_test


def get_omp_coeff(
    K0: int, Domp: np.ndarray, col: np.ndarray, fit_intercept: bool = True
) -> np.ndarray:
    """Compute the coefficients using Orthogonal Matching Pursuit.

    Args:
        K0 : int:
            Number of non-zero coefficients.
        Domp : np.ndarray:
            Dictionary (normalized) matrix.
        col : np.ndarray
            Target column vector.

    Returns:
        np.ndarray
            OMP resulting coefficients.
    """

    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=K0, fit_intercept=fit_intercept)

    omp.fit(Domp, col)
    return omp.coef_


def verify_dic(
    D: np.ndarray,
    Y_train: np.ndarray,
    X_train_true: np.ndarray,
    K0_max: int,
    acc_thresh: float,
) -> Tuple[int, float]:
    """Verify dictionary using Orthogonal Matching Pursuit by evaluating the sparse approximation for several levels of sparsity

    Args:
        D : np.ndarray
            Dictionary matrix.
        Y_train : np.ndarray
            Training data.
        X_train_true : np.ndarray
            True sparse representation of training data.
        K0_max : int
            Maximum number of non-zero coefficients.
        acc_thresh : float
            Accuracy threshold.

    Returns:
        max_possible_sparsity : int
            Maximum possible sparsity to achieve a certain accuracy.
        final_accuracy : float
            Final achieved accuracy.
    """

    dd = la.norm(D, axis=0)
    W = np.diag(1.0 / dd)
    Domp = D @ W
    for K0 in range(1, K0_max + 1):
        idx = np.sum(np.abs(X_train_true) > 0, axis=0) == K0
        try:
            tmp_train = Y_train[:, idx]
            X_true_tmp = X_train_true[:, idx]
            idx_group = np.abs(X_true_tmp) > 0
            X_tr = np.apply_along_axis(
                lambda x: get_omp_coeff(K0, Domp.real, x), axis=0, arr=tmp_train
            )
            idx_train = np.abs(X_tr) > 0
            acc = (
                np.sum(np.sum(idx_group == idx_train, axis=0) == idx_group.shape[0])
                / idx_group.shape[1]
            )
            if acc < acc_thresh:
                fin_acc = acc
                break
            else:
                fin_acc = acc
        except:
            fin_acc = 0
    max_possible_sparsity = K0 - 1
    return max_possible_sparsity, fin_acc


@memoize_or_save
def generate_data(Lu, Ld, **kwargs):

    params = {
        "dictionary_type": "separated",
        "M": 1,
        "P": 1,
        "J": 1,
        "n_sim": 1,
        "m_test": 10,
        "m_train": 10,
        "K0_max": 20,
        "n_search": 10,
        "prob_T": 1.0,
        "sparsity_mode": "random",
        "verbose": False,
    }

    params.update(kwargs)

    dictionary_type = params["dictionary_type"]
    M = params["M"]
    P = params["P"]
    J = params["J"]
    n_sim = params["n_sim"]
    m_test = params["m_test"]
    m_train = params["m_train"]
    K0_max = params["K0_max"]
    n_search = params["n_search"]
    prob_T = params["prob_T"]
    verbose = params["verbose"]
    sparsity_mode = params["sparsity_mode"]

    assert (
        prob_T <= 1 or prob_T >= 0
    ), "You must provide a 'prob_T' input between 0 and 1."

    D_true = np.zeros((M, M * P, n_sim))
    Y_train = np.zeros((M, m_train, n_sim))
    Y_test = np.zeros((M, m_test, n_sim))
    epsilon_true = np.zeros(n_sim)
    c_true = np.zeros(n_sim)
    X_train = np.zeros((M * P, m_train, n_sim))
    X_test = np.zeros((M * P, m_test, n_sim))

    for sim in tqdm(range(n_sim)):
        best_sparsity = 0

        if sparsity_mode == "random":
            for _ in tqdm(range(n_search), leave=False):
                try:
                    (
                        D_try,
                        h,
                        Y_train_try,
                        Y_test_try,
                        epsilon_try,
                        c_try,
                        X_train_try,
                        X_test_try,
                    ) = create_ground_truth(
                        Lu=Lu,
                        Ld=Ld,
                        m_train=m_train,
                        m_test=m_test,
                        P=P,
                        J=J,
                        K0=K0_max,
                        dictionary_type=dictionary_type,
                        sparsity_mode=sparsity_mode,
                    )

                    max_possible_sparsity, _ = verify_dic(
                        D_try, Y_train_try, X_train_try, K0_max, 0.8
                    )

                    if max_possible_sparsity > best_sparsity:
                        best_sparsity = max_possible_sparsity
                        D_true[:, :, sim] = D_try
                        Y_train[:, :, sim] = Y_train_try
                        Y_test[:, :, sim] = Y_test_try
                        epsilon_true[sim] = epsilon_try
                        c_true[sim] = c_try
                        X_train[:, :, sim] = X_train_try
                        X_test[:, :, sim] = X_test_try

                except Exception as e:
                    print(f"Error during dictionary creation: {e}")

            if verbose:
                print(f"...Done! # Best Sparsity: {best_sparsity}")

        else:
            (
                D_true[:, :, sim],
                _,
                Y_train[:, :, sim],
                Y_test[:, :, sim],
                epsilon_true[sim],
                c_true[sim],
                X_train[:, :, sim],
                X_test[:, :, sim],
            ) = create_ground_truth(
                Lu=Lu,
                Ld=Ld,
                m_train=m_train,
                m_test=m_test,
                P=P,
                J=J,
                K0=K0_max,
                dictionary_type=dictionary_type,
                sparsity_mode=sparsity_mode,
            )

    return D_true, Y_train, Y_test, X_train, X_test, epsilon_true, c_true

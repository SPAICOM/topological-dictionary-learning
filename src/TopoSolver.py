import os
import time
import torch
import pickle
import psutil
import numpy as np
import cvxpy as cp
import pandas as pd
import numpy.linalg as la
import scipy.linalg as sla
from functools import wraps
from typing import Tuple, List
from cvxpy.error import SolverError
from einops import rearrange, reduce

import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

from src.Hodgelets import (
    SeparateHodgelet,
    SimplicianSlepians,
    log_wavelet_kernels_gen,
)
from src.utilsHodgelets import (
    cluster_on_neigh,
    get_frequency_mask,
)
from src.utilsTopoSolver import (
    nmse,
    nmsept,
    compute_Luj,
    proximal_op,
    get_omp_coeff,
    rearrange_coeffs,
    sparse_transform,
    indicator_matrix,
    compute_vandermonde,
    indicator_matrix_rev,
    generate_dictionarypt,
)
from src.data_generation import (
    generate_coeffs,
    generate_dictionary,
    compute_Lj_and_lambdaj,
)
from src.EnhancedGraph import EnhancedGraph


class TopoSolver:
    def __init__(self, X_train, X_test, Y_train, Y_test, *args, **kwargs):
        params = {
            "P": None,  # Number of Kernels (Sub-dictionaries)
            "J": None,  # Polynomial order
            "K0": None,  # Sparsity level
            "dictionary_type": None,
            "c": None,  # spectral control parameter
            "epsilon": None,  # spectral control parameter
            "n": 10,  # number of nodes
            "sub_size": None,  # Number of sub-sampled nodes
            "true_prob_T": 1.0,  # True ratio of colored triangles
            "prob_T": 1.0,  # The triangle probability with which we want to bias our topology
            "p_edges": 1.0,  # Probability of edge existence
            "G_true": None,
            "seed": None,  ####
            "option": "One-shot-diffusion",
            "diff_order_sol": 1,
            "diff_order_irr": 1,
            "step_prog": 1,
            "top_k_slepians": 10,
            "B1_true": None,
            "B2_true": None,
        }

        self.testing_trace = {}  ##################################################################

        if args:
            if len(args) != 1 or not isinstance(args[0], dict):
                raise ValueError(
                    "When using positional arguments, must provide a single dictionary"
                )
            params.update(args[0])

        params.update(kwargs)

        # Data
        self.X_train: np.ndarray = X_train
        self.X_test: np.ndarray = X_test
        self.Y_train: np.ndarray = Y_train
        self.Y_test: np.ndarray = Y_test
        self.m_train: int = Y_train.shape[1]
        self.m_test: int = Y_test.shape[1]

        # Topology and geometry behind data (by default we consider a topology with full upper laplacian)
        if params["G_true"] == None:
            self.G = EnhancedGraph(
                n=params["n"],
                p_edges=params["p_edges"],
                p_triangles=params["prob_T"],
                seed=params["seed"],
            )
        # If we know the true topology we completely use it
        else:
            self.G = params["G_true"]

        if np.all(params["B1_true"] is None):
            # Incidence matrices
            self.B1: np.ndarray = self.G.get_b1()
            self.B2: np.ndarray = self.G.get_b2()

            # Sub-sampling if needed to decrease complexity
            if params["sub_size"] != None:
                self.B1 = self.B1[:, : params["sub_size"]]
                self.B2 = self.B2[: params["sub_size"], :]
                self.B2 = self.B2[:, np.sum(np.abs(self.B2), 0) == 3]
            # Laplacians according to the Hodge Theory for cell complexes
            Lu, Ld, L = self.G.get_laplacians(sub_size=params["sub_size"])
            self.Lu: np.ndarray = Lu  # Upper Laplacian
            self.Ld: np.ndarray = Ld  # Lower Laplacian
            self.L: np.ndarray = L  # Sum Laplacian
        else:
            self.B1 = params["B1_true"]
            self.B2 = params["B2_true"]
            self.Ld = (self.B1.T) @ self.B1
            self.Lu = self.B2 @ self.B2.T
            self.L = self.Lu + self.Ld

        # Topology dimensions and hyperparameters
        self.nu: int = self.B2.shape[1]
        self.nd: int = self.B1.shape[1]
        self.true_prob_T = params["true_prob_T"]
        self.T: int = int(np.ceil(self.nu * (1 - self.true_prob_T)))
        self.p = np.ones(self.nu)
        self.M = self.L.shape[0]
        self.dictionary_type = params["dictionary_type"]

        # Init the learning errors and error curve (history)
        self.min_error_train = 1e20
        self.min_error_test = 1e20
        self.global_min = 1e20
        self.train_history: List[np.ndarray] = []
        self.test_history: List[np.ndarray] = []
        self.train_error_hist: List[np.float64] = []
        self.test_error_hist: List[np.float64] = []
        self.opt_upper = 0

        # Dictionary hyperparameters
        self.P = params["P"]  # Number of sub-dicts
        self.J = params["J"]  # Polynomial order for the Hodge Laplacian

        # Assumed sparsity level
        self.K0 = params["K0"]
        self.q_star = int(np.ceil(np.ceil(0.05 * self.nu) + (self.K0 / 5 - 1)))
        # Init optimal values for sparse representations and overcomplete dictionary
        self.D_opt: np.ndarray = np.zeros((self.M, self.M * self.P))
        self.X_opt_train: np.ndarray = np.zeros(
            (self.M * self.P, self.Y_train.shape[1])
        )
        self.X_opt_test: np.ndarray = np.zeros((self.M * self.P, self.Y_test.shape[1]))

        ############################################################################################################
        ##                                                                                                        ##
        ##               This section is only for learnable (data-driven) dictionaries                            ##
        ##                                                                                                        ##
        ############################################################################################################

        # Initialize the optimal values of the dictionary coefficients
        self.zero_out_h()

        # Compute the polynomial extension for the Laplacians and the auxiliary
        # "pseudo-vandermonde" matrix for the constraints in the quadratic form
        if self.dictionary_type == "joint":
            self.Lj, self.lambda_max_j, self.lambda_min_j = compute_Lj_and_lambdaj(
                self.L, self.J
            )
            self.B = compute_vandermonde(self.L, self.J).real
        elif self.dictionary_type == "edge":
            self.Lj, self.lambda_max_j, self.lambda_min_j = compute_Lj_and_lambdaj(
                self.Ld, self.J
            )
            self.B = compute_vandermonde(self.Ld, self.J).real
        elif self.dictionary_type == "separated":
            self.Luj, self.lambda_max_u_j, self.lambda_min_u_j = compute_Lj_and_lambdaj(
                self.Lu, self.J, separated=True
            )
            self.Ldj, self.lambda_max_d_j, self.lambda_min_d_j = compute_Lj_and_lambdaj(
                self.Ld, self.J, separated=True
            )
            self.Bu = compute_vandermonde(self.Lu, self.J).real
            self.Bd = compute_vandermonde(self.Ld, self.J)[:, 1:].real
            self.B = np.hstack([self.Bu, self.Bd])

        # Auxiliary matrix to define quadratic form dor the dictionary learning step
        self.P_aux: np.ndarray = None
        # Flag variable: the dictionary is learnable or analytic
        self.dict_is_learnable = self.dictionary_type in [
            "separated",
            "joint",
            "edge",
        ]

        # Auxiliary tools for the Slepians-based dictionary setup
        if self.dictionary_type == "slepians":
            self.option = params["option"]
            self.diff_order_sol = params["diff_order_sol"]
            self.step_prog = params["step_prog"]
            self.diff_order_irr = params["diff_order_irr"]
            self.source_sol = np.ones((self.nd,))
            self.source_irr = np.ones((self.nd,))
            self.top_K_slepians = params["top_k_slepians"]
            self.spars_level = list(range(10, 80, 10))
            self.F_sol, self.F_irr = get_frequency_mask(
                self.B1, self.B2
            )  # Get frequency bands
            self.S_neigh, self.complete_coverage = cluster_on_neigh(
                self.B1,
                self.B2,
                self.diff_order_sol,
                self.diff_order_irr,
                self.source_sol,
                self.source_irr,
                self.option,
                self.step_prog,
            )
            self.R = [self.F_sol, self.F_irr]
            self.S = self.S_neigh

        # Auxiliary tools for the Wavelet-based dictionary setup
        elif self.dictionary_type == "wavelet":
            # Remember that this part should be updated if B2 or Lu are updated!
            self.w1 = np.linalg.eigvalsh(self.Lu)
            self.w2 = np.linalg.eigvalsh(self.Ld)

        if self.dict_is_learnable:
            # Hyperparameters for dictionary stability in frequency domain
            if params["c"] != None:
                self.c = params["c"]
                self.epsilon = params["epsilon"]
            else:
                self.spectral_control_params()

    def update_Lu(self, Lu_new, load=True, cascade=True):
        if load:
            self.Lu = Lu_new
            self.Luj, self.lambda_max_u_j, self.lambda_min_u_j = compute_Lj_and_lambdaj(
                self.Lu, self.J, separated=True
            )
            self.Bu = compute_vandermonde(self.Lu, self.J).real
            self.B = np.hstack([self.Bu, self.Bd])
        else:
            Luj, _, _ = compute_Lj_and_lambdaj(Lu_new, self.J, separated=True)
            if cascade:
                Bu = compute_vandermonde(Lu_new, self.J).real
                B = np.hstack([Bu, self.Bd])
                return Luj, B
            return Luj

    def zero_out_h(self):
        """Zero-ou the dictionary parameters according to the specific parameterization setup"""

        if self.dictionary_type == "separated":
            hs = np.zeros(
                (self.P, self.J)
            )  # multiplicative coefficients for Upper Laplacian
            hi = np.zeros(
                (self.P, self.J)
            )  # multiplicative coefficients for Lower Laplacian
            hh = np.zeros(
                (self.P, 1)
            )  # multiplicative coefficients for identity matrix
            self.h_opt: List[np.ndarray] = [hh, hs, hi]
        else:
            h = np.zeros((self.P, self.J))
            hi = np.zeros((self.P, 1))
            self.h_opt: List[np.ndarray] = [h, hi]

    def rand_h(self):
        """Randomly initialize the dictionary parameters with null zero vectors according to the specific parameterization setup"""

        if self.dictionary_type == "separated":
            hs = np.random.rand(
                self.P, self.J
            )  # multiplicative coefficients for Upper Laplacian
            hi = np.random.rand(
                self.P, self.J
            )  # multiplicative coefficients for Lower Laplacian
            hh = np.random.rand(
                self.P, 1
            )  # multiplicative coefficients for identity matrix
            self.h_opt: List[np.ndarray] = [hh, hs, hi]
        else:
            h = np.random.rand((self.P, self.J))
            hi = np.random.rand((self.P, 1))
            self.h_opt: List[np.ndarray] = [h, hi]

    def default_solver(self, solver: str, prob: cp.Problem, solver_params: dict = {}):
        """Default solver for the convex optimization problem

        Args:
            solver : str
                Solver to be used for the optimization problem
            prob : cp.Problem
                Convex optimization problem to be solved
            solver_params : dict, optional
                Parameters for the solver, by default {}
        """
        self.init_dict(mode="only_X")
        prob.solve(solver=solver, **solver_params)

    @staticmethod
    def _multiplier_search(
        *arrays: Tuple[np.ndarray], P: int, c: float, epsilon: float
    ) -> Tuple[np.ndarray, bool]:
        """Search for the optimal coefficients for the dictionary initialization

        Args:
            *arrays : Tuple[np.ndarray]
                Arrays to be used for the coefficient generation
            P : int
                Number of kernels (sub-dictionaries)
            c : float
                Boundary constant from the synthetic data generation process
            epsilon : float
                Boundary constant from the synthetic data generation process
        """
        is_okay = 0
        mult = 100
        tries = 0
        while is_okay == 0:
            is_okay = 1
            h, c_try, _, tmp_sum_min, tmp_sum_max = generate_coeffs(
                arrays, P=P, mult=mult
            )
            if c_try <= c:
                is_okay *= 1
            if tmp_sum_min > c - epsilon:
                is_okay *= 1
                incr_mult = 0
            else:
                is_okay = is_okay * 0
                incr_mult = 1
            if tmp_sum_max < c + epsilon:
                is_okay *= 1
                decr_mult = 0
            else:
                is_okay *= 0
                decr_mult = 1
            if is_okay == 0:
                tries += 1
            if tries > 3:
                discard = 1
                break
            if incr_mult == 1:
                mult *= 2
            if decr_mult == 1:
                mult /= 2
        return h, discard

    def init_dict(
        self, h_prior: np.ndarray = None, mode: str = "only_X"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize the dictionary and the signal sparse representation for the alternating
        optimization algorithm.

        Args:
            Lu (np.ndarray): Upper Laplacian matrix
            Ld (np.ndarray): Lower Laplacian matrix
            P (int): Number of kernels (sub-dictionaries).
            J (int): Max order of the polynomial for the single sub-dictionary.
            Y_train (np.ndarray): Training data.
            K0 (int): Sparsity of the signal representation.
            dictionary_type (str): Type of dictionary.
            c (flaot): Dictionary frequency control parameter.
            epsilon (flaot): Dictionary frequency control parameter.
            only (str): Type of initialization. Can be one of: "only_X", "all", "only_D".

        Returns:
            None
        """
        self.min_error_train, self.min_error_test = 1e20, 1e20
        self.zero_out_h()

        # If no prior info on the dictionary
        if np.all(h_prior is None):
            # Init Dictionary
            if mode == "only_D":
                discard = 1
                while discard == 1:
                    if self.dictionary_type != "separated":
                        h_prior, discard = self._multiplier_search(
                            self.lambda_max_j,
                            self.lambda_min_j,
                            P=self.P,
                            c=self.c,
                            epsilon=self.epsilon,
                        )
                        self.D_opt = generate_dictionary(h_prior, self.P, self.Lj)

                    else:
                        h_prior, discard = self._multiplier_search(
                            self.lambda_max_d_j,
                            self.lambda_min_d_j,
                            self.lambda_max_u_j,
                            self.lambda_min_u_j,
                            P=self.P,
                            c=self.c,
                            epsilon=self.epsilon,
                        )
                        self.D_opt = generate_dictionary(
                            h_prior, self.P, self.Luj, self.Ldj
                        )

            # Init Sparse Representations
            elif mode in ["all", "only_X"]:
                L = self.Ld if self.dictionary_type == "edge" else self.L
                _, Dx = sla.eig(L)
                dd = la.norm(Dx, axis=0)
                W = np.diag(1.0 / dd)
                # Dx = Dx / la.norm(Dx)
                Domp = Dx @ W
                X = np.apply_along_axis(
                    lambda x: get_omp_coeff(self.K0, Domp.real, x),
                    axis=0,
                    arr=self.Y_train,
                )
                X = np.tile(X, (self.P, 1))
                self.X_opt_train = X

                if mode == "all":
                    self.rand_h()

        # Otherwise use prior info about the dictionary to initialize both the dictionary and the sparse representation
        else:
            self.h_opt = h_prior

            if self.dictionary_type == "separated":
                self.D_opt = generate_dictionary(h_prior, self.P, self.Luj, self.Ldj)
                self.X_opt_train = sparse_transform(self.D_opt, self.K0, self.Y_train)
            else:
                self.D_opt = generate_dictionary(h_prior, self.P, self.Lj)
                self.X_opt_train = sparse_transform(self.D_opt, self.K0, self.Y_train)

    def topological_dictionary_learn(
        self,
        lambda_: float = 1e-3,
        max_iter: int = 10,
        patience: int = 10,
        tol: float = 1e-7,
        step_h: float = 1.0,
        step_x: float = 1.0,
        solver: str = "MOSEK",
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Dictionary learning algorithm implementation for sparse representations of a signal on complex regular cellular.
        The algorithm consists of an iterative alternating optimization procedure defined in two steps: the positive semi-definite programming step
        for obtaining the coefficients and dictionary based on Hodge theory, and the Orthogonal Matching Pursuit step for constructing
        the K0-sparse solution from the dictionary found in the previous step, which best approximates the original signal.
        Args:
            Y_train (np.ndarray): Training data.
            Y_test (np.ndarray): Testing data.
            J (int): Max order of the polynomial for the single sub-dictionary.
            M (int): Number of data points (number of nodes in the data graph).
            P (int): Number of kernels (sub-dictionaries).
            D0 (np.ndarray): Initial dictionary.
            X0 (np.ndarray): Initial sparse representation.
            Lu (np.ndarray): Upper Laplacian matrix
            Ld (np.ndarray): Lower Laplacian matrix
            dictionary_type (str): Type of dictionary.
            c (float): Boundary constant from the synthetic data generation process.
            epsilon (float): Boundary constant from the synthetic data generation process.
            K0 (int): Sparsity of the signal representation.
            lambda_ (float, optional): Regularization parameter. Defaults to 1e-3.
            max_iter (int, optional): Maximum number of iterations. Defaults to 10.
            patience (int, optional): Patience for early stopping. Defaults to 10.
            tol (float, optional): Tolerance value. Defaults to 1e-s.
            verbose (int, optional): Verbosity level. Defaults to 0.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            minimum training error, minimum testing error, optimal coefficients, optimal testing sparse representation, and optimal training sparse representation.
        """

        # Define hyperparameters
        iter_, pat_iter = 1, 0
        train_hist = []
        test_hist = []

        if self.dict_is_learnable:
            # Init the dictionary and the sparse representation
            D_coll = [
                cp.Constant(self.D_opt[:, (self.M * i) : (self.M * (i + 1))])
                for i in range(self.P)
            ]
            Dsum = cp.Constant(np.zeros((self.M, self.M)))
            h_opt = self.h_opt
            Y = cp.Constant(self.Y_train)
            X_tr = self.X_opt_train
            X_te = self.X_opt_test
            I = cp.Constant(np.eye(self.M))

            while pat_iter < patience and iter_ <= max_iter:
                # SDP Step
                X = cp.Constant(X_tr)
                if iter_ != 1:
                    D_coll = [
                        cp.Constant(D_coll[:, (self.M * i) : (self.M * (i + 1))])
                        for i in range(self.P)
                    ]
                    Dsum = cp.Constant(np.zeros((self.M, self.M)))

                # Define the objective function
                if self.dictionary_type in ["joint", "edge"]:
                    # Init the variables
                    h = cp.Variable((self.P, self.J))
                    hI = cp.Variable((self.P, 1))
                    h.value, hI.value = h_opt
                    for i in range(0, self.P):
                        tmp = cp.Constant(np.zeros((self.M, self.M)))
                        for j in range(0, self.J):
                            tmp += cp.Constant(self.Lj[j, :, :]) * h[i, j]
                        tmp += I * hI[i]
                        D_coll[i] = tmp
                        Dsum += tmp
                    D = cp.hstack([D_coll[i] for i in range(self.P)])
                    term1 = cp.square(cp.norm((Y - D @ X), "fro"))
                    term2 = cp.square(cp.norm(h, "fro") * lambda_)
                    term3 = cp.square(cp.norm(hI, "fro") * lambda_)
                    obj = cp.Minimize(term1 + term2 + term3)

                else:
                    # Init the variables
                    hI = cp.Variable((self.P, self.J))
                    hS = cp.Variable((self.P, self.J))
                    hH = cp.Variable((self.P, 1))
                    hH.value, hS.value, hI.value = h_opt
                    for i in range(0, self.P):
                        tmp = cp.Constant(np.zeros((self.M, self.M)))
                        for j in range(0, self.J):
                            tmp += (cp.Constant(self.Luj[j, :, :]) * hS[i, j]) + (
                                cp.Constant(self.Ldj[j, :, :]) * hI[i, j]
                            )
                        tmp += I * hH[i]
                        D_coll[i] = tmp
                        Dsum += tmp
                    D = cp.hstack([D_coll[i] for i in range(self.P)])

                    term1 = cp.square(cp.norm((Y - D @ X), "fro"))
                    term2 = cp.square(cp.norm(hI, "fro") * lambda_)
                    term3 = cp.square(cp.norm(hS, "fro") * lambda_)
                    term4 = cp.square(cp.norm(hH, "fro") * lambda_)
                    obj = cp.Minimize(term1 + term2 + term3 + term4)

                # Define the constraints
                constraints = (
                    [D_coll[i] >> 0 for i in range(self.P)]
                    + [(cp.multiply(self.c, I) - D_coll[i]) >> 0 for i in range(self.P)]
                    + [
                        (Dsum - cp.multiply((self.c - self.epsilon), I)) >> 0,
                        (cp.multiply((self.c + self.epsilon), I) - Dsum) >> 0,
                    ]
                )

                prob = cp.Problem(obj, constraints)
                prob.solve(solver=eval(f"cp.{solver}"), verbose=False)

                # Dictionary Update
                D = D.value
                if self.dictionary_type in ["joint", "edge"]:
                    h_opt = [
                        h_opt[0] + step_h * (h.value - h_opt[0]),
                        h_opt[1] + step_h * (hI.value - h_opt[1]),
                    ]
                else:
                    h_opt = [
                        h_opt[0] + step_h * (hH.value - h_opt[0]),
                        h_opt[1] + step_h * (hS.value - h_opt[1]),
                        h_opt[2] + step_h * (hI.value - h_opt[2]),
                    ]

                # OMP Step
                X_te_tmp, X_tr_tmp = sparse_transform(
                    D, self.K0, self.Y_test, self.Y_train
                )
                # Sparse Representation Update
                X_tr = X_tr + step_x * (X_tr_tmp - X_tr)
                X_te = X_te + step_x * (X_te_tmp - X_te)

                # Error Update
                error_train = nmse(D, X_tr, self.Y_train, self.m_train)
                error_test = nmse(D, X_te, self.Y_test, self.m_test)
                train_hist.append(error_train)
                test_hist.append(error_test)

                # Error Storing
                if (
                    (error_train < self.min_error_train)
                    and (abs(error_train) > np.finfo(float).eps)
                    and (abs(error_train - self.min_error_train) > tol)
                ):
                    self.X_opt_train = X_tr
                    self.min_error_train = error_train

                if (
                    (error_test < self.min_error_test)
                    and (abs(error_test) > np.finfo(float).eps)
                    and (abs(error_test - self.min_error_test) > tol)
                ):
                    self.h_opt = h_opt
                    self.D_opt = D
                    self.X_opt_test = X_te
                    self.min_error_test = error_test
                    pat_iter = 0

                    if verbose == 1:
                        print("New Best Test Error:", self.min_error_test)
                else:
                    pat_iter += 1

                iter_ += 1

        else:
            # Fourier Dictionary Benchmark
            _, self.D_opt = sla.eigh(self.L)
            self.X_opt_test, self.X_opt_train = sparse_transform(
                self.D_opt, self.K0, self.Y_test, self.Y_train
            )

            # Error Updating
            self.min_error_train = nmse(
                self.D_opt, self.X_opt_train, self.Y_train, self.m_train
            )
            self.min_error_test = nmse(
                self.D_opt, self.X_opt_test, self.Y_test, self.m_test
            )

            train_hist.append(error_train)
            test_hist.append(error_test)

        return self.min_error_test, self.min_error_train, train_hist, test_hist

    def _aux_matrix_update(self, X: np.ndarray) -> None:
        """Update the auxiliary matrix for the dictionary learning step

        Args:
            X (np.ndarray): Updated sparse representation of the signal

        Returns:
            None
        """
        I = [np.eye(self.M)]
        if self.dictionary_type == "separated":
            LL = np.concatenate((I, self.Luj, self.Ldj))
        else:
            LL = np.concatenate((I, self.Lj))
        self.P_aux = np.array(
            [LL @ X[(i * self.M) : ((i + 1) * self.M), :] for i in range(self.P)]
        )
        self.P_aux = rearrange(self.P_aux, "b h w c -> (b h) w c")

    def TDL(
        self,
        lambda_: float = 1e-3,
        lambda2: float = 1e-3,
        max_iter: int = 10,
        patience: int = 10,
        tol: float = 1e-7,
        solver: str = "GUROBI",
        step_h: float = 1.0,
        step_x: float = 1.0,
        verbose: bool = False,
        sparse_coding: str = "OMP",
        third_direction: bool = False,
        Y_tr_pt: torch.Tensor = None,
        Ldj: torch.Tensor = None,
        mode: str = "optimistic",
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """Dictionary learning algorithm implementation for sparse representations of a signal on complex regular cellular.
        The algorithm consists of an iterative alternating optimization procedure defined in two steps: the Quadratic Programming step
        for obtaining the coefficients and dictionary based on Hodge theory, and the Orthogonal Matching Pursuit step for constructing the K0-sparse
        representation from the dictionary found in the previous step, which best approximates the original signal.
        If "third_direction" is set to True, the algorithm will also learn the upper Laplacian of the underlying cell complex.

        Args:
            lambda_ (float, optional): Regularization parameter for the dictionary learning step. Defaults to 1e-3.
            lambda2 (float, optional): Regularization parameter for the polygons' indicator vector. Defaults to 1e-3.
            max_iter (int, optional): Maximum number of iterations. Defaults to 10.
            patience (int, optional): Patience for early stopping. Defaults to 10.
            tol (float, optional): Tolerance value. Defaults to 1e-7.
            solver (str, optional): Solver for the convex optimization problem. Defaults to "GUROBI".
            step_h (float, optional): Step size for the dictionary update. Defaults to 1.0.
            step_x (float, optional): Step size for the sparse representation update. Defaults to 1.0.
            verbose (bool, optional): Verbosity level. Defaults to False.
            sparse_coding (str, optional): Sparse coding method. Defaults to "OMP".
            third_direction (bool, optional): Flag for the third optimization direction. Defaults to False.
            Y_tr_pt (torch.Tensor, optional): Training data in PyTorch format. Defaults to None.
            Ldj (torch.Tensor, optional): Lower Laplacian in PyTorch format. Defaults to None.
            mode (str, optional): Mode for the dictionary initialization. Defaults to "optimistic".

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            minimum training error, minimum testing error, convergence curve for the objective function, convergence curve for the training error, convergence curve for the test error.
        """
        # Define hyperparameters
        iter_, pat_iter, pat_iter2 = 1, 0, 0
        attractor = False
        min_of = np.inf
        train_hist = []
        train_error_hist = []
        test_error_hist = []
        solver = eval(f"cp.{solver}")

        # Learnable Dictionary -> alternating-direction optimization algorithm
        if self.dict_is_learnable:
            h_opt = np.hstack([h.flatten() for h in self.h_opt]).reshape(-1, 1)
            X_tr = self.X_opt_train
            X_te = self.X_opt_test
            D = self.D_opt
            f = 2 if self.dictionary_type == "separated" else 1
            reg = lambda_ * np.eye(self.P * (f * self.J + 1))
            I_s = cp.Constant(np.eye(self.P))
            i_s = cp.Constant(np.ones((self.P, 1)))
            B = cp.Constant(self.B.real)
            patience2 = 60

            while (
                pat_iter <= patience and pat_iter2 < patience2
            ) and iter_ <= max_iter:
                #########################
                #   Dictionary Update   #
                #########################
                h = cp.Variable((self.P * (f * self.J + 1), 1))
                self._aux_matrix_update(X_tr)
                h.value = h_opt
                Q = cp.Constant(
                    np.einsum("imn, lmn -> il", self.P_aux, self.P_aux) + reg
                )
                l = cp.Constant(np.einsum("mn, imn -> i", self.Y_train, self.P_aux))

                # Quadratic term
                term2 = cp.quad_form(h, Q, assume_PSD=True)
                # Linear term
                term1 = l @ h
                term1 = cp.multiply(-2, term1)[0]

                obj = cp.Minimize(term2 + term1)

                # Define the constraints
                cons1 = cp.kron(I_s, B) @ h
                cons2 = cp.kron(i_s.T, B) @ h
                constraints = (
                    [cons1 >= 0]
                    + [cons1 <= self.c]
                    + [cons2 >= (self.c - self.epsilon)]
                    + [cons2 <= (self.c + self.epsilon)]
                )

                prob = cp.Problem(obj, constraints)

                solver_params = {
                    "NumericFocus": 1 | 2 | 3,
                    "Aggregate": 0,
                    "ScaleFlag": 2,
                    "ObjScale": -0.5,
                    "BarHomogeneous": 1,
                    "Method": 1,
                    "verbose": False,
                }
                try:
                    # If we are unable to move from starting conditions -> use default solver parameters
                    if pat_iter > 0 and np.all(h_opt == 0):
                        self.default_solver(solver, prob)
                    else:
                        prob.solve(solver=solver, **solver_params)
                        # If some solver parameters relax too much the problem -> use default solver parameters
                        if prob.status == "infeasible_or_unbounded":
                            self.default_solver(solver, prob)

                except SolverError:
                    # If in any case the solver with tuned parameters fails -> use the default solver parameters
                    try:
                        solver_params = {"QCPDual": 0, "verbose": False}
                        self.default_solver(solver, prob, solver_params)
                    except:
                        solver_params = {"verbose": False}
                        self.default_solver(solver, prob, solver_params)

                # Update the dictionary
                if self.dictionary_type in ["joint", "edge"]:
                    h_list = rearrange_coeffs(h, self.J, self.P)
                    D = generate_dictionary(h_list, self.P, self.Lj)
                    h_opt = h_opt + step_h * (h.value - h_opt)
                else:
                    h_list = rearrange_coeffs(h, self.J, self.P, sep=True)
                    D = generate_dictionary(h_list, self.P, self.Luj, self.Ldj)
                    h_opt = h_opt + step_h * (h.value - h_opt)

                error_train_tmp = nmse(D, X_tr, self.Y_train, self.m_train)

                #########################
                #     Sparse Coding     #
                #########################
                if sparse_coding == "OMP":
                    X_te_tmp, X_tr_tmp = sparse_transform(
                        D, self.K0, self.Y_test, self.Y_train
                    )
                elif sparse_coding == "LASSO":
                    raise ValueError("Sparse coding with 'LASSO' not jet implemented")
                else:
                    raise ValueError("Invalid 'sparse_coding' method")
                # Sparse Representation Update
                X_tr = X_tr + step_x * (X_tr_tmp - X_tr)
                X_te = X_te + step_x * (X_te_tmp - X_te)

                error_train = nmse(D, X_tr, self.Y_train, self.m_train)
                if (error_train - error_train_tmp) > 1e-2:
                    print(f"{error_train - error_train_tmp}")
                    break

                #########################
                #    Topology Update    #
                #########################
                if third_direction:
                    if iter_ > (max_iter / 3):
                        attractor = True
                    self.beta = (iter_ - 2) / (iter_ + 1)
                    _, _, updated, error_train, D = self.topoGD(
                        Y=Y_tr_pt,
                        Ldj=Ldj,
                        lambda1=lambda_,
                        lambda2=lambda2,
                        autodiff=True,
                        max_iter=1,
                        mode=mode,
                        one_attractor=attractor,
                    )

                # Test Error Update
                error_test = nmse(D, X_te, self.Y_test, self.m_test)

                # If learning also the topology do not stop the ADMM according to the NMSE
                # but look at the Objective Function (comprehensive of the regularization terms)
                of = (
                    error_train
                    + lambda_ * la.norm(h_opt, 2)
                    + lambda2 * la.norm(self.p, 1)
                )

                if third_direction:
                    # print(iter_, of, updated, error_train)

                    if not updated or of >= min_of:
                        pat_iter += 1
                    else:
                        pat_iter = 0

                    min_of = of
                    self.h_opt = (
                        h_list if self.dictionary_type == "separated" else h_opt
                    )
                    self.D_opt = D
                    self.X_opt_train = X_tr
                    self.min_error_train = error_train
                    train_error_hist.append(self.min_error_train)
                    train_hist.append(of)
                    self.X_opt_test = X_te
                    self.min_error_test = error_test
                    test_error_hist.append(self.min_error_test)

                # If the ADMM is only for TDL use the MSE as stopping criterion
                else:
                    # print(iter_, self.min_error_train)
                    if (
                        (error_train < self.min_error_train)
                        and (abs(error_train) > np.finfo(float).eps)
                        and (abs(error_train - self.min_error_train) > tol)
                    ):
                        self.h_opt = (
                            h_list if self.dictionary_type == "separated" else h_opt
                        )
                        self.D_opt = D
                        self.X_opt_train = X_tr
                        self.min_error_train = error_train
                        train_error_hist.append(self.min_error_train)
                        train_hist.append(of)

                    if (
                        (error_test < self.min_error_test)
                        and (abs(error_test) > np.finfo(float).eps)
                        and (abs(error_test - self.min_error_test) > tol)
                    ):
                        self.X_opt_test = X_te
                        self.min_error_test = error_test
                        test_error_hist.append(self.min_error_test)
                        pat_iter = 0

                        if verbose:
                            print("New Best Test Error:", self.min_error_test)
                    else:
                        pat_iter += 1

                iter_ += 1

        # Analytic dictionary directly go for the OMP step
        else:
            fit_intercept = True

            # Topological Fourier Dictionary
            if self.dictionary_type == "fourier":
                _, self.D_opt = sla.eig(self.L)

            # Classical Fourier Dictionary
            elif self.dictionary_type == "classic_fourier":
                self.D_opt = sla.dft(self.nd).real

            elif self.dictionary_type == "slepians":
                SS = SimplicianSlepians(
                    self.B1,
                    self.B2,
                    self.S,
                    self.R,
                    verbose=False,
                    top_K=self.top_K_slepians,
                )
                self.D_opt = SS.atoms_flat
                fit_intercept = False

            elif self.dictionary_type == "wavelet":
                SH = SeparateHodgelet(
                    self.B1,
                    self.B2,
                    *log_wavelet_kernels_gen(2, 3, np.log(np.max(self.w1))),
                    *log_wavelet_kernels_gen(2, 3, np.log(np.max(self.w2))),
                )
                self.D_opt = SH.atoms_flat
                fit_intercept = False

            # OMP
            self.X_opt_test, self.X_opt_train = sparse_transform(
                self.D_opt, self.K0, self.Y_test, self.Y_train, fit_intercept
            )
            # Error Updating
            self.min_error_train = nmse(
                self.D_opt, self.X_opt_train, self.Y_train, self.m_train
            )
            self.min_error_test = nmse(
                self.D_opt, self.X_opt_test, self.Y_test, self.m_test
            )

            train_error_hist.append(self.min_error_train)
            test_error_hist.append(self.min_error_test)

        return (
            self.min_error_test,
            self.min_error_train,
            train_hist,
            train_error_hist,
            test_error_hist,
        )

    def mtv(self, gt_mask=None) -> np.ndarray:
        """
        Min Total Variation algorithm, aimed to find the 'q' best candidate triangles
        inside our topology. The class uses this function to contrast the bad initialization problem
        of the coefficient related to the upper laplacian when applying the 'JTDL_greedy()'
        function in 'pessimistic' mode during joint dictionary and topology learning procedure.
        If 'gt_mask' is passed as an argument, the function checks that the selected candidate triangles
        are actually colored in the true topology.

        Args:
            gt_mask (np.ndarray, optional): Ground truth mask for the colored polygons. Defaults to None.

        Returns:
            np.ndarray: Filter for the selected candidate triangles (polygons)
        """

        assert self.q_star < self.nu, (
            "The candidate number of triangles should be smaller than the max number of possible colored triangles in the topology"
        )

        vals, Uirr = sla.eig(self.Ld)
        Uirr = Uirr[:, np.where(vals != 0)[0]]
        Ysh = np.eye(self.nd) - Uirr @ Uirr.T
        d = reduce((Ysh.T @ self.B2) ** 2, "m r -> r", "sum")
        q = np.argsort(d)[: self.q_star]

        if gt_mask != None:
            true_q = np.where(np.sum(gt_mask, axis=1) != 0)[0]
            checks = [q_i in true_q for q_i in q]
            checked = int(np.sum(checks))
            if checked < self.q_star:
                print(
                    f"Warning: {self.q_star - checked} of the {self.q_star} selected candidate triangles are not good!"
                )
            else:
                print("All the selected candidate triangles are good!")
        filter = np.zeros(self.nu)
        filter[q] = 1
        Lu_new = self.B2 @ np.diag(filter) @ self.B2.T
        self.update_Lu(Lu_new)
        return filter

    def JTDL_greedy(
        self,
        Lu_new: np.ndarray = None,
        filter: np.ndarray = 1,
        lambda_: float = 1e-3,
        lambda2: float = 1e-3,
        max_iter: int = 10,
        patience: int = 10,
        tol: float = 1e-7,
        step_h: float = 1.0,
        step_x: float = 1.0,
        mode: str = "optimistic",
        verbose: bool = False,
        warmup: int = 0,
        on_test: bool = False,
        QP=True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Joint Dictionary and Topology Learning algorithm implementation for sparse representations of a signal on second-order cell complex,
        with greedy topology update. The algorithm consists of an iterative alternating optimization procedure defined in three steps: the Quadratic Programming step
        for obtaining the coefficients and dictionary based on Hodge theory, the Orthogonal Matching Pursuit step for constructing the K0-sparse representation from the dictionary found in the previous step,
        and the greedy topology update step for iteratively selecting the best candidate polygons to be colored in the topology.

        Args:
            Lu_new (np.ndarray, optional): Initial upper Laplacian matrix. Defaults to None.
            filter (np.ndarray, optional): Initial filter for the candidate polygons. Defaults to 1.
            lambda_ (float, optional): Regularization parameter for the dictionary learning step. Defaults to 1e-3.
            lambda2 (float, optional): Regularization parameter for polygons' indicator vector. Defaults to 1e-3.
            max_iter (int, optional): Maximum number of iterations. Defaults to 10.
            patience (int, optional): Patience for early stopping. Defaults to 10.
            tol (float, optional): Tolerance value. Defaults to 1e-7.
            step_h (float, optional): Step size for the dictionary update. Defaults to 1.0.
            step_x (float, optional): Step size for the sparse representation update. Defaults to 1.0.
            mode (str, optional): Mode for the greedy algorithm topology initalization. Defaults to "optimistic".
            verbose (bool, optional): Verbosity level. Defaults to False.
            warmup (int, optional): Number of warmup iterations. Defaults to 0.
            on_test (bool, optional): Flag for choosing the candidate polygon looking either at the training or at the test error. Defaults to False.
            QP (bool, optional): Flag for the using the algorithm with QP paramterization of the dictionary update in place of SDP. Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            minimum training error, minimum testing error, convergence curve for the objective function, inferred upper Laplacian Lu, inferred incidence matrix B2.
        """

        assert step_h < 1 or step_h > 0, "You must provide a step-size between 0 and 1."
        assert step_x < 1 or step_x > 0, "You must provide a step-size between 0 and 1."
        assert (mode == "optimistic") or (mode == "pessimistic"), (
            f'{mode} is not a legal mode: "optimistic" or "pessimistic" are the only ones allowed.'
        )

        # Check if we are executing the first recursive iteration
        if np.all(Lu_new is None):
            T = self.B2.shape[1]
            self.warmup = warmup
            self.opt_upper = 0
            # start with a "full" upper Laplacian
            if mode == "optimistic":
                filter = np.ones(T)
            # start with an "empty" upper Laplacian
            elif mode == "pessimistic":
                filter = self.mtv()

        else:
            # if mode == "pessimistic":
            #     self.init_hu()
            self.update_Lu(Lu_new)

        if QP:
            try:
                _, _, train_hist, train_error_hist, _ = self.TDL(
                    lambda_=lambda_,
                    lambda2=lambda2,
                    max_iter=max_iter,
                    patience=patience,
                    tol=tol,
                    step_h=step_h,
                    step_x=step_x,
                    solver="GUROBI",
                )

            except SolverError:
                return (
                    self.min_error_train,
                    self.min_error_test,
                    self.train_history,
                    self.Lu,
                    self.B2,
                )

        else:
            try:
                _, _, train_hist, train_error_hist, _ = (
                    self.topological_dictionary_learn(
                        lambda_=lambda_,
                        max_iter=max_iter,
                        patience=patience,
                        tol=tol,
                        step_h=step_h,
                        step_x=step_x,
                    )
                )
            except SolverError:
                return (
                    self.min_error_train,
                    self.min_error_test,
                    self.train_history,
                    self.Lu,
                    self.B2,
                )

        self.train_history.append(train_hist)
        self.train_error_hist.append(train_error_hist)

        search_space = (
            np.where(filter == 1) if mode == "optimistic" else np.where(filter == 0)
        )
        sigmas = pd.DataFrame({"idx": search_space[0]})

        sigmas["sigma"] = sigmas.idx.apply(lambda _: filter)
        if mode == "optimistic":
            sigmas["sigma"] = sigmas.apply(lambda x: indicator_matrix(x), axis=1)
        else:
            sigmas["sigma"] = sigmas.apply(lambda x: indicator_matrix_rev(x), axis=1)
        sigmas["Luj"] = sigmas.apply(lambda x: compute_Luj(x, self.B2, self.J), axis=1)
        sigmas["D"] = sigmas.apply(
            lambda x: generate_dictionary(self.h_opt, self.P, x.Luj, self.Ldj), axis=1
        )
        if on_test:
            sigmas["X"] = sigmas.D.apply(
                lambda x: sparse_transform(x, self.K0, self.Y_test)
            )
            sigmas["NMSE"] = sigmas.apply(
                lambda x: nmse(x.D, x.X, self.Y_test, self.m_test), axis=1
            )
        else:
            sigmas["X"] = sigmas.D.apply(
                lambda x: sparse_transform(x, self.K0, self.Y_train)
            )
            sigmas["NMSE"] = sigmas.apply(
                lambda x: nmse(x.D, x.X, self.Y_train, self.m_train), axis=1
            )

        candidate_error = sigmas.NMSE.min()
        current_min = self.min_error_test if on_test else self.min_error_train
        idx_min = sigmas.NMSE.idxmin()
        # self.testing_trace[f"{self.opt_upper}"] = (sigmas, current_min, candidate_error)

        # If in warmup look at the third decimal point
        if self.warmup > 0:
            candidate_error = int(candidate_error * 1000)
            current_min = int(current_min * 1000)
            self.warmup -= 1
        # print("TRY", candidate_error)
        if candidate_error <= current_min:
            S = sigmas.sigma[idx_min]
            Lu_new = self.B2 @ S @ self.B2.T
            filter = np.diagonal(S)
            self.opt_upper += 1
            if on_test:
                self.min_error_test = candidate_error
            else:
                self.min_error_train = candidate_error
                self.train_error_hist.append([candidate_error])

            if verbose:
                if mode == "optimistic":
                    print(
                        f"Removing {self.opt_upper} triangles from the topology... \n ... The min error: {candidate_error} !"
                    )
                else:
                    print(
                        f"Adding {self.opt_upper} triangles to the topology... \n ... The min error: {candidate_error:.3f} !"
                    )

            # self.testing_trace[f"{self.opt_upper}"] = (
            #     S,
            #     Lu_new,
            # )  # return the filter flattened matrix and the new best Lu_new

            return self.JTDL_greedy(
                Lu_new=Lu_new,
                filter=filter,
                lambda_=lambda_,
                max_iter=max_iter,
                patience=patience,
                tol=tol,
                step_h=step_h,
                step_x=step_x,
                mode=mode,
                verbose=verbose,
                on_test=on_test,
                QP=QP,
            )

        self.B2 = self.B2 @ np.diag(filter)
        return (
            self.min_error_train,
            self.min_error_test,
            self.train_history,
            self.Lu,
            self.B2,
        )

    def topoGD(
        self,
        Y,
        Ldj,
        lambda1: float,
        lambda2: float,
        autodiff: bool = True,
        max_iter: int = 100,
        decoupled: bool = False,
        mode: str = "optimistic",
        accelerateGD: bool = False,
        verbose: bool = True,
        one_attractor: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, bool, float, np.ndarray]:
        """Proximal Projected Gradient Descent algorithm for the topology learning step. The algorithm consists of an iterative optimization procedure.

        Args:
            Y (np.ndarray): Training data
            Ldj (np.ndarray): Lower Laplacian matrix
            lambda1 (float): Regularization parameter for the dictionary learning step
            lambda2 (float): Regularization parameter for the polygons' indicator vector
            autodiff (bool, optional): Flag for using autodiff. Defaults to True.
            max_iter (int, optional): Maximum number of iterations. Defaults to 100.
            decoupled (bool, optional): Flag for decoupled mode. Defaults to False.
            mode (str, optional): Mode for the dictionary initialization. Defaults to "optimistic".
            accelerateGD (bool, optional): Flag for using the accelerated gradient descent with Nesterov acceleration. Defaults to False.
            verbose (bool, optional): Verbosity level. Defaults to True.
            one_attractor (bool, optional): Flag for using one attractor in proximal operator. Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray, bool, float, np.ndarray]:
            convergence curve for the objective function, convergence curve for the training error, flag for the topology update, minimum training error, updated dictionary.
        """
        # if accelerateGD:
        p_tmp = np.copy(self.p)
        self.v = np.copy(self.p)
        thresh_mode = "soft3" if one_attractor else "soft2"
        c = 1.0 if mode == "optimistic" else 0.5
        train_hist = []
        train_error_hist = []
        i = 1
        updated = True
        while i <= max_iter:
            B2 = torch.tensor(self.B2, dtype=torch.float64)
            if accelerateGD:
                v = torch.tensor(self.v, requires_grad=True)
                Lu_tmp = B2 @ torch.diag(v) @ B2.T
            else:
                p = torch.tensor(self.p, requires_grad=True)
                Lu_tmp = B2 @ torch.diag(p) @ B2.T
            h_list = [torch.from_numpy(h) for h in self.h_opt]
            Luj_tmp = torch.stack(
                [torch.matrix_power(Lu_tmp, i) for i in range(1, self.J + 1)]
            )
            D_tmp = generate_dictionarypt(h_list, self.P, Luj_tmp, Ldj)
            if autodiff:
                X_tr_torch = torch.tensor(self.X_opt_train, dtype=torch.float64)
                fo = nmsept(D_tmp, X_tr_torch, Y, self.m_train)
                fo.backward()
                grad = v.grad.numpy() if accelerateGD else p.grad.numpy()
            else:
                raise ValueError(
                    "No Other methods are implemented for f.o. gradient evaluation"
                )
            if accelerateGD:
                v = self.p + self.beta * (self.p - p_tmp)
                p = proximal_op(self.v - self.mu * grad, lambda2, mode=thresh_mode)
            else:
                p = proximal_op(self.p - self.mu * grad, lambda2, mode=thresh_mode)
            Lu_new = self.B2 @ np.diag(p) @ self.B2.T
            Luj = np.array([la.matrix_power(Lu_new, i) for i in range(1, self.J + 1)])
            D = generate_dictionary(self.h_opt, self.P, Luj, self.Ldj)
            error_train = nmse(D, self.X_opt_train, self.Y_train, self.m_train)

            # Do the check on MSE only if learning the topology in "Doìcoupled" mode
            if decoupled:
                # Stop the GD if it is not improving MSE
                if error_train >= self.min_error_train:
                    if verbose:
                        print(
                            f"Iter {i}: Worse!, {error_train:.5f}  -  Error delta: {self.min_error_train - error_train:.5f}"
                        )
                    break
                else:
                    if verbose:
                        print(
                            f"Iter {i}: Better!, {error_train:.5f}  -  Error delta: {self.min_error_train - error_train:.5f}"
                        )

            h = np.hstack([h.flatten() for h in self.h_opt]).reshape(-1, 1)
            self.min_error_train = error_train
            train_error_hist.append(error_train)
            train_hist.append(
                error_train + lambda1 * la.norm(h, 2) + lambda2 * la.norm(self.p, 1)
            )
            p_tmp = np.copy(self.p)
            self.p = np.copy(p)
            if accelerateGD:
                self.v = np.copy(v)
            self.update_Lu(Lu_new)
            i += 1

        # Check if GD actually updated the topology
        if i == 2:
            if decoupled:
                updated = False
            else:
                t1 = 0.75 if one_attractor else 0.98
                if mode == "optimistic":
                    t2 = 0.5 if one_attractor else 0.6
                    if (
                        np.where(p_tmp == c)[0].shape[0]
                        == np.where(self.p == c)[0].shape[0]
                    ) and (self.p[(self.p < t1) & (self.p > t2)].shape[0] == 0):
                        updated = False
                else:
                    t2 = 0.3 if one_attractor else 0.4
                    if self.p[(self.p < t1) & (self.p > t2)].shape[0] == 0:
                        updated = False

        # if verbose:
        #     print(self.p)
        return train_hist, train_error_hist, updated, error_train, D

    def JTDL(
        self,
        lambda1: float = 1e-3,
        lambda2: float = 1e-3,
        max_iter_in: int = 10,
        max_iter_out: int = 10,
        max_iter_tot: int = 10,
        mu: float = 1e-6,
        patience: int = 10,
        tol: float = 1e-7,
        tol_out: float = 1e-7,
        solver: str = "GUROBI",
        step_h: float = 1.0,
        step_x: float = 1.0,
        sparse_coding: str = "OMP",
        autodiff: bool = True,
        mode: str = "optimistic",
        decouple_learning: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """Joint Dictionary and Topology Learning algorithm implementation for sparse representations of a signal on second-order cell complex,
        with 'relaxed' topology update.

        Args:

        """
        assert self.dictionary_type == "separated", (
            "joint topology and dictionary learning is only implemented with the 'separated' dictionary parameterization."
        )

        train_hist = []
        iter_ = 1
        self.p = (
            np.ones(self.nu)
            if mode == "optimistic"
            # else np.full(self.nu, lambda2 + 0.1)
            else np.full(self.nu, 0.5)
        )
        Y_tr_torch = torch.from_numpy(self.Y_train)
        Ldj = torch.from_numpy(
            self.Ldj,
        )
        self.mu = mu

        ############################################
        #   Decoupled Topology and Dict. Learning  #
        ############################################
        if decouple_learning:
            while iter_ <= max_iter_tot:
                #########################################
                #    Topological Dictionary Learning    #
                #########################################
                try:
                    _, _, train_hist, train_error_hist, _ = self.TDL(
                        lambda_=lambda1,
                        lambda2=lambda2,
                        max_iter=max_iter_in,
                        patience=patience,
                        tol=tol,
                        step_h=step_h,
                        step_x=step_x,
                        solver=solver,
                        sparse_coding=sparse_coding,
                    )

                except SolverError:
                    pass

                self.train_history.append(train_hist)
                self.train_error_hist.append(train_error_hist)

                #########################
                #    Topology Update    #
                #########################
                train_hist, train_error_hist, updated, _ = self.topoGD(
                    Y=Y_tr_torch,
                    Ldj=Ldj,
                    lambda1=lambda1,
                    lambda2=lambda2,
                    autodiff=autodiff,
                    max_iter=max_iter_out,
                    mode=mode,
                    decoupled=True,
                )
                if not updated:
                    print("Unable to update the topology... stop learning!")
                    break
                else:
                    self.train_history.append(train_hist)
                    self.train_error_hist.append(train_error_hist)

                iter_ += 1

        #########################################
        #     Three-direction ADMM for JTDL     #
        #########################################
        else:
            try:
                _, _, train_hist, train_error_hist, _ = self.TDL(
                    lambda_=lambda1,
                    lambda2=lambda2,
                    max_iter=max_iter_in,
                    patience=patience,
                    tol=tol,
                    step_h=step_h,
                    step_x=step_x,
                    solver=solver,
                    sparse_coding=sparse_coding,
                    third_direction=True,
                    Y_tr_pt=Y_tr_torch,
                    Ldj=Ldj,
                    mode=mode,
                    # max_iter_out=max_iter_out,
                )

            except SolverError:
                return (
                    self.min_error_train,
                    self.min_error_test,
                    self.train_history,
                    self.Lu,
                    self.B2,
                )
            self.train_history.append(train_hist)
            self.train_error_hist.append(train_error_hist)

        self.p_probs = np.copy(self.p)
        self.p = np.where(self.p <= 0.5, 0, 1)
        # print(self.p)  #####################
        # self.p = np.where(self.p < 0.8, 0, 1)
        self.B2 = self.B2 @ np.diag(self.p)
        self.update_Lu(self.B2 @ self.B2.T)
        try:
            _, _, train_hist, train_error_hist, _ = self.TDL(
                lambda_=lambda1,
                lambda2=lambda2,
                max_iter=max_iter_in,
                patience=patience,
                tol=tol,
                step_h=step_h,
                step_x=step_x,
                solver=solver,
                sparse_coding=sparse_coding,
                third_direction=False,
            )

        except SolverError:
            return (
                self.min_error_train,
                self.min_error_test,
                self.train_history,
                self.Lu,
                self.B2,
            )
        self.train_history.append(train_hist)
        self.train_error_hist.append(train_error_hist)

        return (
            self.min_error_train,
            self.min_error_test,
            self.train_history,
            self.Lu,
            self.B2,
        )

    def track_performances(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            process = psutil.Process()
            cpu_start = process.cpu_times()
            mem_start = process.memory_info().rss  # Resident Set Size (RSS) in bytes

            start_time = time.time()
            result = method(self, *args, **kwargs)
            end_time = time.time()

            cpu_end = process.cpu_times()
            mem_end = process.memory_info().rss

            performance_data = {
                "execution_time": end_time - start_time,
                "memory_usage_mb": (mem_end - mem_start) / (1024 * 1024),
                "cpu_time_user": cpu_end.user - cpu_start.user,
                "cpu_time_system": cpu_end.system - cpu_start.system,
            }

            # Store in an instance attribute
            if not hasattr(self, "performance_metrics"):
                self.performance_metrics = {}
            self.performance_metrics[method.__name__] = performance_data

            return result

        return wrapper

    def save_results(func):
        """Decorator to save intermediate results when testing learning functions"""

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            outputs = func(self, *args, **kwargs)
            func_name = func.__name__

            if func_name == "test_topological_dictionary_learn":
                path = os.getcwd()
                dir_path = os.path.join(
                    path, "results", "dictionary_learning", f"{self.dictionary_type}"
                )
                name = f"learn_D_{self.dictionary_type}"
                filename = os.path.join(dir_path, f"{name}.pkl")
                save_var = {
                    "min_error_test": self.min_error_test,
                    "min_error_train": self.min_error_train,
                    "train_history": outputs[2],
                    "test_history": outputs[3],
                    "h_opt": self.h_opt,
                    "X_opt_test": self.X_opt_test,
                    "X_opt_train": self.X_opt_train,
                    "D_opt": self.D_opt,
                }

            elif func_name == "test_TDL":
                path = os.getcwd()
                dir_path = os.path.join(path, "results", "no_topology_learning")
                name = f"learn_T{int(self.true_prob_T * 100)}"
                filename = os.path.join(dir_path, f"{name}.pkl")
                save_var = {
                    "min_error_test": self.min_error_test,
                    "min_error_train": self.min_error_train,
                    "train_history": outputs[2],
                    "test_history": outputs[3],
                    "h_opt": self.h_opt,
                    "X_opt_test": self.X_opt_test,
                    "X_opt_train": self.X_opt_train,
                    "D_opt": self.D_opt,
                }

            elif func_name == "test_JTDL_greedy":
                path = os.getcwd()
                dir_path = os.path.join(path, "results", "topology_learning")
                name = f"learn_T{int(self.true_prob_T * 100)}"
                filename = os.path.join(dir_path, f"{name}.pkl")
                save_var = {
                    "min_error_test": self.min_error_test,
                    "min_error_train": self.min_error_train,
                    "train_history": self.train_history,
                    "test_history": self.test_history,
                    "Lu_opt": self.Lu,
                    "B2_opt": self.B2,
                    "h_opt": self.h_opt,
                    "X_opt_test": self.X_opt_test,
                    "X_opt_train": self.X_opt_train,
                    "D_opt": self.D_opt,
                }

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            try:
                with open(filename, "wb") as file:
                    pickle.dump(save_var, file)
            except IOError as e:
                print(f"An error occurred while writing the file: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

            return outputs

        return wrapper

    @save_results
    def test_topological_dictionary_learn(
        self,
        mode: str = "only_X",
        lambda_: float = 1e-7,
        max_iter: int = 100,
        patience: int = 5,
        tol: float = 1e-7,
        solver: str = "MOSEK",
        step_h: float = 1.0,
        step_x: float = 1.0,
        verbose: bool = False,
    ) -> None:
        try:
            self.init_dict(mode=mode)
        except:
            print("Initialization Failed!")

        self.topological_dictionary_learn(
            lambda_, max_iter, patience, tol, solver, step_h, step_x, verbose
        )

    @save_results
    def test_TDL(
        self,
        mode: str = "only_X",
        lambda_: float = 1e-7,
        max_iter: int = 100,
        patience: int = 5,
        tol: float = 1e-7,
        solver: str = "GUROBI",
        step_h: float = 1.0,
        step_x: float = 1.0,
        verbose: bool = False,
    ) -> None:
        try:
            self.init_dict(mode=mode)
        except:
            print("Initialization Failed!")

        self.TDL(lambda_, max_iter, patience, tol, solver, step_h, step_x, verbose)

    @save_results
    def test_JTDL_greedy(
        self,
        init_mode: str = "only_X",
        lambda_: float = 1e-7,
        max_iter: int = 100,
        patience: int = 5,
        tol: float = 1e-7,
        step_h: float = 1.0,
        step_x: float = 1.0,
        mode: str = "optimistic",
        verbose: bool = True,
        warmup: int = 0,
        QP: bool = True,
    ) -> None:
        try:
            self.init_dict(mode=init_mode)
        except:
            print("Initialization Failed!")

        self.JTDL_greedy(
            lambda_=lambda_,
            max_iter=max_iter,
            patience=patience,
            tol=tol,
            step_h=step_h,
            step_x=step_x,
            mode=mode,
            verbose=verbose,
            warmup=warmup,
            QP=QP,
        )

    def get_topology_approx_error(self, Lu_true, round_res=None):
        res = la.norm(Lu_true - self.Lu, ord="fro") / la.norm(self.Lu, ord="fro")
        if round_res == None:
            return res
        return np.round(res, round_res)

    def get_test_error(self, round_res):
        if round_res == None:
            return self.min_error_test
        return np.round(self.min_error_test, round_res)

    def get_train_error(self, round_res=None):
        if round_res == None:
            return self.min_error_train
        return np.round(self.min_error_test, round_res)

    def get_numb_triangles(self, mode: str = None):
        if mode == "optimistic":
            return self.nu - self.opt_upper
        elif mode == "pessimistic":
            return self.nu + self.opt_upper
        return self.nu - self.T

    @track_performances
    def fit(
        self, Lu_true, init_mode="only_X", learn_topology=True, soft=True, **hyperparams
    ):
        # DEFAUTLS
        hp = {
            "lambda_": 1e-7,
            "lambda2": 1e-7,
            "tol": 1e-7,
            "tol_out": 1e-7,
            "patience": 5,
            "patience_out": 10,
            "max_iter": 100,
            "max_iter_out": 100,
            "max_iter_tot": 100,
            "mu": 1e-6,
            "decouple_learning": False,
            "step_x": 1.0,
            "step_h": 1.0,
            "QP": True,
            "mode": "optimistic",
            "autodiff": True,
            "sparse_coding": "OMP",
            "verbose": False,
            "on_test": False,
        }

        hp.update(hyperparams)

        try:
            self.init_dict(mode=init_mode)
        except:
            print("Initialization Failed!")

        if learn_topology:
            if soft:
                self.JTDL(
                    lambda1=hp["lambda_"],
                    lambda2=hp["lambda2"],
                    max_iter_in=hp["max_iter"],
                    max_iter_out=hp["max_iter_out"],
                    max_iter_tot=hp["max_iter_tot"],
                    mu=hp["mu"],
                    patience=hp["patience"],
                    tol=hp["tol"],
                    tol_out=hp["tol_out"],
                    step_h=hp["step_h"],
                    step_x=hp["step_x"],
                    solver="GUROBI",
                    sparse_coding=hp["sparse_coding"],
                    autodiff=hp["autodiff"],
                    mode=hp["mode"],
                    decouple_learning=hp["decouple_learning"],
                )
            else:
                self.JTDL_greedy(
                    lambda_=hp["lambda_"],
                    max_iter=hp["max_iter"],
                    patience=hp["patience"],
                    tol=hp["tol"],
                    step_h=hp["step_h"],
                    step_x=hp["step_x"],
                    mode=hp["mode"],
                    verbose=hp["verbose"],
                    warmup=hp["warmup"],
                    on_test=hp["on_test"],
                    QP=hp["QP"],
                )
        else:
            self.TDL(
                lambda_=hp["lambda_"],
                lambda2=hp["lambda2"],
                max_iter=hp["max_iter"],
                patience=hp["patience"],
                tol=hp["tol"],
                step_h=hp["step_h"],
                step_x=hp["step_x"],
                solver="GUROBI",
                sparse_coding=hp["sparse_coding"],
            )

        Lu_approx_error = self.get_topology_approx_error(Lu_true=Lu_true)

        return self.min_error_train, self.min_error_test, Lu_approx_error

    def spectral_control_params(self, num=20, verbose=True):
        L = self.Ld if self.dictionary_type == "edge" else self.L
        vals, _ = sla.eig(L)
        c_in = np.sort(vals)[-1].real
        e_in = np.std(vals).real
        c_end = c_in**self.J
        e_end = e_in**self.J
        c_space = np.linspace(c_in, c_end, num=num)
        e_space = np.linspace(e_in, e_end, num=num)
        current_min = self.min_error_test

        for e in e_space:
            for c in c_space:
                self.epsilon = e
                self.c = c
                self.init_dict(mode="only_X")
                self.TDL(lambda_=1e-7, max_iter=1)
                if self.min_error_test < current_min:
                    current_min = self.min_error_test

        if verbose:
            print(f"Best c {self.c}")
            print(f"Best eps {self.epsilon}")

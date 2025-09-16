# written by T. Mitchell Roddenberry, July 2021
# follow me on twitter: mitch.roddenberry.xyz/rss.xml

# Thanks to Mitch Roddenberry (Rice ECE) for sharing the code from his paper:
# "Hodgelets: Localized Spectral Representations of Flows on Simplicial Complexes".

import numpy as np
import pandas as pd
from scipy.linalg import svdvals
from sklearn.linear_model import (
    Lasso,
    OrthogonalMatchingPursuit,
    OrthogonalMatchingPursuitCV,
)

import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

from src.utilsHodgelets import *

# ~~~~~~~~~~~~~~~~~#
# wavelet classes #
# ~~~~~~~~~~~~~~~~~#


class Hodgelet:
    def __init__(self, B1, B2):
        self.N0 = self.N1 = self.N2 = 0
        self.B1 = self.B2 = self.L1 = self.L2 = self.L = None
        self.lower = B1 is not None
        self.upper = B2 is not None

        if self.lower:
            self.N0, self.N1 = B1.shape

        if self.upper:
            self.N1, self.N2 = B2.shape

        self.B1 = B1
        self.B2 = B2
        self.L, self.L2, self.L1 = hodge_laplacians(self.B1, self.B2)
        self.atoms_flat = np.eye(self.N1)

    def __str__(self):
        return f"Generic Hodgelet class:\nN0={self.N0}\nN1={self.N1}\nN2={self.N2}"

    def frame(self):
        s = svdvals(self.atoms_flat)
        return s[-1] ** 2, s[0] ** 2

    def babel(self, ks):
        G = sort_rows(gram_matrix(self.atoms_flat))
        return [np.max(np.sum(G[:, 1 : k + 1], axis=1)) for k in ks]

    def omp(self, x, cv=False, k=None, err=None):
        if cv:
            reg = OrthogonalMatchingPursuitCV(fit_intercept=False).fit(
                self.atoms_flat, x
            )
        else:
            reg = OrthogonalMatchingPursuit(
                tol=err, n_nonzero_coefs=k, fit_intercept=False
            ).fit(self.atoms_flat, x)
        return reg

    def lasso_(self, x, alpha_lasso=None, max_iter_lasso=20000):
        reg = Lasso(alpha=alpha_lasso, max_iter=max_iter_lasso).fit(self.atoms_flat, x)
        return reg

    def iht(self, x, M, thresh=1e-15):
        m = self.atoms_flat.shape[1]
        reg = IHT(x, self.atoms_flat, np.linalg.pinv(self.atoms_flat), m, M, thresh)
        return reg

    def flat_transform(self, x):
        return np.einsum("ji,j...->i...", self.atoms_flat, x)

    def uft(self, x, sg_sq, x_oracle=None):
        W = self.atoms_flat.T
        Wp = np.linalg.pinv(W)

        if x_oracle is not None:
            mu = W @ x_oracle
        else:
            mu = W @ x

        A = np.linalg.norm(np.outer(mu, mu) * (Wp.T @ Wp), 1, axis=1)
        B = sg_sq * np.linalg.norm((Wp.T @ Wp) * (W @ W.T), 1, axis=1)
        T = (B < A).astype(int)
        return Wp @ np.diag(T) @ W @ x


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


class FourierBasis(Hodgelet):
    def __init__(self, B1, B2):
        super().__init__(B1, B2)
        _, self.atoms_flat = np.linalg.eigh(self.L)

    def __str__(self):
        return f"Fourier Basis class:\nN0={self.N0}\nN1={self.N1}\nN2={self.N2}"

    def theoretical_frame(self):
        return 1, 1


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


class SimplicianSlepians(Hodgelet):
    def __init__(
        self,
        B1,
        B2,
        S=None,
        F=None,
        verbose=True,
        top_K=None,
        save_dic=True,
        save_dir="/Users/Claudio/Desktop/hodgelets/figs/hexgrid/slepians/saved_dictionaries",
        load_dir="/Users/Claudio/Desktop/hodgelets/figs/hexgrid/slepians/saved_dictionaries",
    ):
        super().__init__(B1, B2)
        try:
            self.atoms_flat = pd.read_csv(
                load_dir + "/top_K_" + str(top_K) + ".csv", header=None
            ).to_numpy()[1:, 1:]
            self.full_rank = 1
        except:
            self.eigenvals, self.U = np.linalg.eigh(self.L)
            self.idx = self.eigenvals.argsort()[::-1]
            self.eigenvals = self.eigenvals[self.idx]
            self.U = self.U[:, self.idx]
            self.top_K = top_K

            # Harmonic
            idx_harm = abs(self.eigenvals) < 1e-12
            if sum(idx_harm) > 0:
                if verbose:
                    print("Harmonic Component is Present!")
                self.atoms_har_np = self.U[:, idx_harm]

            self.atoms_sol = []
            self.Sigma_sol = np.diag(F[0][:, 0])
            self.atoms_irr = []
            self.Sigma_irr = np.diag(F[1][:, 0])
            self.B_sol = self.U @ self.Sigma_sol @ self.U.T
            self.B_irr = self.U @ self.Sigma_irr @ self.U.T
            for e_set in range(max(len(S[0]), len(S[1]))):
                try:
                    # Solenoidal
                    self.D = np.diag(S[0][e_set])
                    self.X = self.B_sol @ self.D @ self.B_sol
                    self.X = (self.X + self.X.T) / 2
                    self.eigenvals, self.atoms_tmp = np.linalg.eigh(self.X)
                    self.eigenvals[abs(self.eigenvals) < 1e-8] = 0
                    self.idx = self.eigenvals.argsort()[::-1]
                    self.eigenvals = self.eigenvals[self.idx]
                    self.atoms_tmp = self.atoms_tmp[:, self.idx]
                    X_rank = np.linalg.matrix_rank(self.X)
                    if self.top_K != None and self.top_K < X_rank:
                        self.atoms_sol.append(self.atoms_tmp[:, 0 : self.top_K])
                    else:
                        self.atoms_sol.append(self.atoms_tmp[:, self.eigenvals != 0])
                except:
                    pass

                try:
                    # Irrotational
                    self.D = np.diag(S[1][e_set])
                    self.B = self.U @ self.Sigma_irr @ self.U.T
                    self.X = self.B_irr @ self.D @ self.B_irr
                    self.X = (self.X + self.X.T) / 2
                    self.eigenvals, self.atoms_tmp = np.linalg.eigh(self.X)
                    self.eigenvals[abs(self.eigenvals) < 1e-8] = 0
                    self.idx = self.eigenvals.argsort()[::-1]
                    self.eigenvals = self.eigenvals[self.idx]
                    self.atoms_tmp = self.atoms_tmp[:, self.idx]
                    X_rank = np.linalg.matrix_rank(self.X)
                    if self.top_K != None and self.top_K < X_rank:
                        self.atoms_irr.append(self.atoms_tmp[:, 0 : self.top_K])
                    else:
                        self.atoms_irr.append(self.atoms_tmp[:, self.eigenvals != 0])
                except:
                    pass

            self.atoms_irr_np = np.concatenate(self.atoms_irr, axis=1)
            self.atoms_sol_np = np.concatenate(self.atoms_sol, axis=1)
            # Overcomplete Dictionary
            if sum(idx_harm) > 0:
                self.atoms_flat = np.hstack(
                    [self.atoms_sol_np, self.atoms_irr_np, self.atoms_har_np]
                )
            else:
                self.atoms_flat = np.hstack([self.atoms_sol_np, self.atoms_irr_np])
            if np.linalg.matrix_rank(self.atoms_flat) < B2.shape[1]:
                # print("Rank Not sufficient! Exiting...")
                self.full_rank = 0
            else:
                self.full_rank = 1
            self.atoms_flat[abs(self.atoms_flat) < 1e-12] = 1e-7
            try:
                pd.DataFrame(self.atoms_flat).to_csv(
                    save_dir + "/top_K_" + str(top_K) + ".csv"
                )
            except:
                if verbose:
                    print("Saving Directory not Valid!")

    def __str__(self):
        return f"Simplicial Slepians class:\nN0={self.N0}\nN1={self.N1}\nN2={self.N2}\nFrequencySet={self.F}\nEdgeSet={self.V}"

    def theoretical_frame(self):
        return 1, 1

    def transform(self, x):
        return np.einsum("ijk,k...->ij...", self.atoms, x)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


class JointHodgelet(Hodgelet):
    def __init__(self, B1, B2, scale, kernels):
        super().__init__(B1, B2)

        w, v = np.linalg.eigh(self.L)
        w_max = np.max(w)

        self.scale = scale
        self.kernels = kernels

        # N1 x M x N1
        self.atoms = np.array(
            [v @ np.diag(k(w)) @ v.T for k in [self.scale] + self.kernels]
        ).transpose([2, 0, 1])

        # N1 x M*N1
        self.atoms_flat = self.atoms.reshape(-1, self.N1).T

    def __str__(self):
        return f"Joint Hodgelet class:\nN0={self.N0}\nN1={self.N1}\nN2={self.N2}"

    def theoretical_frame(self):
        w = np.linalg.eigvalsh(self.L)
        S = self.scale(w) ** 2
        G = S + np.sum(np.array([k(w) ** 2 for k in self.kernels]), axis=0)

        return np.min(G), np.max(G)

    def transform(self, x):
        return np.einsum("ijk,k...->ij...", self.atoms, x)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


class SeparateHodgelet(Hodgelet):
    def __init__(self, B1, B2, scale1, kernels1, scale2, kernels2):
        super().__init__(B1, B2)

        self.scale1 = self.kernels1 = None
        self.scale2 = self.kernels2 = None
        self.atoms1 = self.atoms2 = None

        if self.lower:
            self.scale1 = scale1
            self.kernels1 = kernels1
            self.atoms1 = create_atoms(self.L1, [self.scale1] + self.kernels1)
            atoms1_flat = self.atoms1.reshape(-1, self.N1).T

        if self.upper:
            self.scale2 = scale2
            self.kernels2 = kernels2
            self.atoms2 = create_atoms(self.L2, [self.scale2] + self.kernels2)
            atoms2_flat = self.atoms2.reshape(-1, self.N1).T

        if not self.lower:
            self.atoms_flat = atoms2_flat
        elif not self.upper:
            self.atoms_flat = atoms1_flat
        else:
            self.atoms_flat = np.hstack([atoms1_flat, atoms2_flat])

    def __str__(self):
        return f"Separate Hodgelet class:\nN0={self.N0}\nN1={self.N1}\nN2={self.N2}"

    def theoretical_frame(self):
        S1 = [0]
        G1 = [0]
        S2 = [0]
        G2 = [0]

        if self.lower:
            w1 = np.linalg.eigvalsh(self.L1)
            S1 = self.scale1(w1) ** 2
            G1 = S1 + np.sum(np.array([k(w1) ** 2 for k in self.kernels1]), axis=0)

        if self.upper:
            w2 = np.linalg.eigvalsh(self.L2)
            S2 = self.scale2(w2) ** 2
            G2 = S2 + np.sum(np.array([k(w2) ** 2 for k in self.kernels2]), axis=0)

        lower1 = np.min(G1) + S2[0]
        lower2 = np.min(G2) + S1[0]
        upper1 = np.max(G1) + S2[0]
        upper2 = np.max(G2) + S1[0]

        if not self.lower:
            lower = lower2
            upper = upper2
        elif not self.upper:
            lower = lower1
            upper = upper1
        else:
            lower = np.minimum(lower1, lower2)
            upper = np.maximum(upper1, upper2)

        return lower, upper

    def transform(self, x):
        y1 = y2 = None
        if self.lower:
            y1 = np.einsum("ijk,k...->ij...", self.atoms1, x)
        if self.upper:
            y2 = np.einsum("ijk,k...->ij...", self.atoms2, x)

        return y1, y2


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


class LiftedHodgelet(Hodgelet):
    def __init__(self, B1, B2, scale1, kernels1, scale2, kernels2, eps=1e-8):
        super().__init__(B1, B2)

        self.eps = eps

        self.scale1 = self.kernels1 = None
        self.scale2 = self.kernels2 = None
        self.scaling1 = self.wavelets1 = None
        self.scaling2 = self.wavelets2 = None

        if self.lower:
            self.scale1 = scale1
            self.kernels1 = kernels1
            self.scaling1, self.wavelets1 = lifted_atoms(
                self.B1, self.scale1, self.kernels1, self.eps
            )
            scaling1_flat = self.scaling1.reshape(-1, self.N1).T
            wavelets1_flat = self.wavelets1.reshape(-1, self.N1).T
            atoms1_flat = np.hstack([scaling1_flat, wavelets1_flat])

        if self.upper:
            self.scale2 = scale2
            self.kernels2 = kernels2
            self.scaling2, self.wavelets2 = lifted_atoms(
                self.B2.T, self.scale2, self.kernels2, self.eps
            )
            scaling2_flat = self.scaling2.reshape(-1, self.N1).T
            wavelets2_flat = self.wavelets2.reshape(-1, self.N1).T
            atoms2_flat = np.hstack([scaling2_flat, wavelets2_flat])

        if not self.lower:
            self.atoms_flat = atoms2_flat
        elif not self.upper:
            self.atoms_flat = atoms1_flat
        else:
            self.atoms_flat = np.hstack([atoms1_flat, atoms2_flat])

    def __str__(self):
        return f"Lifted Hodgelet class:\nN0={self.N0}\nN1={self.N1}\nN2={self.N2}"

    def theoretical_frame(self):
        S1 = [0]
        G1 = [0]
        S2 = [0]
        G2 = [0]
        lower1 = lower2 = 0
        upper1 = upper2 = 0

        if self.lower:
            s1 = svdvals(self.B1)
            lower1 = 1 / np.max(s1[np.where(s1 > self.eps)]) ** 2
            upper1 = 1 / np.min(s1[np.where(s1 > self.eps)]) ** 2

            w1 = np.linalg.eigvalsh(self.L1)
            S1 = self.scale1(w1) ** 2
            G1 = np.sum(np.array([k(w1) ** 2 for k in self.kernels1]), axis=0)

        if self.upper:
            s2 = svdvals(self.B2)
            lower2 = 1 / np.max(s2[np.where(s2 > self.eps)]) ** 2
            upper2 = 1 / np.min(s2[np.where(s2 > self.eps)]) ** 2

            w2 = np.linalg.eigvalsh(self.L2)
            S2 = self.scale2(w2) ** 2
            G2 = np.sum(np.array([k(w2) ** 2 for k in self.kernels2]), axis=0)

        lowerb_1 = np.min(S1 + lower1 * G1) + S2[0]
        lowerb_2 = np.min(S2 + lower2 * G2) + S1[0]

        upperb_1 = np.max(S1 + upper1 * G1) + S2[0]
        upperb_2 = np.max(S2 + upper2 * G2) + S1[0]

        if not self.lower:
            lower = lowerb_2
            upper = upperb_2
        elif not self.upper:
            lower = lowerb_1
            upper = upperb_1
        else:
            lower = np.minimum(lowerb_1, lowerb_2)
            upper = np.maximum(upperb_1, upperb_2)

        return lower, upper

    def transform(self, x):
        s1 = s2 = None
        y1 = y2 = None
        if self.lower:
            s1 = np.einsum("ijk,k...->ij...", self.scaling1, x)
            y1 = np.einsum("ijk,k...->ij...", self.wavelets1, x)
        if self.upper:
            s2 = np.einsum("ijk,k...->ij...", self.scaling2, x)
            y2 = np.einsum("ijk,k...->ij...", self.wavelets2, x)

        return s1, s2, y1, y2


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


class MixedLiftedHodgelet(Hodgelet):
    def __init__(self, B1, B2, scale1, kernels1, scale2, kernels2, eps=1e-8):
        super().__init__(B1, B2)

        self.eps = eps

        self.scale1 = self.kernels1 = None
        self.scale2 = self.kernels2 = None
        self.scaling1 = self.wavelets1 = None
        self.scaling2 = self.wavelets2 = None

        if self.lower:
            self.scale1 = scale1
            self.kernels1 = kernels1

            if self.N1 > self.N0:
                self.scaling1, self.wavelets1 = lifted_atoms(
                    self.B1, self.scale1, self.kernels1, self.eps
                )
            else:
                atoms1 = create_atoms(self.L1, [self.scale1] + self.kernels1)
                self.scaling1 = atoms1[0]
                self.wavelets1 = atoms1[1:]

            scaling1_flat = self.scaling1.reshape(-1, self.N1).T
            wavelets1_flat = self.wavelets1.reshape(-1, self.N1).T
            atoms1_flat = np.hstack([scaling1_flat, wavelets1_flat])

        if self.upper:
            self.scale2 = scale2
            self.kernels2 = kernels2

            if self.N1 > self.N2:
                self.scaling2, self.wavelets2 = lifted_atoms(
                    self.B2.T, self.scale2, self.kernels2, self.eps
                )
            else:
                atoms2 = create_atoms(self.L2, [self.scale2] + self.kernels2)
                self.scaling2 = atoms2[0]
                self.wavelets2 = atoms2[1:]

            scaling2_flat = self.scaling2.reshape(-1, self.N1).T
            wavelets2_flat = self.wavelets2.reshape(-1, self.N1).T
            atoms2_flat = np.hstack([scaling2_flat, wavelets2_flat])

        if not self.lower:
            self.atoms_flat = atoms2_flat
        elif not self.upper:
            self.atoms_flat = atoms1_flat
        else:
            self.atoms_flat = np.hstack([atoms1_flat, atoms2_flat])

    def __str__(self):
        return f"Lifted Hodgelet class:\nN0={self.N0}\nN1={self.N1}\nN2={self.N2}"

    def transform(self, x):
        s1 = s2 = None
        y1 = y2 = None
        if self.lower:
            s1 = np.einsum("ijk,k...->ij...", self.scaling1, x)
            y1 = np.einsum("ijk,k...->ij...", self.wavelets1, x)
        if self.upper:
            s2 = np.einsum("ijk,k...->ij...", self.scaling2, x)
            y2 = np.einsum("ijk,k...->ij...", self.wavelets2, x)

        return s1, s2, y1, y2


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


class Laplacelet:
    def __init__(self, L):
        self.L = L
        self.N1, _ = L.shape

        # default set of atoms: identity matrix
        # each *column* is a dictionary atom
        self.atoms_flat = np.eye(self.N1)

    def __str__(self):
        return f"Generic Laplacelet class:\nN1={self.N1}"

    def frame(self):
        s = svdvals(self.atoms_flat)
        return s[-1] ** 2, s[0] ** 2

    def babel(self, ks):
        G = sort_rows(gram_matrix(self.atoms_flat))
        return [np.max(np.sum(G[:, 1 : k + 1], axis=1)) for k in ks]

    def omp(self, x, cv=False, k=None, err=None):
        if cv:
            reg = OrthogonalMatchingPursuitCV(fit_intercept=False).fit(
                self.atoms_flat, x
            )
        else:
            reg = OrthogonalMatchingPursuit(
                tol=err, n_nonzero_coefs=k, fit_intercept=False
            ).fit(self.atoms_flat, x)

        return reg

    def flat_transform(self, x):
        return np.einsum("ji,j...->i...", self.atoms_flat, x)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


class LaplaceFourierBasis(Laplacelet):
    def __init__(self, L):
        super().__init__(L)

        _, self.atoms_flat = np.linalg.eigh(self.L)

    def __str__(self):
        return f"Generic Laplace Fourier Basis class:\nN1={self.N1}"

    def theoretical_frame(self):
        return 1, 1


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


class JointLaplacelet(Laplacelet):
    def __init__(self, L, scale, kernels):
        super().__init__(L)

        w, v = np.linalg.eigh(self.L)
        w_max = np.max(w)

        self.scale = scale
        self.kernels = kernels

        # N1 x M x N1
        self.atoms = np.array(
            [v @ np.diag(k(w)) @ v.T for k in [self.scale] + self.kernels]
        ).transpose([2, 0, 1])

        # N1 x M*N1
        self.atoms_flat = self.atoms.reshape(-1, self.N1).T

    def __str__(self):
        return f"Joint Laplacelet class:\nN1={self.N1}"

    def theoretical_frame(self):
        w = np.linalg.eigvalsh(self.L)
        S = self.scale(w) ** 2
        G = S + np.sum(np.array([k(w) ** 2 for k in self.kernels]), axis=0)

        return np.min(G), np.max(G)

    def transform(self, x):
        return np.einsum("ijk,k...->ij...", self.atoms, x)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Classes for multi-dimensional chains #
# and multi-dimensional transforms     #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


class Chain:
    def __init__(self, xs):
        # reshape
        ys = []
        for x in xs:
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            ys.append(x)

        self.K = len(ys)
        self.xs = ys

        # final reshape
        self.flatten()

        self.Ns = [x.shape[0] for x in self.xs]
        self.Fs = [x.shape[1:] for x in self.xs]

    def __str__(self):
        return f"Chain: K={self.K} Ns={self.Ns} Fs={self.Fs}"

    def flatten(self):
        for idx, x in enumerate(self.xs):
            self.xs[idx] = x.reshape(x.shape[0], -1, x.shape[-1])

    def __add__(self, C):
        return Chain([x0 + x1 for x0, x1 in zip(self.xs, C.xs)])

    def __sub__(self, C):
        return Chain([x0 - x1 for x0, x1 in zip(self.xs, C.xs)])

    def __mul__(self, c):
        return Chain([c * x for x in self.xs])

    def __rmul__(self, c):
        return Chain([c * x for x in self.xs])

    def __or__(self, C):
        return Chain(
            [np.concatenate([x0, x1], axis=1) for x0, x1 in zip(self.xs, C.xs)]
        )

    def __neg__(self):
        return Chain([-x for x in self.xs])

    def apply(self, f):
        return Chain([f(x) for x in self.xs])


class SimplicianSlepians_old(Hodgelet):
    def __init__(self, B1, B2, S=None, F=None):
        super().__init__(B1, B2)
        self.eigenvals, self.U = np.linalg.eigh(self.L)
        self.idx = self.eigenvals.argsort()[::-1]
        self.eigenvals = self.eigenvals[self.idx]
        self.U = self.U[:, self.idx]

        # If Data Agnostic, Based only on Clustering
        if type(S) == list and type(F) == list:
            # Solenoidal part of the Dictionary
            self.Sigma_sol = np.diag(F[0][:, 0])
            self.available_clusters_sol = np.unique(S[0])
            self.atoms_sol = []
            for cluster in self.available_clusters_sol:
                tmp = S[0] == cluster
                self.D = np.diag(tmp[:, 0] * 1)
                self.B = self.U @ self.Sigma_sol @ self.U.T
                self.X = self.B @ self.D @ self.B
                self.X = (self.X + self.X.T) / 2
                self.eigenvals, self.atoms_tmp = np.linalg.eigh(self.X)
                self.idx = self.eigenvals.argsort()[::-1]
                self.eigenvals = self.eigenvals[self.idx]
                self.atoms_tmp = self.atoms_tmp[:, self.idx]
                self.select_in_B_image = np.round(self.U.T @ self.atoms_tmp, 8)
                self.select_in_B_image[F[0][:, 0] == 1, :] = 0
                self.select_in_B_image = np.sum(self.select_in_B_image, 0)
                self.atoms_sol.append(self.atoms_tmp[:, self.select_in_B_image == 0])
            self.atoms_sol_np = np.concatenate(self.atoms_sol, axis=1)
            # Irrotational part of the Dictionary
            self.Sigma_irr = np.diag(F[1][:, 0])
            self.available_clusters_irr = np.unique(S[1])
            self.atoms_irr = []
            for cluster in self.available_clusters_irr:
                tmp = S[1] == cluster
                self.D = np.diag(tmp[:, 0] * 1)
                self.B = self.U @ self.Sigma_irr @ self.U.T
                self.X = self.B @ self.D @ self.B
                self.X = (self.X + self.X.T) / 2
                self.eigenvals, self.atoms_tmp = np.linalg.eigh(self.X)
                self.idx = self.eigenvals.argsort()[::-1]
                self.eigenvals = self.eigenvals[self.idx]
                self.atoms_tmp = self.atoms_tmp[:, self.idx]
                self.select_in_B_image = np.round(self.U.T @ self.atoms_tmp, 8)
                self.select_in_B_image[F[1][:, 0] == 1, :] = 0
                self.select_in_B_image = np.sum(self.select_in_B_image, 0)
                self.atoms_irr.append(self.atoms_tmp[:, self.select_in_B_image == 0])
            self.atoms_irr_np = np.concatenate(self.atoms_irr, axis=1)

            # Overcomplete Dictionary
            self.atoms_flat = np.concatenate(
                (self.atoms_sol_np, self.atoms_irr_np), axis=1
            )

        # If Data Driven, Based on Signal and Hodge
        if type(S) != list and type(F) == list:
            # Solenoidal part of the Dictionary
            self.Sigma_sol = np.diag(F[0][:, 0])
            self.D = np.diag(S)
            self.B = self.U @ self.Sigma_sol @ self.U.T
            self.X = self.B @ self.D @ self.B
            self.X = (self.X + self.X.T) / 2
            self.eigenvals, self.atoms_tmp = np.linalg.eigh(self.X)
            self.idx = self.eigenvals.argsort()[::-1]
            self.eigenvals = self.eigenvals[self.idx]
            self.atoms_tmp = self.atoms_tmp[:, self.idx]
            self.select_in_B_image = np.round(self.U.T @ self.atoms_tmp, 8)
            self.select_in_B_image[F[0][:, 0] == 1, :] = 0
            self.select_in_B_image = np.sum(self.select_in_B_image, 0)
            self.atoms_sol = self.atoms_tmp[:, self.select_in_B_image == 0]

            # Irrotational part of the Dictionary
            self.Sigma_irr = np.diag(F[1][:, 0])
            self.B = self.U @ self.Sigma_irr @ self.U.T
            self.X = self.B @ self.D @ self.B
            self.X = (self.X + self.X.T) / 2
            self.eigenvals, self.atoms_tmp = np.linalg.eigh(self.X)
            self.idx = self.eigenvals.argsort()[::-1]
            self.eigenvals = self.eigenvals[self.idx]
            self.atoms_tmp = self.atoms_tmp[:, self.idx]
            self.select_in_B_image = np.round(self.U.T @ self.atoms_tmp, 8)
            self.select_in_B_image[F[1][:, 0] == 1, :] = 0
            self.select_in_B_image = np.sum(self.select_in_B_image, 0)
            self.atoms_irr = self.atoms_tmp[:, self.select_in_B_image == 0]

            # Overcomplete Dictionary
            self.atoms_flat = np.concatenate((self.atoms_sol, self.atoms_irr), axis=1)
            self.atoms_flat[np.abs(self.atoms_flat) < 1e-8] = 0
            self.atoms_flat = np.round(self.atoms_flat, 8)
            self.atoms_flat = self.atoms_flat / np.linalg.norm(self.atoms_flat)

        # If User Hand-Set
        if type(S) != list and type(F) != list:
            self.Sigma = np.diag(F)
            self.D = np.diag(S)
            self.B = self.U @ self.Sigma @ self.U.T
            self.X = self.B @ self.D @ self.B
            self.X = (self.X + self.X.T) / 2
            self.eigenvals, self.atoms_tmp = np.linalg.eigh(self.X)
            self.idx = self.eigenvals.argsort()[::-1]
            self.eigenvals = self.eigenvals[self.idx]
            self.atoms_tmp = self.atoms_tmp[:, self.idx]
            self.select_in_B_image = np.round(self.U.T @ self.atoms_tmp, 8)
            self.select_in_B_image[F == 1, :] = 0
            self.select_in_B_image = np.sum(self.select_in_B_image, 0)
            self.atoms_flat = self.atoms_tmp[:, self.select_in_B_image == 0]

    def __str__(self):
        return f"Simplicial Slepians class:\nN0={self.N0}\nN1={self.N1}\nN2={self.N2}\nFrequencySet={self.F}\nEdgeSet={self.V}"

    def theoretical_frame(self):
        return 1, 1

    def transform(self, x):
        # return self.atoms_flat.T@x
        return np.einsum("ijk,k...->ij...", self.atoms, x)

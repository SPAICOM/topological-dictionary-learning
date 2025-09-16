# Thanks to C. Battiloro, P. Di Lorenzo, S. Barbarossa for sharing the code from their paper:
# "Topological Slepians: Maximally localized representations of signals over simplicial complexes", C. Battiloro, P. Di Lorenzo, S. Barbarossa

import gudhi
import types
import numpy as np
import networkx as nx
from scipy.linalg import svdvals
from scipy.spatial import Delaunay
from scipy.interpolate import PchipInterpolator


def twoholes_SC(n, coords):
    # sort nodes to be ordered from bottom-left to top-right
    diagonal_coordinates = np.sum(coords, axis=1)  # y = -x + c, compute c
    diagonal_idxs = np.argsort(
        diagonal_coordinates
    )  # sort by c: origin comes first, upper-right comes last
    coords = coords[diagonal_idxs]  # apply sort to original coordinates

    tri = Delaunay(coords)

    valid_idxs = np.where(
        (np.linalg.norm(coords - [1 / 4, 3 / 4], axis=1) > 1 / 8)
        & (np.linalg.norm(coords - [3 / 4, 1 / 4], axis=1) > 1 / 8)
    )[0]

    faces = np.array(
        sorted([sorted(t) for t in tri.simplices if np.in1d(t, valid_idxs).all()])
    )

    # SC matrix construction
    G = nx.OrderedDiGraph()
    G.add_nodes_from(valid_idxs)  # add nodes that are excluded to keep indexing easy
    E = []
    for f in faces:
        [a, b, c] = sorted(f)
        E.append((a, b))
        E.append((b, c))
        E.append((a, c))

    V = np.array(G.nodes)
    E = np.array(sorted(set(E)))

    for e in E:
        G.add_edge(*e)

    edge_to_idx = {tuple(E[i]): i for i in range(len(E))}

    return G, V, E, faces, edge_to_idx, coords, valid_idxs


def onehole_SC(n):
    coords = np.random.rand(n, 2)

    # sort nodes to be ordered from bottom-left to top-right
    diagonal_coordinates = np.sum(coords, axis=1)  # y = -x + c, compute c
    diagonal_idxs = np.argsort(
        diagonal_coordinates
    )  # sort by c: origin comes first, upper-right comes last
    coords = coords[diagonal_idxs]  # apply sort to original coordinates

    tri = Delaunay(coords)

    valid_idxs = np.where(np.linalg.norm(coords - [1 / 2, 1 / 2], axis=1) > 1 / 4)[0]

    faces = np.array(
        sorted([sorted(t) for t in tri.simplices if np.in1d(t, valid_idxs).all()])
    )

    # SC matrix construction
    G = nx.OrderedDiGraph()
    # G.add_nodes_from(np.arange(n)) # add nodes that are excluded to keep indexing easy
    G.add_nodes_from(valid_idxs)
    E = []
    for f in faces:
        [a, b, c] = sorted(f)
        E.append((a, b))
        E.append((b, c))
        E.append((a, c))

    V = np.array(G.nodes)
    E = np.array(sorted(set(E)))

    for e in E:
        G.add_edge(*e)

    edge_to_idx = {tuple(E[i]): i for i in range(len(E))}

    return G, V, E, faces, edge_to_idx, coords, valid_idxs


def gaussian_VR(n, d, r):
    coords = np.random.randn(n, d)
    rips = gudhi.RipsComplex(points=coords, max_edge_length=r)
    tree = rips.create_simplex_tree(max_dimension=2)
    simplices = [s[0] for s in tree.get_skeleton(2)]

    vertices = [s[0] for s in simplices if len(s) == 1]
    edges = [tuple(sorted(s)) for s in simplices if len(s) == 2]
    triangles = [sorted(s) for s in simplices if len(s) == 3]

    G = nx.OrderedDiGraph()
    G.add_nodes_from(vertices)
    G.add_edges_from(edges)
    vertices_cc = list(max(nx.connected_components(nx.to_undirected(G)), key=len))
    G_cc = G.subgraph(vertices_cc).copy()
    edges_cc = list(G_cc.edges)
    triangles_cc = np.array([f for f in triangles if np.in1d(f, vertices_cc).all()])

    edge_to_idx = {edges_cc[i]: i for i in range(len(edges_cc))}

    coords_cc = coords[vertices_cc]

    coords -= np.min(coords_cc, axis=0, keepdims=True)
    coords_cc -= np.min(coords_cc, axis=0, keepdims=True)
    coords /= np.max(coords_cc, axis=0, keepdims=True)

    return G_cc, vertices_cc, edges_cc, triangles_cc, edge_to_idx, coords


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Laplacian and incidence construction #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def incidence_matrices(G, V, E, faces, edge_to_idx):
    B1 = np.array(
        nx.incidence_matrix(G, nodelist=V, edgelist=E, oriented=True).todense()
    )
    B2 = np.zeros([len(E), len(faces)])

    for f_idx, face in enumerate(faces):  # face is sorted
        edges = [face[:-1], face[1:], [face[0], face[2]]]
        e_idxs = [edge_to_idx[tuple(e)] for e in edges]

        B2[e_idxs[:-1], f_idx] = 1
        B2[e_idxs[-1], f_idx] = -1
    return B1, B2


def scale_incidence_matrices(B1, B2):
    s1 = svdvals(B1)
    s2 = svdvals(B2)
    smax = np.maximum(s1[0], s2[0])

    return B1 * smax / s1[0], B2 * smax / s2[0]


def hodge_laplacians(B1, B2):
    L_lower = L_upper = None

    if B1 is not None:
        L_lower = B1.T @ B1
    if B2 is not None:
        L_upper = B2 @ B2.T
    if B1 is None and B2 is None:
        raise ValueError("B1 and B2 can not both be None")

    if L_lower is not None and L_upper is not None:
        L = L_lower + L_upper
    elif L_lower is not None:
        L = L_lower
    elif L_upper is not None:
        L = L_upper

    return L, L_upper, L_lower


# ~~~~~~~~~~~~~~~~~~#
# Kernel functions #
# ~~~~~~~~~~~~~~~~~~#


def hann_gen(R, M, gamma):
    """helper function, returns a Hann window function"""

    def hann(x):
        x = x * (x < 0) * (x > (-R * gamma / (M + 1 - R)))
        return 0.5 + 0.5 * np.cos(2 * np.pi * (0.5 + x * (M + 1 - R) / (R * gamma)))

    return hann


def shifted_scaled_hann_gen(R, M, m, scale=lambda x: x, gamma=1.0):
    hann = hann_gen(R, M, gamma)

    def shifted_hann(x):
        return hann(scale(x) - m * gamma / (M + 1 - R))

    return shifted_hann


def log_shifted_hann_gen(R, M, m, gamma=1.0, eps=1e-8):
    """helper function, returns a shifted and log-scaled Hann window function"""
    return shifted_scaled_hann_gen(R, M, m, lambda x: np.log(x + eps), gamma)


def log_wavelet_kernels_gen(R, M, gamma, eps=1e-8):
    """returns a scaling kernel and a list of wavelet kernels"""
    kernels = [log_shifted_hann_gen(R, M, m + 2, gamma) for m in range(M - 1)]

    def scale(x):
        inside = 3 * R / 8 - np.sum(np.array([y(x) ** 2 for y in kernels]), axis=0)
        return np.sqrt(np.maximum(inside, 0.0) + eps)  # clamp then eps

    return scale, kernels


def adaptive_wavelet_kernels_gen(R, M, w, eps=1e-8, Q=None):
    """returns a scaling kernels and a list of wavelet kernels"""
    ker_idx = np.where(w < eps)
    w_c = w[np.max(ker_idx) :]
    w_c = np.sort(w_c)
    w_c = w_c + np.linspace(0, eps, len(w_c))

    w_max = np.max(w_c)

    if Q is not None:
        qs = np.linspace(0, w_max, Q + 1)
        cdf = np.array([np.sum(w_c < q) for q in qs])
        cdf[0] = 0
        cdf[-1] = len(w_c)
        cdf = cdf / len(w_c)
        scale = PchipInterpolator(qs, cdf)
    else:
        cdf = np.linspace(0, 1, len(w_c))
        scale = PchipInterpolator(w_c, cdf)

    kernels = [
        shifted_scaled_hann_gen(R, M, m + 2, scale, gamma=1.0) for m in range(M - 1)
    ]

    scaling = lambda x: np.sqrt(
        3 * R / 8 - np.sum(np.array([y(x) ** 2 for y in kernels]), axis=0) + eps
    )

    return scaling, kernels


def simple_wavelet_kernels_gen(w):
    w_max = np.max(w)
    scale = lambda x: np.sqrt(1 - (x / w_max) ** 2 + 1e-6)
    kernels = [lambda x: x / w_max]
    return scale, kernels


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Helpers for atom creation #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def create_atoms(L, kernels):
    w, v = np.linalg.eigh(L)
    atoms = np.array([v @ np.diag(k(w)) @ v.T for k in kernels]).transpose([2, 0, 1])
    return atoms


def lifted_atoms(bd, scale, kernels, eps=1e-8):
    L = bd.T @ bd
    L_lifted = bd @ bd.T

    w, v = np.linalg.eigh(L)
    w_lifted, v_lifted = np.linalg.eigh(L_lifted)
    w_thresh = w_lifted.copy()
    w_thresh[np.where(w_thresh < eps)] = 1

    scaling = np.array([v @ np.diag(scale(w)) @ v.T]).transpose([2, 0, 1])
    wavelets = np.array(
        [
            bd.T @ v_lifted @ np.diag(k(w_lifted) / w_thresh) @ v_lifted.T
            for k in kernels
        ]
    ).transpose([2, 0, 1])

    return scaling, wavelets


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Helpers for slepians creation #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def get_frequency_mask(B1, B2):
    """A simple function for getting the irrotational and solenoidal frequencies"""
    L, Lup, Ldown = hodge_laplacians(B1, B2)
    E = L.shape[0]
    F_irr = np.zeros((E, 1))
    F_sol = np.zeros((E, 1))

    Lambdaup, Uup = np.linalg.eigh(Lup)
    Lambdaup[np.abs(Lambdaup) < 1e-8] = 0
    Lambdaup = np.round(Lambdaup, 8)

    Lambdadown, Udown = np.linalg.eigh(Ldown)
    Lambdadown[np.abs(Lambdadown) < 1e-8] = 0
    Lambdadown = np.round(Lambdadown, 8)

    d, _ = np.linalg.eigh(L)
    d[np.abs(d) < 1e-8] = 0
    d = np.round(d, 8)
    idx = d.argsort()[::-1]
    d = d[idx]

    for i in range(len(d)):
        if d[i] in Lambdaup and d[i] > 0:  # Solenoidal Frequencies
            F_sol[i] = 1

        if d[i] in Lambdadown and d[i] > 0:  # Irrotational Frequencies
            F_irr[i] = 1
        if d[i] in Lambdaup and d[i] in Lambdadown and d[i] > 0:
            F_sol[i] = 1
            F_irr[i] = 1
    return F_sol, F_irr


def cluster_on_neigh(
    B1, B2, Kup, Kdown, source_sol, source_irr, option="One-shot-diffusion", step_prog=1
):
    """
    A function for getting the edge concentration set

    Inputs:
        B1,B2: incidence matrices
        Kup,Kdown: order of the upper and lower neighbourhoods to be computed, respectively
        source_sol,source_irr: binary vectors, if the i-th component is equal to 1, then the (Kup,Kdown)-neighborhoods
        of the i-th edge will be used to compute (upper,lower) edge concentration sets
        option: if "One-shot-diffusion", per each of the chosen source edge (the 1s in source_sol an source_irr),
        the (Kup,Kdown)-neighborhoods are computed and included as edge concentration sets;
        if "Progressive-diffusion", per each of the chosen source edge (the 1s in source_sol an source_irr),
        the k-neighborhoods UP TO (Kup,Kdown)-neighborhoods  are computed and included as edge concentration sets at steps of lenght step_prog
        (e.g. Kup = 4, step_prog = 2 -> 2 and 4 neighborhoods are computed);
    """
    L, Lup, Ldown = hodge_laplacians(B1, B2)
    if option == "One-shot-diffusion":
        LKup = ((np.linalg.matrix_power(Lup, Kup) != 0) * 1)[
            source_sol == 1, :
        ].tolist()
        LKdown = ((np.linalg.matrix_power(Ldown, Kdown) != 0) * 1)[
            source_irr == 1, :
        ].tolist()
    elif option == "Progressive-diffusion":
        LKup = []
        for k in range(1, Kup + 1, step_prog):
            LKup_tmp = ((np.linalg.matrix_power(Lup, k) != 0) * 1)[
                source_sol == 1, :
            ].tolist()
            LKup = LKup + LKup_tmp
        LKdown = []
        for k in range(1, Kdown + 1, step_prog):
            LKdown_tmp = ((np.linalg.matrix_power(Ldown, k) != 0) * 1)[
                source_irr == 1, :
            ].tolist()
            LKdown = LKdown + LKdown_tmp
    else:
        print("No valid option!")

    # Checks
    E = B2.shape[0]
    sol_edge_cov = np.sum(np.array(LKup), axis=0)
    irr_edge_cov = np.sum(np.array(LKdown), axis=0)

    complete_coverage = 1
    sol_coverage = len(sol_edge_cov.nonzero()[0])
    irr_coverage = len(irr_edge_cov.nonzero()[0])
    # print("Sol. Covered: " + str(sol_coverage))
    if sol_coverage < E:
        complete_coverage = 0
    if irr_coverage < E:
        complete_coverage = 0
    return [
        list(set(tuple(x) for x in LKup)),
        list(set(tuple(x) for x in LKdown)),
    ], complete_coverage


# ~~~~~~~~~~~~~~~~~~~#
# Quasi-incoherence #
# ~~~~~~~~~~~~~~~~~~~#


def gram_matrix(W):
    """helper function to compute Gram matrix from matrix of atoms where each column is an atoms"""
    norms = np.linalg.norm(W, axis=0, keepdims=True)
    pos_idx = np.where(norms[0] > 0)[0]
    W_norm = W[:, pos_idx] / norms[:, pos_idx]
    G = W_norm.T @ W_norm
    return np.abs(G)


def sort_rows(G):
    """helper function to sort each row of the argument"""
    return np.sort(G, axis=1)[:, ::-1]


def IHT(x, A, AT, m, M, thresh=1e-15):
    """
    Accelerated iterative Hard Thresholding algorithm that keeps exactly M elements each iteration.
    This algorithm includes an additional double overrelaxation step that significantly improves convergence
    speed without destroying any of the theoretical guarantees of the IHT algorithm derived in [1], [2], and [3].
    This algorithm is used to solve the problem A*z = x.

    Inputs:
        x: observation vector to be decomposed
        A: it can be a (n x m) matrix that gives the effect of the forward matrix A on a vector or an operator that does the same
        AT: it can be a (n x m) matrix that gives the effect of the backward matrix A.T on a vector or an operator that does the same
        m: length of the solution vector s
        M: number of non-zero elements to keep in each iteration
        thresh: stopping criterion

    Outputs:
        s: solution vector
        err_mse: vector containing MSE of approximation error for each iteration
    """

    try:
        n1, n2 = x.shape
    except:
        x = np.expand_dims(x, axis=1)
        n1, n2 = x.shape
    if n2 == 1:
        n = n1
    elif n1 == 1:
        x = x.T
        n = n2
    else:
        exit("x must be a vector")

    sigsize = np.dot(x.T, x) / n
    oldERR = sigsize
    err_mse = []
    iter_time = []
    STOPTOL = 1e-16
    MAXITER = n**2
    verbose = True
    initial_given = 0
    s_initial = np.zeros((m, 1))
    MU = 0
    acceleration = 0
    Count = 0

    if isinstance(A, types.FunctionType):
        P = lambda z: A(z)
        PT = lambda z: AT(z)
    else:
        P = lambda z: np.dot(A, z)
        PT = lambda z: np.dot(AT, z)

    s_initial = np.zeros((m, 1))
    Residual = x
    s_old = np.copy(s_initial)
    s = np.copy(s_initial)
    Ps = np.zeros((n, 1))
    old_Ps = np.zeros((n, 1))
    oldErr = sigsize

    x_test = np.random.randn(m, 1)
    x_test = x_test / np.linalg.norm(x_test)
    nP = np.linalg.norm(P(x_test))
    if np.abs(MU * nP) > 1:
        exit(
            "WARNING! Algorithm likely to become unstable. Use smaller step-size or || P ||_2 < 1."
        )

    # Main algorithm
    t = 0
    done = False
    iteration = 1
    min_mu = 1e5
    max_mu = 0

    while not done:
        Count += 1
        if MU == 0:
            # Calculate optimal step size and do line search
            if (Count > 1) & (acceleration == 0):
                s_very_old = s_old
            s_old = s
            IND = s != 0
            d = PT(Residual)

            # If the current vector is zero, we take the largest element in d
            if np.sum(IND) == 0:
                sortind = np.argsort(np.abs(d), axis=0)[::-1]
                IND[sortind[0:M]] = 1

            id = IND * d
            Pd = P(id)
            mu = np.dot(id.T, id) / np.dot(Pd.T, Pd)
            max_mu = np.max([mu, max_mu])
            min_mu = np.min([mu, min_mu])
            mu = min_mu
            s = s_old + mu * d
            sortind = np.argsort(np.abs(s), axis=0)[::-1]
            s[sortind[M:]] = 0

            if (Count > 1) & (acceleration == 0):
                very_old_Ps = old_Ps
            old_Ps = Ps
            Ps = P(s)
            Residual = x - Ps

            if (Count > 2) & (acceleration == 0):
                # First overrelaxation
                Dif = Ps - old_Ps
                a1 = np.dot(Dif.T, Residual) / np.dot(Dif.T, Dif)
                z1 = s + a1 * (s - s_old)
                Pz1 = (1 + a1) * Ps - a1 * old_Ps
                Residual_z1 = x - Pz1

                # Second overrelaxation
                Dif = Pz1 - very_old_Ps
                a2 = np.dot(Dif.T, Residual_z1) / np.dot(Dif.T, Dif)
                z2 = z1 + a2 * (z1 - s_very_old)

                # Threshold z2
                sortind = np.argsort(np.abs(z2), axis=0)[::-1]
                z2[sortind[M:]] = 0
                Pz2 = P(z2)
                Residual_z2 = x - Pz2

                # Decide if z2 is any good
                if (
                    np.dot(Residual_z2.T, Residual_z2) / np.dot(Residual.T, Residual)
                    < 1
                ):
                    s = z2
                    Residual = Residual_z2
                    Ps = Pz2

            # if (acceleration > 0):
            # s, Residual = mySubsetCG(x, s, P, Pt

            # Calculate step-size requirements
            omega = (np.linalg.norm(s - s_old) / np.linalg.norm(Ps - old_Ps)) ** 2

            # As long as the support changes and mu > omega, we decrease mu
            while (
                (mu > 1.5 * omega)
                & (np.sum(np.logical_xor(IND, s != 0)) != 0)
                & (np.sum(IND) != 0)
            ):
                print("Decreasing mu")

                # We use a simple line search, halving mu in each step
                mu = mu / 2
                s = s_old + mu * d
                sortind = np.argsort(np.abs(s), axis=0)[::-1]
                s[sortind[M:]] = 0
                Ps = P(s)

                # Calculate optimal step size and do line search
                Residual = x - Ps
                if (Count > 2) & (acceleration == 0):
                    # First overrelaxation
                    Dif = Ps - old_Ps
                    a1 = np.dot(Dif.T, Residual) / np.dot(Dif.T, Dif)
                    z1 = s + a1 * (s - s_old)
                    Pz1 = (1 + a1) * Ps - a1 * old_Ps
                    Residual_z1 = x - Pz1

                    # Second overrelaxation
                    Dif = Pz1 - very_old_Ps
                    a2 = np.dot(Dif.T, Residual_z1) / np.dot(Dif.T, Dif)
                    z2 = z1 + a2 * (z1 - s_very_old)

                    # Threshold z2
                    sortind = np.argsort(np.abs(z2), axis=0)[::-1]
                    z2[sortind[M:]] = 0
                    Pz2 = P(z2)
                    Residual_z2 = x - Pz2

                    # Decide if z2 is any good
                    if (
                        np.dot(Residual_z2.T, Residual_z2)
                        / np.dot(Residual.T, Residual)
                        < 1
                    ):
                        s = z2
                        Residual = Residual_z2
                        Ps = Pz2

                # Calculate step-size requirements
                omega = (np.linalg.norm(s - s_old) / np.linalg.norm(Ps - old_Ps)) ** 2

            ERR = np.dot(Residual.T, Residual) / n
            err_mse.append(ERR)

            # Are we done yet?
            gap = np.linalg.norm(s - s_old) ** 2 / m
            if gap < thresh:
                done = True

            if not done:
                iteration += 1
                oldERR = ERR
            if verbose:
                print("Iter={0} - gap={1} - target={2}").format(Count, gap, thresh)
    return [s, err_mse]

import os
import sys
import time
import pickle
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from functools import wraps

sys.path.append(str(Path(sys.path[0]).parent))

from src import EnhancedGraph, TopoSolver


def load_topology(topo_params: dict):
    """
    Initializes and loads all variables related to the graph topology.

    Parameters:
    ----------
    topo_params : dict
        Dictionary containing topology parameters such as number of nodes,
        probability of edges, subgraph size, and seed for random generation.

    prob_T : float
        Probability related to the triangle formation in the graph.

    Returns:
    -------
    dict
        A dictionary containing all topology-related variables.
    """

    # Unpack topology parameters
    n = topo_params["n"]
    p_edges = topo_params["p_edges"]
    sub_size = topo_params["sub_size"]
    prob_T = topo_params["p_triangles"]
    seed = topo_params["seed"]

    # Initialize the EnhancedGraph
    G = EnhancedGraph(n=n, p_edges=p_edges, p_triangles=prob_T, seed=seed)

    # Get incidence matrices
    B1 = G.get_b1()
    # print("B1 shape:", B1.shape)
    B2 = G.get_b2()
    # print("B2 shape:", B2.shape)

    # Compute Laplacians
    Lu, Ld, L = G.get_laplacians(sub_size=sub_size)
    Lu_full = G.get_laplacians(sub_size=sub_size, full=True)

    # Sub-sampling
    B1, B2 = G.sub_size_skeleton(B1, B2, sub_size=sub_size)
    B2 = G.triangles_only(B2)
    B2_true = G.mask_B2(B2)

    # Calculate dimensions
    nu = B2.shape[1]
    nd = B1.shape[1]

    # Additional parameters
    M = L.shape[0]
    T = int(np.ceil(nu * (1 - prob_T)))

    # Compile all variables into a dictionary
    topology_data = {
        "G": G,
        "B1": B1,
        "B2": B2,
        "B2_true": B2_true,
        "nu": nu,
        "nd": nd,
        "Lu": Lu,
        "Ld": Ld,
        "L": L,
        "Lu_full": Lu_full,
        "M": M,
        "T": T,
    }

    return topology_data


def handle_diverged(res):
    def correct_values(in_tuple):
        res_tuple = []
        for i in range(2):
            non_diverging_mask = ~np.any(in_tuple[i] == None, axis=1)
            non_diverging_sim = in_tuple[i][non_diverging_mask]
            mean_error = np.mean(non_diverging_sim, axis=0)

            diverging_examples = in_tuple[i] == None
            matrix = np.where(diverging_examples, mean_error, in_tuple[i])
            res_tuple.append(matrix)

        return tuple(res_tuple)

    dict_types = ["edge", "joint", "separated", "complete"]

    for d in dict_types:
        res[d] = correct_values(res[d])

    return res


def polygons_indicator(B2):
    return np.where(np.sum(B2, axis=0) > 0, 1, 0)


def error_rate(v1, v2):
    return np.sum(v1 != v2) / len(v1)


def final_save(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_params = func.__code__.co_varnames
        save = kwargs["simulation_params"].get("save_results", True)
        res, models, rs = func(*args, **kwargs)
        if save:
            if "true_prob_T" in func_params:
                index = func_params.index("true_prob_T")
                if index < len(args):
                    prob_T = args[index]
                else:
                    prob_T = kwargs.get("true_prob_T", None)

                if prob_T is None:
                    raise ValueError(
                        "Parameter 'true_prob_T' not found in function arguments."
                    )
            elif "init_params" in func_params:
                index = func_params.index("init_params")
                if index < len(args):
                    prob_T = args[index].get("true_prob_T", None)
                    k0 = args[index].get("sparsity", 20)
                    J = args[index].get("J", 2)
                    # P = args[index].get("P", 3)
                else:
                    prob_T = kwargs["init_params"].get("true_prob_T", None)
                    k0 = kwargs["init_params"].get("sparsity", 20)
                    J = kwargs["init_params"].get("J", 2)
                    # P = kwargs["init_params"].get("P", 3)

            else:
                raise ValueError(
                    "Parameter 'true_prob_T' not defined in function signature."
                )
                return res, models, rs

            d = kwargs["simulation_params"].get("true_dictionary_type", "separated")
            mode = kwargs["simulation_params"].get("sparsity_mode", "max")
            res_dir = "results\\final"
            dir = f"max_sparsity{k0}" if mode == "max" else f"random_sparsity{k0}"
            path = str(Path(sys.path[0]).parent)
            dir_path = os.path.join(path, res_dir, dir)

            os.makedirs(dir_path, exist_ok=True)
            rcode = np.random.default_rng(None).integers(0, 10000)
            if func.__name__ == "dict_and_topology_learning":
                filename = f"res_{d}_T{int(prob_T * 100)}_{mode}{k0}_J{J}_{rcode}.pkl"
            elif func.__name__ == "param_dict_learning":
                filename = f"res_{d}_T{int(prob_T * 100)}_{rcode}.pkl"
            elif func.__name__ == "analyt_dict_learning":
                filename = f"res_{d}_T{int(prob_T * 100)}_analyt_{rcode}.pkl"
            else:
                return res, models
            file_path = os.path.join(dir_path, filename)
            with open(file_path, "wb") as file:
                pickle.dump(models, file)
                pickle.dump(res, file)

            return res, models, rs, file_path

    return wrapper


def prepare_error_dict(
    dict_list: list[str],
    n_sim: int,
    n_sparse_levels: int,
) -> tuple[dict, dict, dict]:
    dict_maps = {
        "classic_fourier": ("Fourier", "classic_fourier"),
        "fourier": ("Topological Fourier", "fourier"),
        "slepians": ("Topological Slepians", "slepians"),
        "wavelet": ("Hodgelet", "wavelet"),
        "edge": ("Edge Laplacian", "edge"),
        "joint": ("Hodge Laplacian", "joint"),
        "separated": ("Separated Hodge Laplacian", "separated"),
        "complete_greedy": (
            "Separated Hodge Laplacian with Greedy Topology learning",
            "separated",
        ),
        "complete_soft": (
            "Separated Hodge Laplacian with Topology learning",
            "separated",
        ),
    }

    assert all(d in dict_maps for d in dict_list), (
        "Unknown dictionary type in dict_list"
    )

    dict_errors = {
        d: (
            np.zeros((n_sim, n_sparse_levels)),
            np.zeros((n_sim, n_sparse_levels)),
            np.zeros((n_sim, n_sparse_levels)),
        )
        for d in dict_list
    }
    dict_types = {d: dict_maps[d] for d in dict_list}
    runtimes = {k: np.zeros((n_sim, n_sparse_levels)) for k in dict_types.keys()}

    return dict_errors, dict_types, runtimes


@final_save
def dict_and_topology_learning(
    n_sim: int,
    dict_list: list[str],
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    c_true: np.ndarray,
    epsilon_true: np.ndarray,
    Lu_true: np.ndarray,
    init_params: dict,
    algo_params: dict,
    simulation_params: dict,
    K0_coll: list[int],
    verbose: bool = True,
) -> tuple[dict, dict, dict]:
    """
    Learn the sparse representation and the dictionary atoms with the alternating-direction algorithm,
    for given test and training set, and comparing the performances in terms of training and test
    approximation error for several algorithmic setup:
    """
    dict_errors, dict_types, runtimes = prepare_error_dict(
        dict_list, n_sim, len(K0_coll)
    )
    models = {}

    for sim in tqdm(range(n_sim)):
        for k0_index, k0 in tqdm(enumerate(K0_coll), leave=False):
            models[eval(f"{sim},{k0_index}")] = []

            for d in dict_types.items():
                try:
                    model = TopoSolver(
                        X_train=X_train[:, :, sim],
                        X_test=X_test[:, :, sim],
                        Y_train=Y_train[:, :, sim],
                        Y_test=Y_test[:, :, sim],
                        c=c_true[sim],
                        epsilon=epsilon_true[sim],
                        K0=k0,
                        dictionary_type=d[1][1],
                        **init_params,
                    )
                except TypeError:
                    model = TopoSolver(
                        X_train=X_train,
                        X_test=X_test,
                        Y_train=Y_train,
                        Y_test=Y_test,
                        c=c_true,
                        epsilon=epsilon_true,
                        K0=k0,
                        dictionary_type=d[1][1],
                        **init_params,
                    )

                learn_topology = True if "complete" in d[0] else False
                soft = False if "greedy" in d[0] else True

                try:
                    t0 = time.perf_counter()
                    (
                        dict_errors[d[0]][0][sim, k0_index],
                        dict_errors[d[0]][1][sim, k0_index],
                        dict_errors[d[0]][2][sim, k0_index],
                    ) = model.fit(
                        Lu_true=Lu_true,
                        init_mode="only_X",
                        learn_topology=learn_topology,
                        soft=soft,
                        **algo_params,
                    )
                    runtimes[d[0]][sim, k0_index] = time.perf_counter() - t0
                    print(f"Runtime: {runtimes[d[0]][sim, k0_index]:.2f} seconds")
                    if learn_topology:
                        models[eval(f"{sim},{k0_index}")].append(model)

                    if verbose:
                        logging.info(
                            f"Tri: {init_params['true_prob_T']} Simulation: {sim + 1}/{n_sim} Sparsity: {k0} Testing {d[1][0]}... Done! Test Error: {dict_errors[d[0]][1][sim, k0_index]:.3f}"
                        )
                        logging.info(
                            f"Topology Approx. Error: {dict_errors[d[0]][2][sim, k0_index]:.6f}"
                        )

                except Exception as e:
                    logging.error(
                        f"Simulation: {sim + 1}/{n_sim} Sparsity: {k0} Testing {d[1][0]}... Diverged!"
                        + f"\nException: {e}"
                    )

                    (
                        dict_errors[d[0]][0][sim, k0_index],
                        dict_errors[d[0]][1][sim, k0_index],
                        dict_errors[d[0]][2][sim, k0_index],
                    ) = (
                        None,
                        None,
                        None,
                    )

    # dict_errors = handle_diverged(dict_errors)

    return dict_errors, models, runtimes


def find_common_arrays(data_dict):
    if not data_dict:
        return []

    # Convert each list of lists to a set of tuples
    sets_of_arrays = [set(map(tuple, arrays)) for arrays in data_dict.values()]

    # Find the intersection of all sets
    common = set.intersection(*sets_of_arrays)

    # Convert the tuples back to lists
    return [list(t) for t in common]

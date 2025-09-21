import os
import hydra
import pickle
import logging
import warnings
import scipy.io
import numpy as np
from tqdm import tqdm
import networkx as nx
from datetime import datetime
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf
from networkx.algorithms.cycles import find_cycle
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

from scripts.utils import find_common_arrays, dict_and_topology_learning
from scripts.visualize import plot_learnt_topology_real, nmse_curves

warnings.filterwarnings("ignore")


def load_real_data(data_path: str, seed: int = 0):
    mat = scipy.io.loadmat(data_path)
    edge_signals = np.array(mat["signal_edge"].T, dtype=float)
    valid_signal = np.where(np.sum(edge_signals, axis=1) != 0)
    edge_signals = edge_signals[valid_signal]
    Y_train, Y_test = train_test_split(edge_signals, test_size=0.2, random_state=seed)
    Y_train = Y_train.T
    Y_test = Y_test.T
    Lu = mat["B2"] @ mat["B2"].T
    Ld = mat["B1"].T @ mat["B1"]
    L = Lu + Ld
    Lu = np.array(Lu, dtype=float)
    Ld = np.array(Ld, dtype=float)
    L = np.array(L, dtype=float)
    path = str(Path(sys.path[0]).parent)
    with open(
        f"{path}\\data\\synthetic\\max_sparsity25\\full_data_separated.pkl",
        "rb",
    ) as file:
        data = pickle.load(file)
        c = data["c_true"].max()
        epsilon = data["epsilon_true"].mean()

    load_data = {
        "epsilon_true": epsilon,
        "Y_train": Y_train,
        "Y_test": Y_test,
        "X_train": None,
        "X_test": None,
        "c_true": c,
        "B1": mat["B1"],
        "B2": mat["B2"],
        "Lu": Lu,
        "Ld": Ld,
        "L": L,
    }
    return load_data


def setup_logger() -> None:
    log_dir = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs\\run_{current_time}.log"
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)


def log_configurations(cfg) -> None:
    logging.info("Configuration Parameters:")
    logging.info(OmegaConf.to_yaml(cfg))


@hydra.main(
    config_path="../.conf",
    config_name="real",
    version_base=None,
)
def main(cfg: DictConfig):
    # Load configurations
    dictionary_type = cfg.data.dictionary_type
    P = cfg.data.P
    J = cfg.data.J
    sparsity_mode = cfg.data.sparsity_mode
    K0_max = cfg.data.sparsity
    seed = cfg.data.seed

    print("Loading data...")
    data_path = os.path.join(
        str(Path(sys.path[0]).parent), "data", "real", "data_real.mat"
    )
    load_data = load_real_data(data_path, seed)
    print("Data loading complete!")

    epsilon_true = load_data["epsilon_true"]
    X_train = load_data["X_train"]
    X_test = load_data["X_test"]
    Y_train = load_data["Y_train"]
    Y_test = load_data["Y_test"]
    c_true = load_data["c_true"]
    Lu = load_data["Lu"]

    K0_coll = np.arange(
        cfg.algorithm.min_sparsity,
        cfg.algorithm.max_sparsity,
        cfg.algorithm.sparsity_freq,
    )

    log_configurations(cfg)

    simulation_params = {
        "sparsity": K0_max,
        "save_results": False,
        "sparsity_mode": sparsity_mode,
        "true_dictionary_type": dictionary_type,
    }

    init_params = {
        "J": J,
        "P": P,
        "seed": seed,
        "sparsity": K0_max,
        "true_prob_T": 1.0,
        "n": load_data["B1"].shape[0],
        "B1_true": load_data["B1"],
        "B2_true": load_data["B2"],
    }

    algo_params = {
        "lambda_": cfg.algorithm.lambda_,
        "lambda2": cfg.algorithm.lambda2,
        "tol": cfg.algorithm.tol,
        "tol_out": cfg.algorithm.tol_out,
        "patience": cfg.algorithm.patience,
        "max_iter": cfg.algorithm.max_iter,
        "max_iter_out": cfg.algorithm.max_iter_out,
        "max_iter_tot": cfg.algorithm.max_iter_tot,
        "QP": cfg.algorithm.QP,
        "mode": cfg.algorithm.topo_learning_mode,
        "warmup": cfg.algorithm.warmup,
        "on_test": cfg.algorithm.on_test,
        "both": cfg.algorithm.both,
        "verbose": cfg.algorithm.verbose,
        "mu": cfg.algorithm.mu,
        "decouple_learning": cfg.algorithm.decouple_learning,
    }

    dict_types = OmegaConf.to_container(cfg.algorithm.dictionary_types, resolve=True)
    print("Starting the learning algorithm...")

    dict_err, mods, rs = dict_and_topology_learning(
        dict_list=dict_types,
        n_sim=1,
        X_train=X_train,
        X_test=X_test,
        Y_train=Y_train,
        Y_test=Y_test,
        c_true=c_true,
        epsilon_true=epsilon_true,
        simulation_params=simulation_params,
        init_params=init_params,
        algo_params=algo_params,
        K0_coll=K0_coll,
        Lu_true=Lu,
    )

    print("Learning process complete!")

    res_path = os.path.join(
        str(Path(sys.path[0]).parent),
        "results",
        "real",
        "data_real.mat",
        f"real_J{J}P{P}.pkl",
    )
    with open(f"{res_path}", "wb") as file:
        pickle.dump(mods, file)
        pickle.dump(dict_err, file)
        pickle.dump(rs, file)
    logging.info(f"Results saved in: {res_path}")

    print("Plotting time...")
    models_poly = defaultdict(list)
    coll2 = [2, 3, 4]
    for idx, i in enumerate(coll2):
        model = mods[0][f"{i},complete"]
        A = np.zeros((model.B1.shape[0], model.B1.shape[0]))

        for edge_index in range(model.B1.shape[1]):
            nodes = np.where(model.B1[:, edge_index] != 0)[0]
            A[nodes[0], nodes[1]] = 1

        G = nx.from_numpy_array(A)
        B2 = model.B2

        for polygon_index in range(B2.shape[1]):
            np.random.seed(polygon_index)
            polygon_edges = []
            edges = np.where(B2[:, polygon_index] != 0)[0]

            for edge_index in edges:
                nodes = np.array(np.where(model.B1[:, edge_index] != 0)[0], dtype=int)
                polygon_edges.append(tuple(nodes))

            subgraph = G.edge_subgraph(polygon_edges)
            try:
                cycle_edges = find_cycle(subgraph)
                models_poly[idx].append(cycle_edges)
            except Exception:
                pass

        common = find_common_arrays(models_poly)

        model1 = mods[0]["2,complete"]
        model2 = mods[0]["3,complete"]
        model3 = mods[0]["4,complete"]
        node_positions = {
            0: (3, 306),
            1: (297, 494),
            2: (336, 367),
            3: (459, 178),
            4: (151, 212),
            5: (267, 193),
            6: (169, 130),
            7: (156, 97),
            8: (427, 302),
            9: (159, 377),
            10: (90, 251),
            11: (476, 288),
            12: (50, 270),
            13: (300, 293),
            14: (64, 254),
            15: (204, 6),
            16: (162, 357),
            17: (103, 522),
            18: (222, 324),
            19: (159, 323),
            20: (461, 55),
            21: (234, 96),
            22: (221, 184),
            23: (103, 408),
            24: (142, 440),
            25: (208, 267),
            26: (262, 539),
            27: (240, 38),
            28: (91, 336),
            29: (52, 294),
            30: (193, 546),
            31: (395, 261),
            32: (332, 203),
            33: (149, 405),
            34: (339, 503),
            35: (97, 217),
            36: (79, 93),
            37: (302, 399),
            38: (135, 132),
            39: (124, 192),
            40: (454, 473),
            41: (370, 438),
            42: (59, 425),
            43: (315, 94),
            44: (123, 297),
            45: (184, 464),
            46: (37, 385),
            47: (242, 488),
            48: (25, 258),
            49: (239, 382),
        }
        node_positions = {
            i: (x / 500, (576 - y) / 500)
            for i, (x, y) in enumerate(list(node_positions.values()))
        }
        plot_learnt_topology_real(
            load_data["B1"],
            model1,
            model2,
            model3,
            node_positions,
            K0_coll[coll2],
            common,
        )
        nmse_curves(dict_err, K0_coll, real=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    setup_logger()
    main()

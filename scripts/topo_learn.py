import os
import sys
import hydra
import pickle
import logging
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

sys.path.append(str(Path(sys.path[0]).parent))

from src import generate_data
from scripts.utils import load_topology, dict_and_topology_learning


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
    config_name="training",
    version_base=None,
)
def main(cfg: DictConfig):
    # Load configurations
    dictionary_type = cfg.data.dictionary_type
    m_train = cfg.data.m_train
    m_test = cfg.data.m_test
    P = cfg.data.P
    J = cfg.data.J
    sparsity_mode = cfg.data.sparsity_mode
    K0_max = cfg.data.sparsity
    n_search = cfg.data.n_search
    n_sim = cfg.data.n_sim
    n = cfg.data.n
    prob_T = cfg.data.p_triangles
    p_edges = cfg.data.p_edges
    seed = cfg.data.seed
    sub_size = cfg.data.sub_size

    topo_params = {
        "p_edges": p_edges,
        "p_triangles": prob_T,
        "n": n,
        "seed": seed,
        "sub_size": sub_size,
    }

    # Load the topology using the dedicated function
    topology_data = load_topology(topo_params=topo_params)

    # True topology data
    Lu_true = topology_data["Lu"]
    Ld_true = topology_data["Ld"]
    M = topology_data["M"]

    # Generate data
    gen_params = {
        "dictionary_type": dictionary_type,
        "m_train": m_train,
        "m_test": m_test,
        "P": P,
        "M": M,
        "J": J,
        "K0_max": K0_max,
        "sparsity_mode": sparsity_mode,
        "n_search": n_search,
        "n_sim": n_sim,
        "prob_T": prob_T,
    }

    print("Generating data...")
    load_data = generate_data(Lu_true, Ld_true, **gen_params)
    print("Data generating process complete!")

    # D_true = load_data["D_true"]
    Y_train = load_data["Y_train"]
    Y_test = load_data["Y_test"]
    X_train = load_data["X_train"]
    X_test = load_data["X_test"]
    epsilon_true = load_data["epsilon_true"]
    c_true = load_data["c_true"]
    print(Y_train.shape)

    K0_coll = np.arange(
        cfg.algorithm.min_sparsity,
        cfg.algorithm.max_sparsity,
        cfg.algorithm.sparsity_freq,
    )

    log_configurations(cfg)

    simulation_params = {
        "sparsity_mode": sparsity_mode,
        "sparsity": K0_max,
        "true_dictionary_type": dictionary_type,
        "save_results": True,
    }

    init_params = {
        "J": J,
        "P": P,
        "true_prob_T": prob_T,
        "n": n,
        "p_edges": p_edges,
        "sub_size": sub_size,
        "seed": seed,
        "sparsity": K0_max,
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

    _, _, results_path = dict_and_topology_learning(
        dict_list=dict_types,
        n_sim=n_sim,
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
        Lu_true=Lu_true,
    )

    print("Learning process complete!")

    logging.info(f"Results saved in: {results_path}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    setup_logger()
    main()

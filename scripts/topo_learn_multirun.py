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
    logging.info("Algorithmic Parameters:")
    logging.info(OmegaConf.to_yaml(cfg.algorithm))


@hydra.main(
    config_path="../.conf",
    config_name="multirun",
    version_base="1.3",
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
    p_edges = cfg.data.p_edges
    seed = cfg.data.seed
    sub_size = cfg.data.sub_size

    topo_params = {
        "p_edges": p_edges,
        "n": n,
        "seed": seed,
        "sub_size": sub_size,
    }

    gen_params = {
        "dictionary_type": dictionary_type,
        "m_train": m_train,
        "m_test": m_test,
        "P": P,
        "J": J,
        "K0_max": K0_max,
        "sparsity_mode": sparsity_mode,
        "n_search": n_search,
        "n_sim": n_sim,
    }

    K0_coll = np.array([cfg.algorithm.algo_sparsity])
    log_configurations(cfg)

    simulation_params = {
        "sparsity_mode": sparsity_mode,
        "sparsity": K0_max,
        "true_dictionary_type": dictionary_type,
        "save_results": False,
    }

    init_params = {
        "J": J,
        "P": P,
        "n": n,
        "p_edges": p_edges,
        "sub_size": sub_size,
        "seed": seed,
        "sparsity": K0_max,
    }

    algo_params = {
        "lambda2": cfg.algorithm.lambda2,
        "tol": cfg.algorithm.tol,
        "tol_out": cfg.algorithm.tol_out,
        "patience": cfg.algorithm.patience,
        "max_iter_out": cfg.algorithm.max_iter_out,
        "max_iter_tot": cfg.algorithm.max_iter_tot,
        "QP": cfg.algorithm.QP,
        "mode": cfg.algorithm.topo_learning_mode,
        "warmup": cfg.algorithm.warmup,
        "on_test": cfg.algorithm.on_test,
        "both": cfg.algorithm.both,
        "verbose": cfg.algorithm.verbose,
        "decouple_learning": cfg.algorithm.decouple_learning,
    }

    dict_errors = []
    models = []
    run_times = []

    p_tr = cfg.algorithm.tr_list
    mu_schedule = cfg.algorithm.mu_schedule
    lambda_schedule = cfg.algorithm.lambda_schedule
    mi_schedule = cfg.algorithm.mi_schedule
    algo = cfg.algorithm.algo

    for i, p in enumerate(p_tr):
        # Load the topology
        topo_params["p_triangles"] = p
        topology_data = load_topology(topo_params=topo_params)

        # Topological data
        Lu_true = topology_data["Lu"]
        Ld_true = topology_data["Ld"]
        M = topology_data["M"]
        gen_params["prob_T"] = p
        gen_params["M"] = M
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

        init_params["true_prob_T"] = p
        algo_params["mu"] = mu_schedule[i]
        algo_params["lambda_"] = lambda_schedule[i]
        algo_params["max_iter"] = mi_schedule[i]

        print("Starting the learning algorithm...")
        dict_err, mods, rs = dict_and_topology_learning(
            dict_list=["complete_" + algo],
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
        dict_errors.append(dict_err)
        models.append(mods)
        run_times.append(rs)

    print("Learning process complete!")

    p_tr_name = "".join([f"{int(p * 100)}" for p in p_tr])
    results_path = f"results/final/{sparsity_mode}_sparsity{K0_max}/tri_learnS{cfg.algorithm.algo_sparsity}{cfg.algorithm.algo}{p_tr_name}.pkl"
    with open(results_path, "wb") as file:
        pickle.dump(models, file)
        pickle.dump(dict_errors, file)
        pickle.dump(run_times, file)
    logging.info(f"Results saved in: {results_path}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    setup_logger()
    main()

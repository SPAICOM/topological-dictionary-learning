import sys
from pathlib import Path

path = str(Path.cwd())
sys.path.insert(0, path)

from datetime import datetime
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import argparse
import logging
from tqdm import tqdm
import numpy as np
import warnings
from utils import *
from topolearn import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(
    description="Run dictionary and topology learning with specified configuration."
)
parser.add_argument(
    "--config-dir", type=str, default="config", help="Configuration directory"
)
parser.add_argument(
    "--config-name", type=str, default="config.yaml", help="Configuration file name"
)
args = parser.parse_args()


# Setup Logger
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


def log_configurations(cfg, algo_cfg):
    logging.info("Configuration Parameters:")
    logging.info(OmegaConf.to_yaml(cfg))
    logging.info("Algorithmic Parameters:")
    logging.info(OmegaConf.to_yaml(algo_cfg))


@final_save
def param_dict_learning(
    X_train,
    X_test,
    Y_train,
    Y_test,
    c_true,
    epsilon_true,
    simulation_params,
    init_params,
    algo_params,
    K0_coll,
    Lu_true,
    verbose: bool = True,
):
    """
    Learn the sparse representation and the dictionary atoms with the alternating-direction algorithm,
    for given test and training set, and comparing the performances in terms of training and test
    approximation error for several algorithmic setup, principally differencing by dictionary parameterization:
        - Fourier dictionary parameterization
        - Edge Laplacian dictionary parameterization
        - Joint Hodge Laplacian dictionary parameterization
        - Separated Hodge Laplacian dictionary parameterization
        - (Optionally) Separated Hodge Laplacian dictionary parameterization plus Topology Learning
    """
    n_sim = X_train.shape[-1]

    min_error_fou_train = np.zeros((n_sim, len(K0_coll)))
    min_error_fou_test = np.zeros((n_sim, len(K0_coll)))
    min_error_sep_train = np.zeros((n_sim, len(K0_coll)))
    min_error_sep_test = np.zeros((n_sim, len(K0_coll)))
    min_error_edge_train = np.zeros((n_sim, len(K0_coll)))
    min_error_edge_test = np.zeros((n_sim, len(K0_coll)))
    min_error_joint_train = np.zeros((n_sim, len(K0_coll)))
    min_error_joint_test = np.zeros((n_sim, len(K0_coll)))
    approx_fou = np.zeros((n_sim, len(K0_coll)))
    approx_sep = np.zeros((n_sim, len(K0_coll)))
    approx_joint = np.zeros((n_sim, len(K0_coll)))
    approx_edge = np.zeros((n_sim, len(K0_coll)))
    approx_comp = np.zeros((n_sim, len(K0_coll)))

    dict_errors = {
        "fourier": (min_error_fou_train, min_error_fou_test, approx_fou),
        "edge": (min_error_edge_train, min_error_edge_test, approx_edge),
        "joint": (min_error_joint_train, min_error_joint_test, approx_joint),
        "separated": (min_error_sep_train, min_error_sep_test, approx_sep),
    }

    dict_types = {
        "fourier": ("Fourier", "fourier"),
        "edge": ("Edge Laplacian", "edge"),
        "joint": ("Hodge Laplacian", "joint"),
        "separated": ("Separated Hodge Laplacian", "separated"),
    }

    if algo_params["complete"]:
        min_error_complete_train = np.zeros((n_sim, len(K0_coll)))
        min_error_complete_test = np.zeros((n_sim, len(K0_coll)))
        dict_errors["complete"] = (
            min_error_complete_train,
            min_error_complete_test,
            approx_comp,
        )
        dict_types["complete"] = (
            "Separated Hodge Laplacian with Topology learning",
            "separated",
        )

    models = {}

    for sim in tqdm(range(n_sim)):

        for k0_index, k0 in tqdm(enumerate(K0_coll), leave=False):

            for d in dict_types.items():

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

                learn_topology = True if d[0] == "complete" else False

                try:

                    (
                        dict_errors[d[0]][0][sim, k0_index],
                        dict_errors[d[0]][1][sim, k0_index],
                        dict_errors[d[0]][2][sim, k0_index],
                    ) = model.fit(
                        Lu_true=Lu_true,
                        init_mode="only_X",
                        learn_topology=learn_topology,
                        **algo_params,
                    )

                    models[f"{sim},{k0_index}"] = model

                    if verbose:
                        logging.info(
                            f"Simulation: {sim+1}/{n_sim} Sparsity: {k0} Testing {d[1][0]}... Done! Test Error: {dict_errors[d[0]][1][sim,k0_index]:.3f}"
                        )
                        logging.info(
                            f"Topology Approx. Error: {dict_errors[d[0]][2][sim,k0_index]:.3f}"
                        )

                except Exception as e:
                    logging.error(
                        f"Simulation: {sim+1}/{n_sim} Sparsity: {k0} Testing {d[1][0]}... Diverged!"
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

    return dict_errors, models


@hydra.main(
    config_path=args.config_dir, config_name=args.config_name, version_base=None
)
def main(cfg: DictConfig):

    # Load configurations
    dictionary_type = cfg.dictionary_type
    m_train = cfg.m_train
    m_test = cfg.m_test
    P = cfg.P
    J = cfg.J
    sparsity_mode = cfg.sparsity_mode
    K0_max = cfg.sparsity
    n_search = cfg.n_search
    n_sim = cfg.n_sim
    n = cfg.n
    prob_T = cfg.p_triangles
    p_edges = cfg.p_edges
    seed = cfg.seed
    sub_size = cfg.sub_size

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
    L_true = topology_data["L"]
    Lu_true = topology_data["Lu"]
    Ld_true = topology_data["Ld"]
    # B2_true = topology_data["B2_true"]
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

    # Load algorithmic configurations
    algo_cfg = hydra.compose(config_name="algorithm.yaml")
    K0_coll = np.arange(
        algo_cfg.min_sparsity, algo_cfg.max_sparsity, algo_cfg.sparsity_freq
    )

    log_configurations(cfg, hydra.compose(config_name="algorithm.yaml"))

    simulation_params = {
        "sparsity_mode": sparsity_mode,
        "true_dictionary_type": dictionary_type,
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
        "lambda_": algo_cfg.lambda_,
        "tol": algo_cfg.tol,
        "patience": algo_cfg.patience,
        "max_iter": algo_cfg.max_iter,
        "QP": algo_cfg.QP,
        "mode": algo_cfg.topo_learning_mode,
        "warmup": algo_cfg.warmup,
        "on_test": algo_cfg.on_test,
        "complete": algo_cfg.complete,
        "verbose": algo_cfg.verbose,
    }

    print("Starting the learning algorithm...")

    _, _, results_path = param_dict_learning(
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

    # results_path = f"results/final/resT{int(prob_T*100)}.pkl"
    logging.info(f"Results saved in: {results_path}")


if __name__ == "__main__":
    main()

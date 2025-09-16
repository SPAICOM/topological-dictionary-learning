import re
import os
import glob
import hydra
import pickle
import numpy as np
import pandas as pd
from omegaconf import DictConfig

import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

from scripts.utils import (
    load_topology,
    polygons_indicator,
    error_rate,
)


def greedy_iters(p: np.ndarray) -> int:
    d = int(np.sum(p))
    n = p.shape[0]
    res = int(np.sum(np.array([n - i for i in range(1, d + 1)])))
    return res


def get_pickle_files(directory):
    pattern = os.path.join(directory, "tri_learn*.pkl")
    return glob.glob(pattern)


def extract_info(file):
    s_pre = file.split(".")[0]
    match = re.search(r"(greedy|soft)((?:\d{2})+)", s_pre)
    if match:
        algo_method = match.group(1)
        digits_str = match.group(2)
        # Split the digit string into 2-digit integers.
        tr_list = [int(digits_str[i : i + 2]) for i in range(0, len(digits_str), 2)]
    else:
        algo_method = None
        tr_list = []

    with open(file, "rb") as f:
        models = pickle.load(f)
        dict_errors = pickle.load(f)
        runtimes = pickle.load(f)
    return tr_list, algo_method, models, dict_errors, runtimes


@hydra.main(
    config_path="../config",
    config_name="table.yaml",
    version_base="1.3",
)
def organize_data(cfg: DictConfig) -> None:
    path = os.getcwd()
    df = pd.DataFrame()
    df_mse = pd.DataFrame()
    data_sparsity = cfg.data_sparsity
    sparsity_mode = cfg.sparsity_mode
    topo_params = {
        "p_edges": cfg.p_edges,
        "n": cfg.n,
        "seed": cfg.seed,
        "sub_size": cfg.sub_size,
    }
    name = ""
    for s in data_sparsity:
        name += str(s)
        directory = f"{path}\\results\\final\\{sparsity_mode}_sparsity{s}"
        files = get_pickle_files(directory)
        for file in files:
            print(file)
            tr_list, algo_method, models, dict_errors, runtimes = extract_info(file)
            for i, p in enumerate(tr_list):
                topo_params["p_triangles"] = p / 100
                topology_data = load_topology(topo_params=topo_params)
                p_star = polygons_indicator(topology_data["B2_true"])
                for sim in range(10):
                    try:
                        p_hat = polygons_indicator(models[i][(sim, 0)][0].B2)
                        gi = greedy_iters(p_hat) if algo_method == "greedy" else 0
                        df = pd.concat(
                            [
                                df,
                                pd.DataFrame(
                                    {
                                        "Error": dict_errors[i][
                                            f"complete_{algo_method}"
                                        ][2][sim],
                                        "Method": algo_method,
                                        "Sim": sim,
                                        "Sparsity": s,
                                        "Triangles": p,
                                        "Error2": error_rate(p_star, p_hat),
                                        "Runtime": runtimes[i][
                                            f"complete_{algo_method}"
                                        ][sim],
                                        "Greedy_iters": gi,
                                    }
                                ),
                            ]
                        )
                        a = []
                        for arr in models[i][(sim, 0)][0].train_error_hist:
                            a += arr
                        df_mse = pd.concat(
                            [
                                df_mse,
                                pd.DataFrame(
                                    {
                                        "MSE": a,
                                        "Method": algo_method,
                                        "Sim": sim,
                                        "Sparsity": s,
                                        "Triangles": p,
                                    }
                                ),
                            ]
                        )
                    except IndexError:
                        print(
                            f"No results for simulation {s} with: \nMethod: {algo_method} \nGT % of triangles: {p} \nSparsity{s}"
                        )

    with open(f"{path}\\results\\paper\\lu_errorS{name}.pkl", "wb") as ff:
        pickle.dump(df, ff)
        pickle.dump(df_mse, ff)


if __name__ == "__main__":
    organize_data()

import pickle
import os
import hashlib
from functools import wraps
import matplotlib.pyplot as plt
import time
import psutil


def memoize(file_name="cache\\DataTS.pkl"):
    """
    Decorator for caching function results in a pickle file
    and speed up the calculation.
    """

    def decorator(func):
        def wrapper(self, *args, **kwargs):

            graph_hash = hashlib.sha256(
                str((sorted(self.edges()), self.id)).encode()
            ).hexdigest()
            key = f"{func.__name__}_{graph_hash}"
            path = os.path.join(os.getcwd(), file_name)

            if os.path.exists(path):

                with open(path, "rb") as file:
                    try:
                        data = pickle.load(file)
                    except EOFError:
                        data = {}
                if key in data:
                    return data[key]

            result = func(self, *args, **kwargs)
            try:
                if os.path.exists(path):
                    with open(path, "rb") as file:
                        data = pickle.load(file)
                else:
                    data = {}
            except EOFError:
                data = {}
            data[key] = result
            with open(path, "wb") as file:
                pickle.dump(data, file)

            return result

        return wrapper

    return decorator


def memoize_or_save(func):
    @wraps(func)
    def wrapper(Lu, Ld, **kwargs):
        dictionary_type = kwargs.get("dictionary_type", "separated")
        prob_T = kwargs.get("prob_T", 1)
        sparsity_mode = kwargs.get("sparsity_mode", "max")
        sparsity = kwargs.get("K0_max", 20)

        if prob_T == 1:
            name = f"full_data_{dictionary_type}"
        else:
            name = f"top_data_T{int(prob_T*100)}"
        path = os.getcwd()
        dir = (
            f"max_sparsity{sparsity}"
            if sparsity_mode == "max"
            else f"random_sparsity{sparsity}"
        )
        dir_path = f"{path}\\synthetic_data\\{dir}"
        filename = f"{dir_path}\\{name}.pkl"

        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
            f.close()
        except FileNotFoundError:
            D_true, Y_train, Y_test, X_train, X_test, epsilon_true, c_true = func(
                Lu, Ld, **kwargs
            )
            data = {
                "D_true": D_true,
                "Y_train": Y_train,
                "Y_test": Y_test,
                "X_train": X_train,
                "X_test": X_test,
                "epsilon_true": epsilon_true,
                "c_true": c_true,
            }

            os.makedirs(dir_path, exist_ok=True)
            with open(filename, "wb") as f:
                pickle.dump(data, f)
            f.close()
        return data

    return wrapper


def final_save(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_params = func.__code__.co_varnames
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
            else:
                prob_T = kwargs["init_params"].get("true_prob_T", None)
                k0 = kwargs["init_params"].get("sparsity", 20)

        else:
            raise ValueError(
                "Parameter 'true_prob_T' not defined in function signature."
            )

        if "simulation_params" in func_params:
            d = kwargs["simulation_params"].get("true_dictionary_type", "separated")
            mode = kwargs["simulation_params"].get("sparsity_mode", "max")

        res_dir = "results\\final"
        res, models = func(*args, **kwargs)
        dir = f"max_sparsity{k0}" if mode == "max" else f"random_sparsity{k0}"
        path = os.getcwd()
        dir_path = os.path.join(path, res_dir, dir)

        os.makedirs(dir_path, exist_ok=True)

        if func.__name__ == "dict_and_topology_learning":
            filename = f"res_{d}_T{int(prob_T*100)}_{mode}{k0}.pkl"
        elif func.__name__ == "param_dict_learning":
            filename = f"res_{d}_T{int(prob_T*100)}.pkl"
        elif func.__name__ == "analyt_dict_learning":
            filename = f"res_{d}_T{int(prob_T*100)}_analyt.pkl"
        else:
            return res, models, file_path
        file_path = os.path.join(dir_path, filename)
        with open(file_path, "wb") as file:
            pickle.dump(models, file)
            pickle.dump(res, file)

        return res, models, file_path

    return wrapper


def save_plot(func):
    """
    Decorator to save the plot returned by the wrapped function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        func_name = func.__name__
        plot = func(*args, **kwargs)
        path = os.getcwd()
        dir_name = "plots\\final"
        d = kwargs.get("dictionary_type", "separated")
        p = kwargs.get("prob_T", 1)
        te = kwargs.get("test_error", True)
        k0 = kwargs.get("sparsity", 20)
        k = kwargs.get("algo_sparsity", 20)
        mode = kwargs.get("sparsity_mode", "max")
        learn_mode = kwargs.get("mode", "optimistic")
        j = kwargs.get("j", 2)
        P = kwargs.get("p", 3)
        dir_name += f"\\max_sparsity{k0}" if mode == "max" else f"\\random_sparsity{k0}"

        if func_name == "plot_error_curves":
            file_name = (
                f"test_error_{d}_T{int(p*100)}.png"
                if te
                else f"train_error_{d}_T{int(p*100)}.png"
            )
        elif func_name == "plot_error_curves_real":
            file_name = f"test_error_real_J{j}P{P}.png"
            dir_name = "plots\\real"

        elif func_name == "plot_topology_approx_errors":
            file_name = "topology_approx_error.png"

        elif func_name == "plot_topology_approx_errors_dual":
            file_name = "topology_approx_error_dual.png"

        elif func_name == "plot_learnt_topology":
            file_name = f"learnt_topology_T{int(p*100)}_S{k}.png"

        elif func_name == "plot_learnt_topology_real":
            file_name = f"learnt_topology_S{k}J{j}P{P}.png"
            dir_name = "plots\\real"

        elif func_name == "plot_changepoints_curve":
            if learn_mode == "optimistic":
                file_name = f"topocp_learning_curve_T{int(p*100)}_S{k}.png"
            else:
                file_name = f"topocp_learning_curve_T{int(p*100)}_S{k}_pess.png"

        elif func_name == "plot_analytic_error_curves":
            file_name = f"test_error_{d}_T{int(p*100)}_analyt.png"

        else:
            return plot

        dir_path = os.path.join(path, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, file_name)
        plt.savefig(file_path)
        plt.close()
        print(f"Plot saved to {file_path}")

        return plot

    return wrapper


def track_performances(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        cpu_start = process.cpu_times()
        mem_start = process.memory_info().rss  # Resident Set Size (RSS) in bytes

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        cpu_end = process.cpu_times()
        mem_end = process.memory_info().rss

        performance_data = {
            "execution_time": end_time - start_time,
            "memory_usage_mb": (mem_end - mem_start) / (1024 * 1024),
            "cpu_time_user": cpu_end.user - cpu_start.user,
            "cpu_time_system": cpu_end.system - cpu_start.system,
        }

        return (
            result,
            performance_data,
        )

    return wrapper

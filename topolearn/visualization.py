from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import seaborn as sns
import networkx as nx
import pandas as pd
import numpy as np
from topolearn.utils import save_plot
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from networkx.algorithms.cycles import find_cycle
import matplotlib.pylab as pyl


@save_plot
def plot_error_curves(
    dict_errors,
    K0_coll,
    **kwargs,
) -> plt.Axes:
    """
    Plot the test error curves for learning algorithms comparing Fourier, Edge, Joint, and Separated dictionary parametrization.

    Parameters:
    - k0_coll (List[int]): Collection of sparsity levels.
    - dictionary_type (str): Type of dictionary used ('fou', 'edge', 'joint', 'sep', 'comp').

    Returns:
    - plt.Axes: The Axes object of the plot.
    """
    sns.set(font_scale=1)
    params = {"dictionary_type": "separated", "prob_T": 1.0, "test_error": True}

    params.update(kwargs)
    dictionary_type = params["dictionary_type"]
    prob_T = params["prob_T"]
    test_error = params["test_error"]

    dict_types = {
        "fourier": "Topological Fourier",
        "edge": "Edge Laplacian",
        "joint": "Joint Hodge Laplacian",
        "separated": "Separated Hodge Laplacian",
        "complete": "Separated Hodge Laplacian with Topology Learning",
        "complete_pess": "Separated Hodge Laplacian with Pessimistic Topology Learning",
    }
    # TITLE = [dict_types[typ] for typ in dict_types.keys() if typ in dictionary_type][0]
    i = 0 if test_error else 1
    res_df = pd.DataFrame()
    for typ in dict_errors.keys():
        tmp_df = pd.DataFrame(dict_errors[typ][i])
        tmp_df.columns = K0_coll
        tmp_df = tmp_df.melt(var_name="Sparsity", value_name="Error")
        tmp_df["Method"] = dict_types[typ] if typ in dict_types.keys() else None
        pass
        res_df = pd.concat([res_df, tmp_df]).reset_index(drop=True)

    markers = (
        [">", "^", "v", "d"]
        if len(dict_errors.keys()) == 4
        else [">", "^", "v", "d", "s"]
    )
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    my_plt = sns.lineplot(
        data=res_df,
        x="Sparsity",
        y="Error",
        hue="Method",
        palette=sns.color_palette()[: len(dict_errors)],
        markers=markers,
        dashes=False,
        style="Method",
    )

    my_plt.set(yscale="log")
    # my_plt.set_title(f"True dictionary: {TITLE}")
    xlabel = "Test" if test_error else "Training"
    my_plt.set_ylabel(f"{xlabel} NMSE (log scale)", fontsize=15)
    my_plt.set_xlabel(f"Sparsity", fontsize=15)
    handles, labels = my_plt.get_legend_handles_labels()
    my_plt.legend(handles=handles[0:], labels=labels[0:])
    pyl.setp(my_plt.get_legend().get_texts(), fontsize="14")  # for legend text
    pyl.setp(my_plt.get_legend().get_title(), fontsize="14")
    return my_plt


@save_plot
def plot_error_curves_real(dict_errors, K0_coll, **kwargs) -> plt.Axes:
    """
    Plot the test error curves for learning algorithms comparing Fourier, Edge, Joint, and Separated dictionary parametrization.

    Parameters:
    - k0_coll (List[int]): Collection of sparsity levels.
    - dictionary_type (str): Type of dictionary used ('fou', 'edge', 'joint', 'sep', 'comp').

    Returns:
    - plt.Axes: The Axes object of the plot.
    """
    sns.set(font_scale=1)
    dict_types = {
        "classic_fourier": "Fourier",
        "fourier": "Topological Fourier",
        "slepians": "Topological Slepians",
        "wavelet": "Hodgelet",
        "separated": "Learnable Hodge Laplacian",
        "complete": "Learnable Hodge Laplacian with Topology learning",
        # "complete_pess": "Separated Hodge Laplacian with Pessimistic Topology learning",
    }

    res_df = pd.DataFrame()
    for typ in dict_types.keys():
        tmp_df = pd.DataFrame(dict_errors[typ][0])
        tmp_df = tmp_df.transpose()
        tmp_df.columns = K0_coll
        tmp_df = tmp_df.melt(var_name="Sparsity", value_name="Error")
        tmp_df["Method"] = dict_types[typ]
        res_df = pd.concat([res_df, tmp_df]).reset_index(drop=True)

    markers = (
        ["p", ">", "*", "8", "d", "s"]
        if len(list(dict_errors.keys())) == 5
        else [">", "*", "8", "d", "s"]
    )
    colors1 = sns.color_palette()[: len(list(dict_errors.keys())) - 1]
    colors2 = sns.color_palette("Dark2", 5)
    colors1[1] = colors2[-2]
    colors1[2] = colors2[-1]
    colors1 = [colors1[-1]] + colors1[:-1]

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    my_plt = sns.lineplot(
        data=res_df,
        x="Sparsity",
        y="Error",
        hue="Method",
        palette=colors1,
        markers=markers,
        dashes=False,
        style="Method",
    )
    my_plt.set(yscale="log")
    my_plt.set_ylabel("Test NMSE (log scale)", fontsize=15)
    my_plt.set_xlabel(f"Sparsity", fontsize=15)
    handles, labels = my_plt.get_legend_handles_labels()
    my_plt.legend(handles=handles[0:], labels=labels[0:])
    pyl.setp(my_plt.get_legend().get_texts(), fontsize="12")  # for legend text
    pyl.setp(my_plt.get_legend().get_title(), fontsize="12")
    return my_plt


@save_plot
def plot_analytic_error_curves(
    analytic_dict_errors,
    dict_errors,
    K0_coll,
    **kwargs,
) -> plt.Axes:
    """
    Plot the test error curves for learning algorithms comparing Fourier, Edge, Joint, and Separated dictionary parametrization.

    Parameters:
    - k0_coll (List[int]): Collection of sparsity levels.
    - dictionary_type (str): Type of dictionary used ('fou', 'edge', 'joint', 'sep', 'comp').

    Returns:
    - plt.Axes: The Axes object of the plot.
    """
    sns.set(font_scale=1)
    params = {"dictionary_type": "separated", "prob_T": 1.0, "test_error": True}

    params.update(kwargs)
    dictionary_type = params["dictionary_type"]
    prob_T = params["prob_T"]
    test_error = params["test_error"]

    dict_types = {
        "fourier": "Topological Fourier",
        "slepians": "Topological Slepians",
        "wavelet": "Separated Hodgelet",
        "separated": "Separated Hodge Laplacian",
    }
    # TITLE = [dict_types[typ] for typ in dict_types.keys() if typ in dictionary_type][0]
    i = 0 if test_error else 1
    res_df = pd.DataFrame()
    for typ in analytic_dict_errors.keys():
        tmp_df = pd.DataFrame(analytic_dict_errors[typ][i])
        tmp_df.columns = K0_coll
        tmp_df = tmp_df.melt(var_name="Sparsity", value_name="Error")
        tmp_df["Method"] = dict_types[typ]
        res_df = pd.concat([res_df, tmp_df]).reset_index(drop=True)

    for typ in dict_errors.keys():
        if typ in dict_types.keys():
            tmp_df = pd.DataFrame(dict_errors[typ][i])
            tmp_df.columns = K0_coll
            tmp_df = tmp_df.melt(var_name="Sparsity", value_name="Error")
            tmp_df["Method"] = dict_types[typ]
            res_df = pd.concat([res_df, tmp_df]).reset_index(drop=True)
        else:
            pass

    markers = [">", "*", "8", "d"]
    colors1 = sns.color_palette()[: len(dict_types)]
    colors2 = sns.color_palette("Dark2", 5)
    colors1[1] = colors2[-2]
    colors1[2] = colors2[-1]

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    my_plt = sns.lineplot(
        data=res_df,
        x="Sparsity",
        y="Error",
        hue="Method",
        palette=colors1,
        markers=markers,
        dashes=False,
        style="Method",
    )

    my_plt.set(yscale="log")
    my_plt.set_ylabel("Test NMSE (log scale)", fontsize=15)
    my_plt.set_xlabel(f"Sparsity", fontsize=15)
    handles, labels = my_plt.get_legend_handles_labels()
    my_plt.legend(handles=handles[0:], labels=labels[0:])
    pyl.setp(my_plt.get_legend().get_texts(), fontsize="14")  # for legend text
    pyl.setp(my_plt.get_legend().get_title(), fontsize="14")

    return my_plt


@save_plot
def plot_topology_approx_errors(res_df, K0_coll, log_scale=True, **kwargs):

    sns.set(font_scale=1)
    print(res_df)
    mask = res_df["Sparsity"].isin(K0_coll)
    # Filter the original dataframe to retain only useful info
    res_df = res_df[mask]
    res_df = res_df.reset_index()
    my_plt = sns.lineplot(
        data=res_df,
        x="Sparsity",
        y="Error",
        hue="Number of Triangles",
        style="Number of Triangles",
        markers=True,
        palette=sns.color_palette("viridis"),
        errorbar=None,
    )
    if log_scale:
        my_plt.set(yscale="log")
    my_plt.set_ylabel("Laplacian approx. error", fontsize=15)
    my_plt.set_xlabel(f"Number of Triangles", fontsize=15)
    sns.set_style("whitegrid")
    pyl.setp(my_plt.get_legend().get_texts(), fontsize="14")
    pyl.setp(my_plt.get_legend().get_title(), fontsize="14")

    return my_plt


@save_plot
def plot_topology_approx_errors_dual(res_df, K0_coll, log_scale=True, **kwargs):

    sns.set(font_scale=1)
    mask = res_df["Sparsity"].isin(K0_coll)
    # Filter the original dataframe to retain only useful info
    res_df = res_df[mask]
    res_df = res_df.reset_index()
    my_plt = sns.lineplot(
        data=res_df,
        x="Number of Triangles",
        y="Error",
        hue="Sparsity",
        style="Sparsity",
        markers=True,
        palette=sns.color_palette("rocket"),
        errorbar=None,
    )
    if log_scale:
        my_plt.set(yscale="log")
    my_plt.set_ylabel("Laplacian approx. error", fontsize=15)
    my_plt.set_xlabel(f"Number of Triangles", fontsize=15)
    sns.set_style("whitegrid")
    pyl.setp(my_plt.get_legend().get_texts(), fontsize="14")
    pyl.setp(my_plt.get_legend().get_title(), fontsize="14")

    return my_plt


@save_plot
def plot_changepoints_curve(
    history,
    nu,
    burn_in: float = 0,
    a=0.1,
    b=0.1,
    c=0.9,
    d=0.01,
    yscale: str = "log",
    zoom_region: bool = True,
    sparse_plot=False,
    include_burn_in=False,
    change_region_len=3,
    **kwargs,
):

    # mode = kwargs.get("mode", "optimistic")
    # p = kwargs.get("prob_T")
    # k0 = kwargs.get("algo_sparsity")
    # T = int(np.ceil(nu * (1 - p)))
    start_iter = 0
    end_iter = 0
    change_points = []
    change_points_y1 = []
    change_points_y2 = []
    burn_in_iter = 0
    his = []
    xx = []
    for i, h in enumerate(history):
        if i == 0:
            burn_in_iter = int(np.ceil(burn_in * len(h)))
        if h != []:
            his += h
            end_iter += len(h) - 1
            tmp = range(start_iter, end_iter + 1)
            xx += tmp
            start_iter = end_iter
            change_points.append(end_iter)
            change_points_y1.append(h[-1])
            change_points_y2.append(h[0])
        else:
            pass

    plt_data = pd.DataFrame({"y": his[burn_in_iter:], "x": xx[burn_in_iter:]})

    change_points = np.array(change_points[:-1])
    change_points_y1 = np.array(change_points_y1[:-1])
    change_points_y2 = np.array(change_points_y2[1:])
    # change_points_y = plt_data[plt_data['x'].isin(change_points)].y.to_numpy()[np.arange(0, len(change_points), 1)]

    my_fig = plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid", {"grid.color": ".6", "grid.linestyle": ":"})

    my_plt = sns.lineplot(x=plt_data["x"], y=plt_data["y"], estimator=None, sort=False)

    # labels = (
    #     ("removing", "Optimistic")
    #     if mode == "optimistic"
    #     else ("adding", "Pessimistic")
    # )
    # operation = "from" if mode == "optimistic" else "to"
    # Change-points
    sns.scatterplot(
        x=np.hstack([change_points, change_points]),
        y=np.hstack([change_points_y1, change_points_y2]),
        # label=f"Change Point: optimally {labels[0]} \n a triangle {operation} Upper Laplacian.",
        color="purple",
        marker="d",
    )

    plt.vlines(
        x=change_points,
        color="lightblue",
        linestyle="dotted",
        ymax=change_points_y1,
        ymin=change_points_y2,
    )

    # Burn-in area
    plt.axvspan(0, burn_in_iter, color="grey", alpha=0.2, hatch="//")

    if include_burn_in:
        x0, xmax = plt.xlim()
    else:
        x0, xmax = plt.xlim()
        x0 = burn_in_iter

    y0, ymax = plt.ylim()

    # my_plt.set_title(f"{labels[1]} topology learning", fontsize=16, pad=25)
    # plt.suptitle(
    #     f"Assumed signal sparsity: {k0}  -  Step size h: {step_h}  -  Step size X: {step_x}",
    #     fontsize=12,
    #     # color="gray",
    #     x=0.5,
    #     y=0.92,
    # )
    if burn_in_iter > 0.0:
        plt.text(
            y=ymax * a,
            x=xmax * b,
            s=f"Burn-in: {burn_in_iter} iters.",
            fontsize=15,
            color="gray",
        )
    # ni = nu - change_points.shape[0] if mode == "optimistic" else change_points.shape[0]

    # plt.text(
    #     s=f" Number of inferred triangles: {ni} \n Number of true triangles: {nu-T}",
    #     y=ymax * c,
    #     x=xmax * d,
    #     fontsize=12,
    #     color="purple",
    # )
    my_plt.set_xlabel("Iteration")

    if sparse_plot:
        tmp_vector = np.ones(len(change_points))
        tmp_vector[1::2] = 0
        plt.xticks(change_points * tmp_vector)
    else:
        if change_points.shape[0] > 1:
            plt.xticks(change_points)
    plt.xlim(left=x0, right=xmax)

    if yscale == "log":
        my_plt.set_ylabel("F.O. (Log Scale)")
        plt.yscale("log")
    else:
        my_plt.set_ylabel("F.O.")

    plt.yticks([])

    # Identify the region where y-values change slowly
    if zoom_region:
        if change_region_len == 0:
            y_diff = np.abs(np.diff(plt_data["y"]))
            slow_change_indices = np.where(y_diff < 1e-2)[0]

            if len(slow_change_indices) > 0:
                # Select the first significant region of slow change
                zoom_start = slow_change_indices[0]
                zoom_end = slow_change_indices[-1]

                sns.lineplot(
                    x=plt_data["x"].iloc[zoom_start:],
                    y=plt_data["y"].iloc[zoom_start:],
                    estimator=None,
                    sort=False,
                    ax=ax_inset,
                )
        elif change_region_len == None:
            pass
        else:
            zoom_start = change_points[-change_region_len]
            # zoom_end = len(plt_data["x"])-1
            # Create inset axes for zoomed-in region
            ax_inset = inset_axes(
                my_plt, width="50%", height="40%", loc="center right", borderpad=2
            )
            sns.lineplot(
                x=plt_data["x"].iloc[zoom_start:],
                y=plt_data["y"].iloc[zoom_start:],
                estimator=None,
                sort=False,
                ax=ax_inset,
            )
            # Change-points
            sns.scatterplot(
                x=np.hstack(
                    [
                        change_points[-change_region_len:],
                        change_points[-change_region_len:],
                    ]
                ),
                y=np.hstack(
                    [
                        change_points_y1[-change_region_len:],
                        change_points_y2[-change_region_len:],
                    ]
                ),
                color="purple",
                marker="d",
            )

        # Set limits for the zoomed-in region
        # print(zoom_end)
        # print(plt_data["x"].iloc[zoom_end])
        # ax_inset.set_xlim(plt_data["x"].iloc[zoom_start], plt_data["x"].iloc[zoom_end])
        # ax_inset.set_ylim(
        #     plt_data["y"].iloc[zoom_start:].min(),
        #     plt_data["y"].iloc[zoom_start:].max(),
        # )
        ax_inset.set_xlabel(None)
        ax_inset.set_ylabel(None)
        # ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        if yscale == "log":
            ax_inset.set_yscale("log")
        # ax_inset.set_title("Zoomed-in region", fontsize=10)

    return my_fig


@save_plot
def plot_learnt_topology(
    G_true,
    Lu_true,
    B2_true,
    model_gt,
    model_opt,
    model_pess=None,
    sub_size=100,
    **kwargs,
):
    sns.set(font_scale=1.7)
    if model_pess != None:

        topos = [model_gt, model_opt, model_pess]
        num_triangles = [
            model_gt.get_numb_triangles(),
            model_opt.get_numb_triangles("optimistic"),
            0,
        ]
        incidence_mat = [B2_true, model_opt.B2, model_pess.B2]
        titles = [
            "True number of triangles: ",
            "Inferred number of triangles: ",
            "Inferred number of triangles: ",
        ]
        _, axs = plt.subplots(1, 3, figsize=(16, 6))
    else:
        topos = [model_gt, model_opt]
        num_triangles = [
            model_gt.get_numb_triangles(),
            model_opt.get_numb_triangles("optimistic"),
        ]
        incidence_mat = [B2_true, model_opt.B2]
        titles = [
            "True number of triangles: \n",
            "Inferred number of triangles: \n",
        ]
        _, axs = plt.subplots(1, 2, figsize=(12, 6))

    i = 0
    for ax, title in zip(axs, titles):
        A = G_true.get_adjacency()
        tmp_G = nx.from_numpy_array(A)
        pos = nx.kamada_kawai_layout(tmp_G)
        nx.draw(tmp_G, pos, with_labels=False, node_color="purple", node_size=15, ax=ax)

        for triangle_index in range(B2_true.shape[1]):
            np.random.seed(triangle_index)
            color = np.random.rand(3)
            triangle_vertices = []

            for edge_index, edge in enumerate(tmp_G.edges):
                if edge_index < sub_size:
                    if incidence_mat[i][edge_index, triangle_index] != 0:
                        pos1 = tuple(pos[edge[0]])
                        pos2 = tuple(pos[edge[1]])
                        if pos1 not in triangle_vertices:
                            triangle_vertices.append(pos1)
                        if pos2 not in triangle_vertices:
                            triangle_vertices.append(pos2)
            if triangle_vertices != []:
                if i == 2:
                    num_triangles[-1] += 1
                triangle_patch = Polygon(
                    triangle_vertices,
                    closed=True,
                    facecolor=color,
                    edgecolor="black",
                    alpha=0.3,
                )
                ax.add_patch(triangle_patch)

        ax.set_title(title + str(num_triangles[i]))
        ax.text(
            0.5,
            -0,
            r"$\frac{||L_u - \hat{L}_u^*||^2}{||L_u||^2}$:"
            + f" {np.round(topos[i].get_topology_approx_error(Lu_true, 4)/np.linalg.norm(Lu_true), 4)}      NMSE: {topos[i].get_test_error(4)}",
            ha="center",
            transform=ax.transAxes,
        )

        i += 1

    plt.tight_layout()


@save_plot
def plot_learnt_topology_real(
    B1_true,
    sep_model,
    model_opt,
    model_pess,
    sub_size=np.inf,
    **kwargs,
):

    # # Add edges to the graph based on the incidence matrix
    # for edge_index in range(B1_true.shape[1]):  # iterate over edges (columns)
    #     nodes = np.where(np.abs(B1_true[:, edge_index]) == 1)[
    #         0
    #     ]  # nodes connected by this edge
    #     if len(nodes) == 2:
    #         G.add_edge(nodes[0], nodes[1])
    # # print(G.edges)
    # # Get the adjacency matrix from the graph
    # A = nx.adjacency_matrix(G)

    sns.set(font_scale=2)
    A = np.zeros((B1_true.shape[0], B1_true.shape[0]))

    for edge_index in range(B1_true.shape[1]):
        nodes = np.where(B1_true[:, edge_index] != 0)[0]
        A[nodes[0], nodes[1]] = 1

    G = nx.from_numpy_array(A)

    topos = [sep_model, model_opt, model_pess]
    num_polygons = [
        0,
        0,
        0,
    ]
    incidence_mat = [sep_model.B2, model_opt.B2, model_pess.B2]
    titles = [
        "Assumed number of polygons: ",
        "Inferred number of polygons : ",
        "Inferred number of polygons: ",
    ]
    _, axs = plt.subplots(1, 3, figsize=(16, 6))

    i = 0
    for ax, title in zip(axs, titles):
        pos = nx.kamada_kawai_layout(G)
        nx.draw(G, pos, with_labels=False, node_color="purple", node_size=15, ax=ax)
        B2 = incidence_mat[i]
        for polygon_index in range(B2.shape[1]):
            np.random.seed(polygon_index)
            color = np.random.rand(3)

            polygon_nodes = np.array([])
            polygon_edges = []
            edges = np.where(B2[:, polygon_index] != 0)[0]

            for edge_index in edges:

                nodes = np.array(np.where(B1_true[:, edge_index] != 0)[0], dtype=int)
                polygon_nodes = np.array(
                    np.concatenate((polygon_nodes, nodes)), dtype=int
                )
                polygon_edges.append(tuple(nodes))

            subgraph = G.edge_subgraph(polygon_edges)

            # Find the cycle in the subgraph to get the boundary nodes
            try:
                cycle_edges = find_cycle(subgraph)
                polygon_nodes = [pos[edge[0]] for edge in cycle_edges] + [
                    pos[cycle_edges[0][0]]
                ]
                num_polygons[i] += 1
                p = Polygon(
                    polygon_nodes,
                    facecolor=color,
                    fill=True,
                    edgecolor="black",
                    alpha=0.3,
                )
                ax.add_patch(p)
            except:
                pass

        ax.set_title(title + str(num_polygons[i]))
        ax.text(
            0.5,
            -0,
            f"NMSE: {topos[i].get_test_error(4)}",
            ha="center",
            transform=ax.transAxes,
        )

        i += 1

    plt.tight_layout()


def plot_algo_errors(errors: dict[str, np.ndarray], k0_coll: np.ndarray) -> plt.Axes:
    """
    Plot the algorithm errors against the sparsity levels, comparing different implementations
    of algorithms for learning representations of topological signals:
    - Semi-definite Programming dictionary and sparse representation joint learning (SDP);
    - Quadratic Programming dictionary and sparse representation joint learning (QP);
    - Quadratic Programming dictionary, sparse representation and topology (upper laplacian) joint learning (QP COMPLETE).

    All of the above methods in this case are considered in the "Separated Hodge" laplacian parametrization setup.

    Parameters:
    errors (dict[str, np.ndarray]): A dictionary containing error matrices for different algorithms.
                                    The keys are algorithm names, and the values are 2D numpy arrays
                                    where each row represents the errors for a single simulation.
    k0_coll (np.ndarray): An array of sparsity levels.

    Returns:
    plt.Axes: The axes object with the plotted data.
    """
    dict_types = {"qp": "QP", "sdp": "SDP", "qp_comp": "QP COMPLETE"}

    res_df = pd.DataFrame()
    n_sim = errors["qp"].shape[0]

    for algorithm, error_matrix in errors.items():
        for sim in range(n_sim):
            tmp_df = pd.DataFrame()
            tmp_df["Error"] = error_matrix[sim, :]
            tmp_df["Sparsity"] = k0_coll
            tmp_df["Algorithm"] = dict_types[algorithm]
            res_df = pd.concat([res_df, tmp_df])

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    my_plt = sns.lineplot(
        data=res_df,
        x="Sparsity",
        y="Error",
        hue="Algorithm",
        palette=sns.color_palette("husl"),
        markers=[">", "^", "v"],
        dashes=False,
        style="Algorithm",
    )
    my_plt.set(yscale="log")
    my_plt.set_title("Topology learning: algorithms comparison")
    my_plt.set_ylabel("Error (log scale)")

    return my_plt

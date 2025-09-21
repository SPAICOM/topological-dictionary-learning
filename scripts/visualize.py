import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from networkx.algorithms.cycles import find_cycle
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

plt.style.use("../.conf/plotting/plt.mplstyle")


def plot_learnt_topology_real(
    B1_true,
    model1,
    model2,
    model3,
    pos,
    K0_coll,
    common,
    sub_size=np.inf,
    **kwargs,
):
    sns.set(font_scale=2)
    A = np.zeros((B1_true.shape[0], B1_true.shape[0]))

    for edge_index in range(B1_true.shape[1]):
        nodes = np.where(B1_true[:, edge_index] != 0)[0]
        A[nodes[0], nodes[1]] = 1

    G = nx.from_numpy_array(A)

    topos = [model1, model2, model3]
    num_polygons = [
        0,
        0,
        0,
    ]
    incidence_mat = [model1.B2, model2.B2, model3.B2]
    _, axs = plt.subplots(1, 3, figsize=(14, 5))
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times"]
    plt.rcParams["mathtext.fontset"] = "stix"
    i = 0
    for ax in axs:
        nx.draw(
            G,
            pos,
            with_labels=False,
            node_color="purple",
            node_size=45,
            ax=ax,
            font_size=20,
        )
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
            try:
                cycle_edges = find_cycle(subgraph)
                polygon_coords = [pos[edge[0]] for edge in cycle_edges] + [
                    pos[cycle_edges[0][0]]
                ]
                if cycle_edges in common:
                    color = "red"
                    alpha = 0.7
                else:
                    alpha = 0.4
                num_polygons[i] += 1
                p = Polygon(
                    polygon_coords,
                    facecolor=color,
                    fill=True,
                    edgecolor="black",
                    alpha=alpha,
                )
                ax.add_patch(p)
            except Exception:
                pass

        ax.set_title(
            r"$K_0$: "
            + f"{K0_coll[i]}"
            + "\n"
            + r"$\operatorname{NMSE}$: "
            + f"{topos[i].get_test_error(4)}",
            fontdict={"family": "Times New Roman"},
        )
        ax.text(
            0.5,
            -0,
            f"Inferred Polygons: {num_polygons[i]}",
            ha="center",
            transform=ax.transAxes,
        )

        i += 1
    plt.savefig("plots\\RealTopos.pdf", format="pdf", bbox_inches="tight")


def nmse_curves(
    dict_errors: pd.DataFrame,
    K0_coll: np.ndarray,
    real: bool = False,
) -> None:
    dict_types = (
        {
            "classic_fourier": "Fourier",
            "fourier": "Topological Fourier [12]",
            "slepians": "Topological Slepians [45]",
            "wavelet": "Hodgelets [44]",
            "complete": "GTDL",
            "complete_soft": "RTDL",
        }
        if real
        else {
            "fourier": "Topological Fourier [12]",
            "edge": "Edge Laplacian",
            "joint": "Joint Hodge Laplacian",
            "separated": "Separated Hodge Laplacian",
            "complete_greedy": "GTDL",
            "complete": "RTDL",
        }
    )

    res_df = pd.DataFrame()
    for typ in dict_types.keys():
        if typ == "complete_soft":
            tmp_df = pd.DataFrame(dict_errors[typ])
        else:
            tmp_df = pd.DataFrame(dict_errors[typ][0])
        tmp_df = tmp_df.transpose()
        tmp_df.columns = K0_coll
        tmp_df = tmp_df.melt(var_name="Sparsity", value_name="Error")
        tmp_df["Method"] = dict_types[typ]
        res_df = pd.concat([res_df, tmp_df]).reset_index(drop=True)

    markers = ["o", ">", "*", "8", "s", "X"] if real else [">", "^", "v", "d", "s", "X"]
    if real:
        colors1 = sns.color_palette()[: len(list(dict_errors.keys())) - 1]
        colors2 = sns.color_palette("Dark2", 5)
        colors1[1] = colors2[-2]
        colors1[2] = colors2[-1]
        colors1 = [colors1[-1]] + colors1[:-1]
        colors1[4] = colors2[-3]
        colors1[5] = sns.color_palette()[5]
    else:
        colors1 = sns.color_palette()[: len(list(dict_errors.keys()))]

    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["font.size"] = 18
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times"]
    plt.rcParams["mathtext.fontset"] = "stix"
    my_plt = sns.lineplot(
        data=res_df,
        x="Sparsity",
        y="Error",
        hue="Method",
        palette=colors1,
        markers=markers,
        markersize=12,
        dashes=False,
        style="Method",
    )
    my_plt.set(yscale="log")
    my_plt.set_ylabel(r"NMSE$(\mathbf{Y},\widehat{\mathbf{Y}})$", fontsize=19)
    my_plt.set_xlabel(f"Number of non-zero coefficients", fontsize=20)
    handles, labels = my_plt.get_legend_handles_labels()
    my_plt.legend(handles=handles[0:], labels=labels[0:])
    plt.legend(
        fontsize=14,
        frameon=False,
        framealpha=0.3,
    )
    name = "SignalErrorReal" if real else "SignalError"
    plt.savefig(f"plots\\{name}.pdf", format="pdf", bbox_inches="tight")

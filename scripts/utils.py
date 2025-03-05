from topolearn import EnhancedGraph
import numpy as np


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
    B2 = G.get_b2()

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

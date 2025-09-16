"This file contains the EnhancedGraph class that extends the networkx Graph class generalizing it to a second-order cell complex"

import numpy as np
import networkx as nx
from scipy.linalg import qr
from typing import List, Tuple
from numpy.linalg import matrix_rank

import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

from src.utils import memoize


class EnhancedGraph(nx.Graph):
    """EnhancedGraph, a class for generating a second-order cell complex

    Args:
        n : int
            Number of nodes in the complex
        p_edges : float
            Probability for edge creation.
        p_triangles : float
            Probability for triangles/polygons creation.
        seed : int
            Seed for the random number generator.
        *args, **kwargs
            Additional arguments passed to the nx.Graph constructor.
    """

    def __init__(self, n=10, p_edges=1.0, p_triangles=1.0, seed=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = f"{seed}_{n}_{p_edges}"
        self.p_triangles = p_triangles
        er_graph = nx.erdos_renyi_graph(n, p_edges, seed=seed)
        self.add_nodes_from(er_graph.nodes(data=True))
        self.add_edges_from(er_graph.edges(data=True))
        self.seed = seed
        self.mask = None

    @memoize()
    def get_adjacency(self) -> np.ndarray:
        """Compute the adjacency matrix of the cell complex skeleton."""
        return nx.adjacency_matrix(self).todense()

    @memoize()
    def get_b1(self) -> np.ndarray:
        """Compute the oriented node-edge incidence matrix B1."""
        return (-1) * nx.incidence_matrix(self, oriented=True).todense()

    def get_cycles(self, max_len=np.inf) -> List[tuple]:
        """Find all cycles in the graph within a specified maximum length.

        Args:
            max_len : int
                Maximum length of the cycles to be considered.

        Returns:
            final : list
                List of cycles found in the cell complex skeleton (graph)
        """

        A = self.get_adjacency()
        G = nx.DiGraph(A)
        cycles = nx.simple_cycles(G)

        seen = set()
        final = []

        for cycle in cycles:
            cycle_tuple = tuple(sorted(cycle))
            if cycle_tuple not in seen and 3 <= len(cycle) <= max_len:
                seen.add(cycle_tuple)
                final.append(cycle)

        final.sort(key=len)
        return final

    @memoize()
    def get_b2(self) -> np.ndarray:
        """Compute the oriented edge-polygon incidence B2."""

        E_list = list(self.edges)
        All_P = self.get_cycles()
        cycles = [cycle + [cycle[0]] for cycle in All_P]
        edge_index_map = {edge: i for i, edge in enumerate(E_list)}
        B2 = np.zeros((len(E_list), len(cycles)))

        for cycle_index, cycle in enumerate(cycles):
            for i in range(len(cycle) - 1):
                edge = (cycle[i], cycle[i + 1])
                edge_reversed = (cycle[i + 1], cycle[i])

                if edge in edge_index_map:
                    B2[edge_index_map[edge], cycle_index] = 1
                elif edge_reversed in edge_index_map:
                    B2[edge_index_map[edge_reversed], cycle_index] = -1

        QR = qr(B2, pivoting=True)
        rank = matrix_rank(B2)
        B2 = B2[:, QR[2][:rank]]

        return B2

    def sub_size_skeleton(
        self, B1: np.ndarray, B2: np.ndarray, sub_size: int
    ) -> Tuple[np.ndarray]:
        """Subsize the cell complex skeleton to decrease complexity."""

        # Sub-sampling if needed to decrease complexity
        if sub_size != None:
            B1 = B1[:, :sub_size]
            B2 = B2[:sub_size, :]

        return B1, B2

    def triangles_only(self, B2: np.ndarray) -> np.ndarray:
        """Consider only triangles as polygons and consequently modify the B2 matrix."""
        B2 = B2[:, np.sum(np.abs(B2), 0) == 3]
        return B2

    def get_laplacians(
        self, sub_size: int = None, full: bool = False, only_triangles: bool = True
    ) -> Tuple[np.ndarray]:
        """Compute the Laplacian of the cell complex and either only return the upper Laplacian or the upper, the lower and the full Laplacian."""

        B1 = self.get_b1()
        B2 = self.get_b2()
        B1, B2 = self.sub_size_skeleton(B1, B2, sub_size=sub_size)

        if only_triangles:
            B2 = self.triangles_only(B2)
        self.nu = B2.shape[1]
        Ld = np.matmul(np.transpose(B1), B1, dtype=float)
        if full:
            Lu = np.matmul(B2, np.transpose(B2), dtype=float)
            return Lu

        self.polygons_mask()
        Lu = B2 @ self.mask @ B2.T
        L = Lu + Ld
        return Lu, Ld, L

    def polygons_mask(self):
        """Create a mask to consider only a subset of the polygons in the cell complex."""
        prob_T = self.p_triangles
        T = int(np.ceil(self.nu * (1 - prob_T)))
        np.random.seed(self.seed)
        mask = np.random.choice(np.arange(self.nu), size=T, replace=False)
        I_T = np.ones(self.nu)
        I_T[mask] = 0
        self.mask = np.diag(I_T)

    def get_mask(self):
        return self.mask

    def mask_B2(self, B2):
        """Mask the B2 matrix to subsize the set of the polygons in the cell complex."""
        return B2 @ self.get_mask()

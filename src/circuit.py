from __future__ import annotations
from typing import Dict
from functools import cached_property
from importlib.resources import path
import math
import numpy as np
import networkx as nx
import rustworkx as rx
from suffix_tree import Tree
from qiskit.circuit.random import random_circuit
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import FilterOpNodes
from qiskit.dagcircuit import DAGOpNode, DAGCircuit
from qiskit.visualization import dag_drawer

from mqt.bench import BenchmarkLevel, get_benchmark
from mqt.bench.benchmarks import get_benchmark_catalog
from mqt.bench.targets import get_target_for_gateset
from src.utils import bcolors


def to_flattened_feature_dict(circuit: Circuit) -> Dict[str, int | float]:
    feature_dict: Dict[str, int | float] = {
        "num_qubits": circuit.num_qubits,
        "size": circuit.size,
        "num_gates": circuit.num_gates,
        "num_1q_gates": circuit.num_1q_gates,
        "num_2q_gates": circuit.num_2q_gates,
        "pct_2q_gates": circuit.pct_2q_gates,
        "depth": circuit.depth,
        "ig_aspl": circuit.ig_aspl,
        "ig_std_adj_mat": circuit.ig_std_adj_mat,
        "ig_diameter": circuit.ig_diameter,
        "ig_max_betweenness": circuit.ig_max_betweenness,
        "ig_avg_degree": circuit.ig_avg_degree,
        "ig_max_degree": circuit.ig_max_degree,
        "ig_std_degree": circuit.ig_std_degree,
        "ig_avg_strength": circuit.ig_avg_strength,
        "ig_max_strength": circuit.ig_max_strength,
        "ig_std_strength": circuit.ig_std_strength,
        "ig_max_cliques_num": circuit.ig_max_cliques[0],
        "ig_max_cliques_size": circuit.ig_max_cliques[1],
        "ig_transitivity": circuit.ig_transitivity,
        "ig_avg_clustering_coef": circuit.ig_avg_clustering_coef,
        "ig_vertex_connectivity": circuit.ig_vertex_connectivity,
        "ig_edge_connectivity": circuit.ig_edge_connectivity,
        "ig_avg_coreness": circuit.ig_avg_coreness,
        "ig_min_coreness": circuit.ig_min_coreness,
        "ig_max_coreness": circuit.ig_max_coreness,
        "ig_std_coreness": circuit.ig_std_coreness,
        "ig_max_pagerank": circuit.ig_max_pagerank,
        "ig_min_pagerank": circuit.ig_min_pagerank,
        "ig_std_pagerank": circuit.ig_std_pagerank,
        "ig_normalized_hhi_pagerank": circuit.ig_normalized_hhi_pagerank,
        "gdg_critical_path_length": circuit.gdg_critical_path_length,
        "gdg_num_critical_paths": circuit.gdg_num_critical_paths,
        "gdg_mean_path_length": circuit.gdg_mean_path_length,
        "gdg_std_path_length": circuit.gdg_std_path_length,
        "gdg_percentage_gates_in_critical_path": circuit.gdg_percentage_gates_in_critical_path,
        "density_score": circuit.density_score,
        "idling_score": circuit.idling_score,
    }
    return feature_dict


class Circuit:
    """Wrapper class for a qiskit.QuantumCircuit with the purpose of exposing an API for a wide range of features characterizing a QuantumCircuit."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        n_qubits: int,
        *,
        name: str = "wrapped",
        desc: str = "No description provided",
        filter_gates: list[str] | None = ["measure", "reset", "barrier", "delay"],
    ):
        if filter_gates and len(filter_gates) > 0:
            circuit = Circuit._filter_gates_by_name(circuit, set(filter_gates))

        self.circuit = circuit
        self.n_qubits = n_qubits
        self.name = name
        self.desc = desc

        self._rng = np.random.default_rng(0)

    @classmethod
    def from_mqt_bench(
        cls,
        name: str,
        n_qubits: int,
        *,
        desc: str = None,
        filter_gates: list[str] | None = ["measure", "reset", "barrier", "delay"],
        **mqt_kwargs,
    ) -> Circuit:
        """
        Wraps an MQTBench benchmark circuit into a Circuit object.
        See MQTBench documentation to find the full list of available benchmarks.
        Alternatively, use get_benchmark_catalog().keys() to list available benchmark names (from mqt.bench.benchmarks import get_benchmark_catalog).

        Args:
            name (str): the name of the benchmark circuit to load from MQTBench.
            n_qubits (int): the number of qubits in the circuit.
            desc (str, optional): a description of the circuit. Defaults to None.
            filter_gates (list[str] | None, optional): a list of gate names to filter out. Defaults to ["measure", "reset", "barrier", "delay"].
            **mqt_kwargs: Additional keyword arguments to pass to the MQTBench API. Defaults to level=BenchmarkLevel.NATIVEGATES, target="ibm_falcon", opt_level=0.

        Returns:
            Circuit: the wrapped Circuit object.
        """
        if not name in get_benchmark_catalog().keys():
            raise ValueError(
                f"Circuit {name} not recognized. Available: {list(get_benchmark_catalog().keys())}"
            )
        qc = get_benchmark(
            benchmark=name,
            level=mqt_kwargs.get("level", BenchmarkLevel.NATIVEGATES),
            circuit_size=n_qubits,
            target=mqt_kwargs.get(
                "target", get_target_for_gateset("ibm_falcon", num_qubits=n_qubits)
            ),
            opt_level=mqt_kwargs.get("opt_level", 0),
        )
        return cls(
            qc,
            n_qubits,
            name=name,
            desc=desc
            or f"MQTBench benchmark '{name}' with {n_qubits} qubits: {get_benchmark_catalog()[name]}",
            filter_gates=filter_gates,
        )

    @classmethod
    def from_qasm_file(
        cls,
        file_path: str,
        *,
        name: str = None,
        desc: str = None,
        filter_gates: list[str] | None = ["measure", "reset", "barrier", "delay"],
    ) -> Circuit:
        """Creates a Circuit object by loading a QuantumCircuit from a QASM file.

        Args:
            file_path (str): The path to the QASM file.
            name (str, optional): The name of the circuit. Defaults to None.
            desc (str, optional): A description of the circuit. Defaults to None.
            filter_gates (list[str] | None, optional): A list of gate names to filter out. Defaults to ["measure", "reset", "barrier", "delay"].

        Returns:
            Circuit: The wrapped Circuit object.
        """
        qc = QuantumCircuit.from_qasm_file(file_path)
        n_qubits = qc.num_qubits
        return cls(
            qc,
            n_qubits,
            name=name or f"circuit_from_{file_path}",
            desc=desc or f"Circuit loaded from QASM file: {file_path}",
            filter_gates=filter_gates,
        )

    @classmethod
    def from_random_circuit(
        cls,
        n_qubits: int,
        *,
        name: str = "random circuit",
        desc: str = None,
        filter_gates: list[str] | None = ["measure", "reset", "barrier", "delay"],
        **random_kwargs,
    ) -> Circuit:
        """Creates a random qiskit.QuantumCircuit using Qiskit's random_circuit function and wraps it into a Circuit object.

        Args:
            n_qubits (int): The number of qubits in the circuit.
            name (str, optional): The name of the circuit. Defaults to "random circuit".
            desc (str, optional): A description of the circuit. Defaults to None.
            filter_gates (list[str] | None, optional): A list of gate names to filter out. Defaults to ["measure", "reset", "barrier", "delay"].
            **random_kwargs: Additional keyword arguments to pass to Qiskit's random_circuit function.

        Returns:
            Circuit: The wrapped Circuit object.
        """
        qc = random_circuit(n_qubits, **random_kwargs)

        return cls(
            qc,
            name=name,
            n_qubits=n_qubits,
            desc=desc
            or f"Randomly generated circuit using Qiskit's random_circuit function. n_qubits={n_qubits}, "
            + ", ".join(f"{k}={v}" for k, v in random_kwargs.items()),
            filter_gates=filter_gates,
        )

    @classmethod
    def from_qiskit(
        cls,
        qc: QuantumCircuit,
        *,
        name: str = "wrapped",
        desc: str = "No description provided",
        filter_gates: list[str] | None = ["measure", "reset", "barrier", "delay"],
    ) -> Circuit:
        """Wraps a Qiskit QuantumCircuit instance into a Circuit object.

        Args:
            qc (QuantumCircuit): The Qiskit QuantumCircuit instance to wrap.
            name (str, optional): The name of the circuit. Defaults to "wrapped".
            desc (str, optional): A description of the circuit. Defaults to "No description provided".
            filter_gates (list[str] | None, optional): A list of gate names to filter out. Defaults to ["measure", "reset", "barrier", "delay"].

        Returns:
            Circuit: The wrapped Circuit object.
        """
        return cls(
            qc,
            n_qubits=qc.num_qubits,
            name=name,
            desc=desc,
            filter_gates=filter_gates,
        )

    # Now following: many cached properties (meaning lazy load once accessed and cache)

    ### size metrics

    @property
    def num_qubits(self) -> int:
        """
        Returns:
            The number of qubits in the circuit: $n_q$.
        """
        return self.n_qubits

    @cached_property
    def size(self) -> int:
        """
        Returns:
            The size of the circuit: $n_g$ (number of operations/gates).
        """
        return self.circuit.size()

    @property
    def num_gates(self) -> int:
        """
        Returns:
            The size of the circuit: $n_g$ (number of operations/gates). Mirror of size().
        """
        return self.size

    @property
    def num_1q_gates(self) -> int:
        """
        Returns:
            The number of one-qubit gates in the circuit: $n_{1qg}$.
        """
        return len(self.one_qubit_gates)

    @property
    def num_2q_gates(self) -> int:
        """
        Returns:
            The number of two-qubit gates in the circuit: $n_{2qg}$.
        """
        return len(self.two_qubit_gates)

    @property
    def pct_2q_gates(self) -> float:
        """
        Returns:
            The ratio of two-qubit gates to total gates: $n_{2qg} / n_g$.
        """
        ng = self.num_gates or 1
        return self.num_2q_gates / ng

    @cached_property
    def depth(self) -> int:
        """
        Returns:
            The depth of the circuit: $d$.
        """
        return self.circuit.depth()

    ### IG metrics

    @cached_property
    def ig_adj_mat(self) -> np.ndarray:
        """
        Returns:
            The adjacency matrix of the circuit including one-qubit gates in the diagonal.
        """
        adj_mat = self._compute_adj_mat()
        return adj_mat

    @property
    def ig_adj_mat_2q(self) -> np.ndarray:
        """
        Returns:
            The adjacency matrix of the circuit considering only two-qubit gates -> 0s in diagonal.
        """
        adj_mat = self.ig_adj_mat.copy()
        np.fill_diagonal(adj_mat, 0)
        return adj_mat

    @cached_property
    def ig(self) -> rx.PyGraph:
        """
        Returns:
            The weighted interaction graph of the circuit. The edge weights correspond to the number of two-qubit gates between qubits, and node weights correspond to the number of one-qubit gates on each qubit saved as node data.
        """
        return self._compute_interaction_graph(weighted=True)

    # not really necessary but since its relatively cheap (generally not many qubits) I rather decide to be explicit
    @cached_property
    def unweighted_ig(self) -> rx.PyGraph:
        """
        Returns:
            The unweighted interaction graph of the circuit. The unweighted version has edges representing the presence of two-qubit gates between qubits.
        """
        return self._compute_interaction_graph(weighted=False)

    @cached_property
    def nx_ig(self) -> nx.Graph:
        """
        Returns:
            The interaction graph as a NetworkX graph with weights.
        """
        G_rx = self.ig
        G_nx = nx.Graph()
        G_nx.add_nodes_from(
            (i, {"n1q": int(G_rx.get_node_data(i))}) for i in G_rx.node_indices()
        )
        G_nx.add_weighted_edges_from(
            (u, v, G_rx.get_edge_data(u, v)) for (u, v) in G_rx.edge_list()
        )
        return G_nx

    @cached_property
    def nx_unweighted_ig(self) -> nx.Graph:
        """
        Returns:
            The unweighted interaction graph as a NetworkX graph.
        """
        G_rx = self.unweighted_ig
        G = nx.Graph()
        G.add_nodes_from(range(G_rx.num_nodes()))
        G.add_edges_from(G_rx.edge_list())
        return G

    @property
    def ig_aspl(self) -> float:
        r"""Average hop count between all pairs of nodes in the IG. Computed using rustworkx's unweighted_average_shortest_path_length(G) function. It is defined by rustworkx as:
        \\[aspl = \sum_{s,t \in V, s \neq t}\frac{d(s,t)}{n(n-1)}\\]

        where $V$ is the set of nodes in $G$, $d(s,t)$ is the shortest path length from $s$ to $t$, and $n$ is the number of nodes in $G$.

        Returns:
            The average shortest path length in the interaction graph.
        """
        if self.num_qubits < 2:
            return 0.0
        return rx.unweighted_average_shortest_path_length(
            self.unweighted_ig, disconnected=True
        )

    @property
    def ig_std_adj_mat(self) -> float:
        """
        Returns:
            Std. dev. of the interaction graphs edge-weight distribution. Undirected: takes upper triangle, including zeros for non-edges: $\\sigma(A)$
        """
        A = self.ig_adj_mat_2q
        iu = np.triu_indices(A.shape[0], k=1)
        return float(np.std(A[iu]))

    @property
    def ig_diameter(self) -> int:
        r"""The diameter of the IG, defined as the longest shortest path between any pair of nodes in the IG:
            \\[dm = \\max_{n_i \\in N}(\\epsilon_i),\\]
            where $\epsilon_i$ is the longest hop count from node $n_i$ to any other node in the IG.

        Returns:
            The diameter of the interaction graph.
        """
        if self.num_qubits < 2:
            return 0
        D = rx.graph_distance_matrix(self.unweighted_ig, null_value=float("inf"))
        finite = [int(v) for row in D for v in row if v != float("inf")]
        return max(finite) if finite else 0

    @property
    def ig_max_betweenness(self) -> float:
        """Equals Central Point Dominance as defined in Bandic et al. 2025.

        Returns:
            The maximal betweenness centrality of any node in the IG. Normalized between 0 and 1.
        """
        if self.num_qubits < 3:
            return 0.0
        betweenness = rx.betweenness_centrality(self.unweighted_ig, normalized=True)
        return max(betweenness.values())

    @cached_property
    def ig_degrees(self) -> np.ndarray:
        """
        Returns:
            Unweighted degree: #distinct neighbors per qubit. (how many distionct qubits each qubit interacts with via 2q gates)
        """
        A = self.ig_adj_mat_2q
        return np.asarray((A > 0).sum(axis=1), dtype=int).ravel()

    @property
    def ig_avg_degree(self) -> float:
        """
        Returns:
            The average degree of the interaction graph. (How many distinct qubits each qubit interacts with via 2q gates on average)
        """
        return float(np.mean(self.ig_degrees))

    @property
    def ig_max_degree(self) -> int:
        """
        Returns:
            The maximum degree of the interaction graph. (How many distinct qubits the most connected qubit interacts with via 2q gates)
        """
        return int(max(self.ig_degrees))

    @property
    def ig_std_degree(self) -> float:
        """
        Returns:
            The standard deviation of the degree distribution of the interaction graph. (How much the number of distinct qubits each qubit interacts with via 2q gates varies)
        """
        return float(np.std(self.ig_degrees))

    @cached_property
    def ig_strengths(self) -> np.ndarray:
        """
        Returns:
            Weighted degree (node strength): sum of 2q edge weights per qubit. (total number of 2q gates each qubit participates in)
        """
        A = self.ig_adj_mat_2q
        return np.asarray(A.sum(axis=1), dtype=float).ravel()

    @property
    def ig_avg_strength(self) -> float:
        """
        Returns:
            The average strength of the interaction graph. (How many 2q gates per qubit on average)
        """
        return np.mean(self.ig_strengths)

    @property
    def ig_max_strength(self) -> int:
        """
        Returns:
            The maximum strength of the interaction graph.
        """
        return int(max(self.ig_strengths))

    @property
    def ig_std_strength(self) -> float:
        """
        Returns:
            The standard deviation of the strength distribution of the interaction graph. (How much the number of 2q gates each qubit participates in varies)
        """
        return float(np.std(self.ig_strengths))

    @property
    def ig_max_cliques(self) -> tuple[int, int]:
        """Finds all maximal cliques in the unweighted IG and returns the number of largest cliques and their size. Uses a NetworkX representation.

        Returns:
            A tuple (num_max_cliques, size_max_clique).
        """
        cliques = list(nx.find_cliques(self.nx_unweighted_ig))
        if not cliques:
            return 0, 0
        max_size = max(len(c) for c in cliques)
        num_max = sum(1 for c in cliques if len(c) == max_size)
        return num_max, max_size

    @property
    def ig_transitivity(self) -> float:
        """
        Returns:
            The transitivity of the interaction graph. (global clustering coefficient)
        """
        if self.num_qubits < 3:
            return 0.0
        return float(rx.transitivity(self.unweighted_ig))

    @property
    def ig_avg_clustering_coef(self) -> float:
        """Uses a NetworkX representation to compute the average local clustering coefficient of the IG.

        Returns:
            The average local clustering coefficient of the interaction graph.
        """
        if self.num_qubits < 3:
            return 0.0
        return float(nx.average_clustering(self.nx_unweighted_ig))

    @property
    def ig_vertex_connectivity(self) -> int:
        """The vertex connectivity of the IG, defined as the minimum number of nodes that need to be removed to disconnect the graph. Uses a NetworkX representation.

        Returns:
            The vertex connectivity of the interaction graph.
        """
        G = self.nx_unweighted_ig
        return 0 if G.number_of_nodes() < 2 else int(nx.node_connectivity(G))

    @property
    def ig_edge_connectivity(self) -> int:
        """The edge connectivity of the IG, defined as the minimum number of edges that need to be removed to disconnect the graph.

        Returns:
            The edge connectivity of the interaction graph.
        """
        G = self.nx_unweighted_ig
        nx_version = 0 if G.number_of_nodes() < 2 else int(nx.edge_connectivity(G))

        cut_value, _ = rx.stoer_wagner_min_cut(self.unweighted_ig)
        edge_reliability = int(cut_value)
        if not math.isclose(nx_version, edge_reliability):
            print(
                f"{bcolors.WARNING}Warning: Discrepancy between NetworkX edge connectivity ({nx_version}) and Rustworkx Stoer-Wagner min cut ({edge_reliability}){bcolors.ENDC}"
            )
            return int(edge_reliability)
        else:
            print(
                f"{bcolors.OKGREEN}Info: Agreement between NetworkX edge connectivity ({nx_version}) and Rustworkx Stoer-Wagner min cut ({edge_reliability}) BTW, its probably a good moment to remove nx now{bcolors.ENDC}"
            )
        return edge_reliability

    @cached_property
    def ig_coreness(self) -> dict[int, int]:
        """maximal $k$ for specific node $i$ such that $i$ is present in $k$-core graph but removed from $(k + 1)$-core (k-core is a subgraph of some graph made by removing all the nodes of degree <= k).

        Returns:
            A dictionary mapping node indices to their core numbers.
        """
        return rx.core_number(self.unweighted_ig)

    @property
    def ig_avg_coreness(self) -> float:
        """A k-core is a maximal subgraph in which each node has at least degree k. The core number of a node is the largest k for which the node is in the k-core.
        Returns:
            Average core number across all nodes in the IG.
        """
        coreness_values = np.fromiter(self.ig_coreness.values(), dtype=float)
        return float(coreness_values.mean())

    @property
    def ig_std_coreness(self) -> float:
        """
        Returns:
            Standard deviation of core numbers across all nodes in the IG.
        """
        coreness_values = np.fromiter(self.ig_coreness.values(), dtype=float)
        return float(coreness_values.std())

    @property
    def ig_max_coreness(self) -> int:
        """
        Returns:
            Maximum core number across all nodes in the IG.
        """
        return int(max(self.ig_coreness.values()))

    @property
    def ig_min_coreness(self) -> int:
        """
        Returns:
            Minimum core number across all nodes in the IG.
        """
        return int(min(self.ig_coreness.values()))

    @cached_property
    def ig_pageranks(self) -> np.ndarray:
        """
        Returns:
            The node-wise PageRank vector of the interaction graph. (Uses alpha=0.85)
        """
        Gdir = self.ig.to_directed()
        pr = rx.pagerank(Gdir, alpha=0.85)
        vec = np.zeros(Gdir.num_nodes(), dtype=float)
        for idx, score in pr.items():
            vec[idx] = score
        return vec

    @property
    def ig_max_pagerank(self) -> float:
        """
        Returns:
            The maximum PageRank score among all nodes in the interaction graph.
        """
        return float(max(self.ig_pageranks))

    @property
    def ig_min_pagerank(self) -> float:
        """
        Returns:
            The minimum PageRank score among all nodes in the interaction graph.
        """
        return float(min(self.ig_pageranks))

    @property
    def ig_std_pagerank(self) -> float:
        """
        Returns:
            The standard deviation of the PageRank scores among all nodes in the interaction graph.
        """
        return float(np.std(self.ig_pageranks))

    @property
    def ig_normalized_hhi_pagerank(self) -> float:
        """Normalized Herfindahl-Hirschman Index (HHI) of the PageRank vector.

        Returns:
            A float value between 0 and 1 indicating the concentration of PageRank scores.
        """
        p = self.ig_pageranks
        N = p.size
        if N <= 1:
            return 0.0
        hhi = float(np.sum(p * p))
        return (hhi - 1.0 / N) / (1.0 - 1.0 / N)

    ### gate dependency graph metrics

    @property
    def dag_qiskit(self):
        return circuit_to_dag(self.circuit)

    @cached_property
    def dag_graph(self) -> rx.PyDiGraph:
        return self._to_rustworkx(self.dag_qiskit)

    @cached_property
    def _gdg_metrics(self) -> dict:
        """cached helper to compute all GDG metrics in one pass as we need to traverse the GDG only once."""
        return self._compute_gdg_metrics()

    @property
    def gdg_critical_path_length(self) -> int:
        """Note that under the assumption of unit weights for all gates, the critical path length in the GDG corresponds to the circuit depth-1 (edges vs gates).

        Returns:
            The length of the critical path in the gate dependency graph (DAG).
        """
        return self._gdg_metrics.get("critical_path_length", 0)

    @property
    def gdg_num_critical_paths(self) -> int:
        """
        Returns:
            The number of critical paths in the gate dependency graph (DAG).
        """
        return self._gdg_metrics.get("num_critical_paths", 0)

    @property
    def gdg_mean_path_length(self) -> float:
        """
        Returns:
            The mean of all path lengths in the GDG.
        """
        mean = self._gdg_metrics.get("path_length_mean", 0.0)
        return mean

    @property
    def gdg_std_path_length(self) -> float:
        """
        Returns:
            The standard deviation of all path lengths in the GDG.
        """
        return self._gdg_metrics.get("path_length_std", 0.0)

    @property
    def gdg_percentage_gates_in_critical_path(self) -> float:
        """
        Returns:
            The percentage of gates that are on the critical path in the GDG.
        """
        # since its edges -> +1 node
        gates_on_cp = (
            self.gdg_critical_path_length + 1
            if self.gdg_critical_path_length > 0
            else 0
        )
        denom = max(self.num_gates, 1)
        return min(1.0, gates_on_cp / denom)

    ### circuit density metrics

    @property
    def density_score(self) -> float:
        r"""Density score as defined in Bandic et al. 2025:
        \[\mathcal{D} = \frac{\frac{2 \cdot n_{2qg} + n_{1qg}}{d} - 1}{n_{qubits} - 1}\]

        Returns:
            Density score as defined in Bandic et al. 2025.
        """
        return (((2 * self.num_2q_gates + self.num_1q_gates) / self.depth) - 1) / (
            self.num_qubits - 1
        )

    @property
    def idling_score(self) -> float:
        r"""Idling score as defined in Bandic et al. 2025:
        \[ \mathcal{I} = \frac{n_{qubits} \otimes d - \sum_{i=1}^{d} q_i}{n_{qubits} \otimes d}, \]
        where $q_i$ is the number of active qubits in moment $i$.

        Returns:
            Idling score as defined in Bandic et al. 2025.
        """
        # sum_qi is the total number of active qubit-moments
        sum_qi = self._compute_active_qubit_layers()

        numerator = (self.num_qubits * self.depth) - sum_qi
        denominator = self.num_qubits * self.depth

        return numerator / denominator if denominator > 0 else 0.0

    # other -> e.g. gate sampling
    @property
    def node_1q_counts(self) -> np.ndarray:
        return np.diag(self.ig_adj_mat)

    @property
    def edge_2q_counts(self) -> dict[tuple[int, int], int]:
        return self._compute_edge_2q_counts()

    @property
    def one_qubit_gates(self) -> list[tuple]:
        return self._get_xq_gates(1)

    @property
    def two_qubit_gates(self) -> list[tuple]:
        return self._get_xq_gates(2)

    @property
    def other_gates(self) -> list[tuple]:
        return [
            (g.qubits, g.operation, g.clbits)
            for g in self.circuit.data
            if len(g.qubits) not in (1, 2)
        ]

    # private helpers

    @staticmethod
    def _filter_gates_by_name(qc: QuantumCircuit, drop_names: set[str]):
        pm = PassManager([FilterOpNodes(lambda node: node.op.name not in drop_names)])
        return pm.run(qc.copy())

    def _get_xq_gates(self, x: int, greater_equals: bool = False) -> list[tuple]:
        out = []
        for gate in self.circuit.data:
            if len(gate.qubits) == x or (len(gate.qubits) >= x and greater_equals):
                out.append((gate.qubits, gate.operation, gate.clbits))
        return out

    def _compute_adj_mat(self) -> np.ndarray:
        adj_matrix = np.zeros((self.n_qubits, self.n_qubits), dtype=np.int64)
        for gate in self.circuit.data:
            qubits = gate.qubits
            if not qubits:
                raise ValueError("Found gate with no qubits -> unexpected case.")
            if len(qubits) == 1:
                q = self.circuit.find_bit(qubits[0]).index
                adj_matrix[q, q] += 1
            elif len(qubits) >= 2:
                for q in qubits:
                    for r in qubits:
                        if q == r:
                            continue
                        q_idx = self.circuit.find_bit(q).index
                        r_idx = self.circuit.find_bit(r).index
                        adj_matrix[q_idx, r_idx] += 1
            else:
                print(
                    bcolors.WARNING
                    + "Warning: [Adjacency Matrix] Found gate with no qubits assigned, ignoring."
                    + bcolors.ENDC
                )
                continue  # ignoring gates with no qubits
        return adj_matrix

    @staticmethod
    def _to_rustworkx(dag: DAGCircuit) -> rx.PyDiGraph:
        """
        copys a Qiskit DAGCircuit into a rustworkx.PyDiGraph.

        - Node payloads: the original DAG nodes (DAGOpNode/DAGInNode/DAGOutNode)
        - Edge payloads: the wire object (Qubit/Clbit) for that dependency edge
        - Parallel edges: preserved (one per wire)
        """
        G = rx.PyDiGraph()
        index_of = {}

        # payload is the DAG node itself
        for node in dag.nodes():
            index_of[node] = G.add_node(node)

        # payload is the wire, multiedges OK
        for src, dst, wire in dag.edges():
            G.add_edge(index_of[src], index_of[dst], wire)
        return G

    def _compute_interaction_graph(self, weighted: bool = True) -> rx.PyGraph:
        G = rx.PyGraph()
        # adding a node per qubit
        G.add_nodes_from(range(self.n_qubits))
        # adding edges
        if weighted:
            # edges with weights -> number of 2q gates between qubits
            G.add_edges_from([(i, j, w) for (i, j), w in self.edge_2q_counts.items()])
            # node weights -> number of 1q gates on each qubit
            for i, w in enumerate(self.node_1q_counts):
                G[i] = int(w)
        else:
            G.add_edges_from_no_data([(i, j) for (i, j) in self.edge_2q_counts.keys()])
        return G

    def _compute_edge_2q_counts(self) -> dict[tuple[int, int], int]:
        A = self.ig_adj_mat
        iu, ju = np.triu_indices_from(A, k=1)
        w = A[iu, ju]
        nonzero = w > 0
        return {
            (int(i), int(j)): int(w_ij)
            for i, j, w_ij in zip(iu[nonzero], ju[nonzero], w[nonzero])
        }

    def _compute_active_qubit_layers(self) -> int:
        """Helper for idling_score. Counts total active qubit-moments."""
        total_active_qubits = 0
        for layer in self.dag_qiskit.layers():
            layer_qubits = set()
            for node in layer["graph"].op_nodes():
                for q in node.qargs:
                    layer_qubits.add(self.circuit.find_bit(q).index)
            total_active_qubits += len(layer_qubits)
        return total_active_qubits

    @staticmethod
    def _logsumexp(vals):
        if not vals:
            return float("-inf")
        a = max(vals)
        if a == float("-inf"):
            return float("-inf")
        return a + math.log(sum(math.exp(v - a) for v in vals))

    def _compute_gdg_per_node_metrics(self, H: rx.PyDiGraph):
        nodes_in_rev_topo_order = list(rx.topological_sort(H))[::-1]

        # L: Longest path length from node to a sink (for critical path calcs)
        # N: Number of critical paths from node to a sink
        # log_n: Log of total number of paths from node to a sink -> stable for large counts
        # m: Mean length of paths from node to a sink
        # v: Variance of the length of paths from node to a sink
        L, N, log_n, m, v = {}, {}, {}, {}, {}

        for w in nodes_in_rev_topo_order:
            succ = H.successor_indices(w)

            if not succ:  # sink
                L[w], N[w], log_n[w], m[w], v[w] = 0, 1, 0.0, 0.0, 0.0
                continue

            # Longest path length and #critical paths
            L[w] = 1 + max(L[u] for u in succ)
            N[w] = sum(N[u] for u in succ if L[u] == L[w] - 1) or 1

            # log counts and probs
            logZ = self._logsumexp([log_n[u] for u in succ])
            probs = [math.exp(log_n[u] - logZ) for u in succ]  # stable, sums to 1

            # first + second moments of path length
            mu_terms = [p * (1.0 + m[u]) for p, u in zip(probs, succ)]
            mu = sum(mu_terms)
            m[w] = mu

            second_terms = [
                p * (((1.0 + m[u]) ** 2) + v[u]) for p, u in zip(probs, succ)
            ]
            second = sum(second_terms)
            v[w] = max(0.0, second - mu * mu)

            # Total path count in log-space: n[w] = sum n[u]  ->  log_n[w] = logsumexp(log_n[u])
            log_n[w] = logZ

        return L, N, log_n, m, v

    def _compute_gdg_metrics(self) -> dict:
        """
        Calculates all GDG metrics by operating on a dependency graph of only quantum gates.

        Returns:
            A dictionary containing the final summary of GDG metrics.
        """

        # graph with gate operations only
        op_node_indices = [
            n
            for n in self.dag_graph.node_indices()
            if isinstance(self.dag_graph[n], DAGOpNode)
        ]
        H = self.dag_graph.subgraph(op_node_indices)

        if H.num_nodes() == 0:
            return {
                "critical_path_length": 0,
                "num_critical_paths": 0,
                "path_length_mean": 0.0,
                "path_length_std": 0.0,
            }

        L, N, log_n, m, v = self._compute_gdg_per_node_metrics(H)

        critical_path_length = rx.dag_longest_path_length(H)

        sources = [node for node in H.node_indices() if H.in_degree(node) == 0]

        # number of critical paths that originate at sources on a global critical path
        critical_sources = [s for s in sources if L.get(s) == critical_path_length]
        num_critical_paths = sum(N.get(s, 0) for s in critical_sources)

        if not sources:
            mean_len, std_len = 0.0, 0.0
        else:
            # logZ = log sum(s) n[s]
            logZ = self._logsumexp([log_n.get(s, float("-inf")) for s in sources])

            if logZ == float("-inf"):  # no paths found
                mean_len, std_len = 0.0, 0.0
            else:
                # p_s ∝ n[s]  (stable exp-normalize)
                probs = [math.exp(log_n[s] - logZ) for s in sources]

                # mixture mean: E = Σ_s p_s * m[s]
                mean_len = sum(p * m[s] for p, s in zip(probs, sources))

                # mixture second moment: Σ_s p_s * (v[s] + m[s]^2)
                second = sum(p * (v[s] + m[s] ** 2) for p, s in zip(probs, sources))

                var_len = max(0.0, second - mean_len**2)  # numerical safety
                std_len = float(var_len**0.5)

        return {
            "critical_path_length": critical_path_length,
            "num_critical_paths": num_critical_paths,
            "path_length_mean": mean_len,
            "path_length_std": std_len,
        }

    def plot_gdg(self, filename: str = None, **kwargs):
        """Plots the GDG / DAG of the circuit using Qiskit's built-in dag_drawer function.

        Args:
            filename (str, optional): The path to save the DAG plot image. If None, the plot is not saved to a file instead the PIL figure is returned. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the dag_drawer function.

        Returns:
            If filename is None, returns a PIL Image object of the DAG plot. Otherwise, saves the plot to the specified file and returns None.
        """

        fig = dag_drawer(self.dag_qiskit, filename=filename, **kwargs)

        if filename is None:
            return fig

    def plot_ig(
        self,
        filename: str = None,
        weighted: bool = True,
        *,
        layout: str = "spring",
        **kwargs,
    ):
        """Plots the interaction graph (IG) of the circuit using rustworkx's matplotlib visualization.

        Args:
            filename (str, optional): The path to save the IG plot image. If None, the plot is displayed instead of being saved to a file. Defaults to None.
            weighted (bool, optional): Whether to use the weighted interaction graph. Defaults to True.
            layout (str, optional): The layout algorithm to use for the plot. Defaults to "spring". Options are "spring", "circular", "random".
            with_labels (bool, optional): Whether to display labels on the plot. Defaults to True.
            **kwargs: Additional keyword arguments to customize the plot.

        Returns:
            None. Displays the plot or saves it to a file if 'filename' is provided
        """
        import matplotlib.pyplot as plt
        from rustworkx.visualization import mpl_draw

        G: rx.PyGraph = getattr(self, "ig" if weighted else "unweighted_ig")

        if layout == "spring":
            pos = rx.spring_layout(G, seed=42)
        elif layout == "circular":
            pos = rx.circular_layout(G)
        elif layout == "random":
            pos = rx.random_layout(G, seed=42)
        else:
            raise ValueError(f"Unknown layout '{layout}'")

        draw_kwargs = dict(pos=pos, with_labels=not weighted, **kwargs)
        if weighted:
            draw_kwargs["edge_labels"] = str  # weights on edges
        mpl_draw(G, **draw_kwargs)

        counts = list(self.node_1q_counts)

        the_ax = plt.gca()
        if weighted:
            for i in G.node_indices():
                x, y = pos[i]
                the_ax.text(
                    x, y, f"{i}: {int(counts[i])}", ha="center", va="center", fontsize=8
                )
        if filename:
            plt.gcf().savefig(filename, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def __str__(self):
        return f"Circuit(name={self.name}, n_qubits={self.n_qubits}, depth={self.depth}, size={self.size})"

    # def is_cached(self, prop_name: str) -> bool:
    #     return prop_name in self.__dict__

    # def __getattr__(self, name):
    #     if hasattr(self.circuit, name):
    #         return getattr(self.circuit, name)
    #     else:
    #         raise AttributeError(f"'Circuit' object has no attribute '{name}'")

    # def __dir__(self):
    #     # Better tab complete: combine attrs with the wrapped circuits
    #     own = set(super().__dir__())
    #     circuit = object.__getattribute__(self, "circuit")
    #     return sorted(own | set(dir(circuit)))

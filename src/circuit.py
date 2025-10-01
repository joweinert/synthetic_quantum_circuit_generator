from __future__ import annotations
from pathlib import Path
import numpy as np
import networkx as nx
from qiskit.circuit.random import random_circuit
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from mqt.bench import BenchmarkLevel, get_benchmark
from mqt.bench.benchmarks import get_benchmark_catalog
from mqt.bench.targets import (
    get_target_for_gateset,
    get_device,
    get_available_device_names,
    get_available_gateset_names,
)


# ADD QUBIT DEPENDENCY ANALYSIS!
# -> normilaztion
# find a way of comparing different objective functions results


CIRCUIT_PATH = Path("../assets/circuits/MQTBench/")


class Circuit:

    def __init__(
        self,
        circuit: QuantumCircuit,
        n_qubits: int,
        *,
        name: str = "wrapped",
        desc: str = "No description provided",
        # filter_gates: list[str] = ["measure", "reset", "barrier"],TODO
    ):
        self.circuit = circuit
        self.n_qubits = n_qubits
        self.name = name
        self.desc = desc

        self._rng = np.random.default_rng(0)

        self._depth = None
        self._per_gate_count = None
        self._size = None
        self._adj_mat = None
        self._interaction_graph = None
        self._dag_qiskit = None
        self._dag_graph = None
        self._degrees = None
        self._node_1q_counts = None
        self._edge_2q_counts = None
        self._total_1q = None
        self._total_2q = None
        self._2qubit_gates = None
        self._1qubit_gates = None
        self._other_gates = None

    @classmethod
    def from_mqt_bench(
        cls, name: str, n_qubits: int, *, desc: str = None, **mqt_kwargs
    ) -> Circuit:
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
        )

    @classmethod
    def from_random_circuit(
        cls, n_qubits: int, name: str = "random", *, desc: str = None, **random_kwargs
    ) -> Circuit:
        qc = random_circuit(n_qubits, **random_kwargs)
        return cls(
            qc,
            name=name,
            n_qubits=n_qubits,
            desc=desc
            or f"Randomly generated circuit using Qiskit's random_circuit function. n_qubits={n_qubits}, "
            + ", ".join(f"{k}={v}" for k, v in random_kwargs.items()),
        )

    @classmethod
    def from_qiskit(
        cls,
        qc: QuantumCircuit,
        *,
        name: str = "wrapped",
        desc: str = "No description provided",
    ) -> Circuit:
        return cls(qc, n_qubits=qc.num_qubits, name=name, desc=desc)

    @property
    def depth(self) -> int:
        if self._depth is None:
            self._depth = self.circuit.depth()
        return self._depth

    @property
    def per_gate_count(self) -> dict[str, int]:
        if self._per_gate_count is None:
            self._per_gate_count = self.circuit.count_ops()
        return self._per_gate_count

    @property
    def size(self) -> int:
        if self._size is None:
            self._size = sum(self.per_gate_count.values())
        return self._size

    @property
    def adj_mat(self) -> np.ndarray:
        if self._adj_mat is None:
            self._adj_mat = self._compute_adj_matrix()
        return self._adj_mat

    @property
    def dag_qiskit(self):
        if self._dag_qiskit is None:
            self._dag_qiskit = circuit_to_dag(
                self.circuit
            )  # DAG representation of the circuit
        return self._dag_qiskit

    @property
    def dag_graph(self) -> nx.DiGraph:
        if self._dag_graph is None:
            self._dag_graph = self.dag_qiskit.to_networkx()
        return self._dag_graph

    @property
    def interaction_graph(self) -> nx.Graph:
        if self._interaction_graph is None:
            self._interaction_graph = self._compute_interaction_graph()
        return self._interaction_graph

    @property
    def node_1q_counts(self) -> np.ndarray:
        if self._node_1q_counts is None:
            self._node_1q_counts = np.diag(self.adj_mat)
        return self._node_1q_counts

    @property
    def edge_2q_counts(self) -> dict[tuple[int, int], int]:
        if self._edge_2q_counts is None:
            self._edge_2q_counts = self._compute_edge_2q_counts()
        return self._edge_2q_counts

    @property
    def degrees(self) -> np.ndarray:
        if self._degrees is None:
            self._degrees = self.adj_mat.sum(axis=1) - np.diag(self.adj_mat)
        return self._degrees

    @property
    def total_1q(self) -> int:
        if self._total_1q is None:
            self._total_1q = int(np.trace(self.adj_mat))
        return self._total_1q

    @property
    def total_2q(self) -> int:
        if self._total_2q is None:
            off = self.adj_mat.sum() - np.trace(self.adj_mat)
            self._total_2q = int(off // 2)
        return self._total_2q

    def sample_2q_gate(self, rng=None) -> tuple:
        """Samples a 2-qubit gate (qubits, operation, clbits) according to the distribution of 2-qubit gates in the circuit.

        Returns:
            tuple: A tuple containing the qubits, operation and and (possibly empty) list of clbits.
        """
        if self._2qubit_gates is None:
            self._2qubit_gates = self._get_xq_gates(2)

        rng = self._rng if rng is None else rng
        i = rng.integers(0, len(self._2qubit_gates))
        return self._2qubit_gates[i]

    def sample_1q_gate(self, rng=None) -> tuple:
        """Samples a 1-qubit gate (qubits, operation, clbits) according to the distribution of 1-qubit gates in the circuit.

        Returns:
            tuple: A tuple containing the qubits, operation and and (possibly empty) list of clbits.
        """
        if self._1qubit_gates is None:
            self._1qubit_gates = self._get_xq_gates(1)

        rng = self._rng if rng is None else rng
        i = rng.integers(0, len(self._1qubit_gates))
        return self._1qubit_gates[i]

    def sample_gt2q_gate(self, rng=None) -> tuple:
        """Samples a gate with 3 or more qubits (qubits, operation, clbits) according to the distribution of such gates in the circuit.

        Returns:
            tuple: A tuple containing the qubits, operation and and (possibly empty) list of clbits.
        """
        if self._other_gates is None:
            self._other_gates = self._get_xq_gates(3, greater_equals=True)

        rng = self._rng if rng is None else rng
        i = rng.integers(0, len(self._other_gates))
        return self._other_gates[i]

    def _get_xq_gates(self, x: int, greater_equals: bool = False) -> list[tuple]:
        out = []
        for gate in self.circuit.data:
            if len(gate.qubits) == x or (len(gate.qubits) >= x and greater_equals):
                out.append((gate.qubits, gate.operation, gate.clbits))
        return out

    def clear_cached(self, attributes: list[str] = None):
        if attributes is None:
            attributes = [
                "depth",
                "per_gate_count",
                "size",
                "adj_mat",
                "interaction_graph",
                "dag_qiskit",
                "dag_graph",
                "degrees",
                "node_1q_counts",
                "edge_2q_counts",
                "total_1q",
                "total_2q",
            ]
        for attr in attributes:
            if not attr.startswith("_"):
                attr = "_" + attr
            if hasattr(self, attr):
                setattr(self, attr, None)

    def _compute_adj_matrix(self) -> np.ndarray:
        """Computes the adjacency matrix of the circuit.

        Returns:
            np.ndarray: The adjacency matrix of the circuit.
        """
        adj_matrix = np.zeros((self.n_qubits, self.n_qubits), dtype=np.int64)

        for gate in self.circuit.data:
            qubits = gate.qubits

            if len(qubits) > 2:
                raise ValueError(
                    "Found >2 qubit gate -> transpile to 1/2-qubit gates first."
                )
            elif len(qubits) < 1:
                raise ValueError("Found gate with no qubits -> unexpected case.")

            elif (
                len(qubits) == 1
                or self.circuit.find_bit(qubits[0]).index
                == self.circuit.find_bit(qubits[1]).index
            ):
                q = self.circuit.find_bit(qubits[0]).index
                adj_matrix[q, q] += 1

            elif len(qubits) == 2:
                q1 = self.circuit.find_bit(qubits[0]).index
                q2 = self.circuit.find_bit(qubits[1]).index
                adj_matrix[q1, q2] += 1
                adj_matrix[q2, q1] += 1

        return adj_matrix

    def _compute_interaction_graph(self) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(self.n_qubits))
        for (i, j), w in self.edge_2q_counts.items():
            G.add_edge(i, j, weight=w)
        # 1Quants counts as node attribute
        for i, w in enumerate(self.node_1q_counts):
            G.nodes[i]["w_1q"] = int(w)
        return G

    def _compute_edge_2q_counts(self) -> dict[tuple[int, int], int]:
        A = self.adj_mat
        out = {}
        for i in range(A.shape[0]):
            for j in range(i + 1, A.shape[1]):
                if A[i, j]:
                    out[(i, j)] = int(A[i, j])
        return out

    def __str__(self):
        return f"Circuit(name={self.name}, n_qubits={self.n_qubits}, depth={self.depth}, size={self.size})"

    def __getattr__(self, name):
        # this delegates unknown attributes and methods to the inner QuantumCircuit -> allows to call for stuff like circuit.count_ops() directly on Circuit instance
        return getattr(self.circuit, name)

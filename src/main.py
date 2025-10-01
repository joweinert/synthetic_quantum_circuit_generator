from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional
from circuit import Circuit
from generator import QuantumCircuitGenerator
from utils import bcolors
import os
import pickle

from mqt.bench import BenchmarkLevel
from mqt.bench.benchmarks import get_benchmark_catalog


def prep_qubit_range(
    name: str, n_vals: int = 8, min_qubits: int = 2, max_qubits: int = 130
) -> List[int]:
    """Prepare qubit ranges for different benchmark circuits."""
    assert n_vals > 0, "n_vals must be positive"
    assert min_qubits > 1, "min_qubits must be at least 2"
    assert (
        max_qubits >= min_qubits and max_qubits <= 130
    ), "max_qubits must be >= min_qubits and <= 130"

    RULES: Dict[str, Callable[[int], bool]] = {
        "shor": lambda n: n in {18, 42, 58, 74},
        "half_adder": lambda n: (n % 2 == 1) and (n >= 3),
        "full_adder": lambda n: (n % 2 == 0) and (n >= 4),
        "modular_adder": lambda n: (n % 2 == 0),
        "multiplier": lambda n: (n % 4 == 0),
        "rg_qft_multiplier": lambda n: (n % 4 == 0) and (n >= 4),
        "cdkm_ripple_carry_adder": lambda n: (n % 2 == 0) and (n >= 4),
        "hrs_cumulative_multiplier": lambda n: ((n - 1) % 4 == 0) and (n >= 5),
        "draper_qft_adder": lambda n: (n % 2 == 0),
        "bmw_quark_copula": lambda n: (n % 2 == 0),
        "hhl": lambda n: (n >= 3),
        "qwalk": lambda n: (n >= 3),
        "graphstate": lambda n: (n >= 3),
        "vbe_ripple_carry_adder": lambda n: ((n - 1) % 3 == 0) and (n >= 4),
        "grover": lambda n: (n <= 50),
    }

    candidates = [n for n in range(min_qubits, max_qubits + 1)]
    pred = RULES.get(name)
    if pred:
        candidates = [n for n in candidates if pred(n)]

    if not candidates:
        print(
            bcolors.WARNING
            + f"Warning: No valid qubit counts for circuit {name} in range [{min_qubits}, {max_qubits}]."
            + bcolors.ENDC
        )
        return []

    if len(candidates) > n_vals:
        idxs = {round(i * (len(candidates) - 1) / (n_vals - 1)) for i in range(n_vals)}
        candidates = [candidates[i] for i in sorted(idxs)]

    return candidates


def create_circuit_instance(
    name: str, n_qubits: int, level: BenchmarkLevel = BenchmarkLevel.NATIVEGATES
) -> Optional[Circuit]:
    try:
        circuit = Circuit.from_mqt_bench(name, n_qubits, level=level)
        return circuit
    except Exception as e:
        print(
            bcolors.FAIL
            + f"Error creating circuit {name} with {n_qubits} qubits: {e}"
            + bcolors.ENDC
        )
        return None


def instantiate_circuits(
    circuit_names: Optional[Iterable[str]] = None,
    except_names: Optional[Iterable[str]] = None,
    n_vals: int = 8,
    min_qubits: int = 2,
    max_qubits: int = 130,
    level: BenchmarkLevel = BenchmarkLevel.NATIVEGATES,
) -> Dict[str, List[int]]:

    if circuit_names is None:
        circuit_names = list(get_benchmark_catalog().keys())

    if except_names is not None:
        circuit_names = [name for name in circuit_names if name not in except_names]

    circuits: Dict[str, List[Circuit]] = {circuit: [] for circuit in circuit_names}
    for circuit in circuit_names:
        n_qubit_list = prep_qubit_range(circuit, n_vals, min_qubits, max_qubits)

        for n in n_qubit_list:
            c = create_circuit_instance(circuit, n, level)
            if c is None:
                continue
            print(f"Created circuit {circuit} with {n} qubits.")
            circuits[circuit].append(c)

    return circuits


def create_dataset(
    circuits: Dict[str, List[Circuit]], out_dir: str = "../data/generated"
) -> None:
    pass
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)

    # for name, circuit_list in circuits.items():
    #     for circuit in circuit_list:
    #         filename = (
    #             f"{out_dir}/{name}_{circuit.n_qubits}q_{circuit.mqt_level.name}.pkl"
    #         )
    #         with open(filename, "wb") as f:
    #             pickle.dump(circuit, f)
    #         print(f"Saved circuit {name} with {circuit.n_qubits} qubits to {filename}")


if __name__ == "__main__":
    circuits = instantiate_circuits(
        # takes kinda long (ae, shor)
        # grover: takes long for nqubit 50, more even: Error creating circuit grover with 100 qubits: loop of ufunc does not support argument 0 of type int which has no callable sqrt method
        # multiplier: Error creating circuit multiplier with x qubits: 'HighLevelSynthesis is unable to synthesize "measure"'
        except_names=["ae", "shor", "multiplier", "grover"],
        n_vals=2,
        min_qubits=2,
        max_qubits=130,
        level=BenchmarkLevel.NATIVEGATES,
    )
    dataset = create_dataset(circuits)

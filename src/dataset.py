import os
import gc
import pandas as pd
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional
from src.circuit import Circuit, to_flattened_feature_dict
from src.utils import bcolors

from mqt.bench import BenchmarkLevel
from mqt.bench.benchmarks import get_benchmark_catalog


INDEX = ["name", "num_qubits"]


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


def collect_circuit_data(
    circuits: Dict[str, List[Circuit]],
    dict_fn: Callable[[Circuit], Dict[str, Any]] = to_flattened_feature_dict,
    del_instances: bool = False,
) -> pd.DataFrame:
    rows = []

    for name, clist in circuits.items():
        if not del_instances:
            clist = clist.copy()
        while clist:
            c = clist.pop()
            print(f"Processing circuit {name} with {c.n_qubits} qubits.")
            try:
                feats = dict_fn(c)
                feats["name"] = name
                rows.append(feats)
            finally:
                if del_instances:
                    del c
                    gc.collect()

    return pd.DataFrame(rows).set_index(INDEX)


def create_append_or_update_dataset(
    new_data: pd.DataFrame,
    out_dir: str = "./data/generated",
    file_name: str = "circuits",
    save_csv: bool = True,
) -> pd.DataFrame:
    if file_name.endswith(".csv"):
        file_name = file_name[:-4]
    path = f"{out_dir}/{file_name}.csv"
    if not os.path.exists(path):
        if save_csv:
            os.makedirs(out_dir, exist_ok=True)
            new_data.to_csv(path, index=True, index_label=INDEX)
        else:
            print(
                bcolors.WARNING
                + f"Warning: File {path} does not exist and save_csv is False. Returning new data without saving. (No action taken.)"
                + bcolors.ENDC
            )
        return new_data

    existing = pd.read_csv(path, index_col=INDEX)

    # union of rows/cols -> keeps existing values
    combined = existing.combine_first(new_data)

    # updates existing rows with new values
    combined.update(new_data)

    # bring back orig column order + add any new columns at the end
    desired_cols = list(existing.columns) + [
        c for c in new_data.columns if c not in existing.columns
    ]
    combined = combined.reindex(columns=desired_cols)

    if save_csv:
        if path.endswith(".csv"):
            path = path[:-4]
        combined.to_csv(
            path + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv",
            index=True,
            index_label=INDEX,
        )

    return combined

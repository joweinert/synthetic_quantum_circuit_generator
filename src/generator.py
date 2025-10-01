from typing import Protocol, List
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction
import math
from utils import bcolors
from tqdm import tqdm

from circuit import Circuit


class ObjectiveFunction(Protocol):
    name: str
    requires: set[str]  # feature keys this objective needs
    initial_loss: float = -1.0  # set to -1.0 to indicate uninitialized

    # should return normalized loss -> divided by initial loss
    def score(self, cand_circuit: Circuit, target_feats: dict) -> float: ...

    def add_required_features(
        self, circuit: Circuit, target_feats: dict | None = None
    ) -> dict: ...


class DepthObjectiveFunction:

    name = "depth"
    requires = {"depth"}
    initial_loss = -1.0

    def score(self, cand_circuit: Circuit, target_feats: dict) -> float:
        loss = abs(cand_circuit.depth - target_feats["depth"])
        if self.initial_loss < 0:
            self.initial_loss = max(loss, 1e-6)  # to avoid division by zero
        return loss / self.initial_loss

    def add_required_features(
        self, circuit: Circuit, target_feats: dict | None = None
    ) -> dict:
        if target_feats is None:
            target_feats = {}
        if "depth" not in target_feats:
            target_feats["depth"] = circuit.depth
        return target_feats


# class GatePerSliceObjective:
#     name = "gps"
#     requires = {"slice_hist"}  # implement add_required_features to compute target hist

#     def score(self, cand, target):
#         h_s = slice_hist(cand)
#         h_t = target["slice_hist"]
#         # pad to same length
#         L = max(len(h_s), len(h_t))
#         hs = np.pad(h_s, (0, L - len(h_s)))
#         ht = np.pad(h_t, (0, L - len(h_t)))
#         return float(np.sum(np.abs(hs - ht)))


class QuantumCircuitGenerator:

    def __init__(
        self,
        op_min: float = 0.2,
        op_ratio: float = 0.1,  # REDEFINED: ratio to increase/decrease gates_to_reorder, if acceptance rate is too low (< 20%) / high (> 80%) -> descrease/increase gates_to_reorder respectively. 0.1 -> 10% descrease/increase
        temp_init: float = None,
        alpha: float = None,
        delta: float = None,
        reorder_ratio_init: float = 0.5,  # initial ratio of gates to reorder
        temp_min: float = 1e-6,
        verbose: bool = False,
        seed: int = 0,
    ):
        self.verbose = verbose
        self.op_min = op_min
        self.op_ratio = op_ratio
        self.temp_init = temp_init
        self.temp_min = temp_min
        self.alpha = alpha
        self.delta = delta
        self.reorder_ratio_init = reorder_ratio_init
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.obj_weights = None
        self.objectives = None
        self.circuit = None
        self.target_feats = None
        self.target_override = None

    def generate_circuits(
        self,
        circuit: Circuit,  # a circuit to mimic
        target_override: dict = None,  # target features to override the ones from the circuit
        objectives: List[ObjectiveFunction] = [
            DepthObjectiveFunction()
        ],  # list of objective function instances to optimize
        obj_weights: list[float] = None,  # weights for each objective function
        n=10,  # number of circuits to generate
        max_iter=np.inf,  # max number of iterations to try per generated circuit
        min_iter=1000,  # minimum number of iterations to try per generated circuit, i.e used for simulated annealing style stopping criterion
        wait_iter=6500,  # number of iterations to wait for improvement before stopping
        seed=None,
    ):
        # checks if inputs are valid and stores them as class attributes

        self._check_inputs(
            circuit=circuit,
            target_override=target_override,
            objectives=objectives,
            obj_weights=obj_weights,
            seed=seed,
        )
        self.target_feats = self._get_gold_standard()
        generated_circuits = []

        self._generate_single_circuit()

        for _ in tqdm(range(n)):
            (
                cand_circuit,
                best_circuit,
                iter,
                last_improved_iter,
                accepted_cnt,
                loss_cache,
                best_loss,
                end_reason,
            ) = (
                None,
                None,
                0,
                0,
                0,
                [],
                float("inf"),
                "max iter",
            )
            n_gates = len(self.circuit.data)
            gates_to_reorder = int(self.reorder_ratio_init * n_gates)
            temp = self.temp_init

            while (iter - last_improved_iter) < wait_iter and iter <= max_iter:
                iter += 1

                if iter % 500 == 0:
                    if temp is not None and self.alpha is not None:
                        temp = max(self.alpha * temp, self.temp_min)
                    if accepted_cnt / 500 < 0.2:
                        gates_to_reorder = max(
                            int(self.op_min * n_gates),
                            int((1 - self.op_ratio) * gates_to_reorder),
                        )
                    elif accepted_cnt / 500 > 0.8:
                        gates_to_reorder = min(
                            n_gates, int((1 + self.op_ratio) * gates_to_reorder)
                        )
                    accepted_cnt = 0

                cand_circuit = self._create_cand_circuit(best_circuit, gates_to_reorder)
                loss = self._calc_loss(cand_circuit)
                loss_cache.append(loss)

                if (
                    iter == 1
                    or (loss < best_loss)
                    or (
                        temp is not None
                        and self.rng.random() < math.exp(-abs(loss - best_loss) / temp)
                    )
                ):
                    best_loss, best_circuit, last_improved_iter = (
                        loss,
                        cand_circuit,
                        iter,
                    )
                    accepted_cnt += 1

                if loss == 0:
                    end_reason = "LOSS = 0"
                    break

                # simulated annealing style stopping criterion
                n = len(loss_cache)
                if self.delta is not None and n > min_iter:
                    mu = float(np.mean(loss_cache))
                    if np.std(loss_cache) / np.sqrt(n) < self.delta * mu:
                        end_reason = f"simulated annealing style stopping criterion met after {iter} iterations"
                        break

            if end_reason == "max iter" and iter < max_iter:
                end_reason = f"no improvement for {wait_iter} iterations"

            generated_circuits.append(
                {
                    "circuit": best_circuit,
                    "loss": best_loss,
                    "found_iter": last_improved_iter,
                    "total_iter": iter,
                    "end_reason": end_reason,
                }
            )

        return generated_circuits[0] if n == 1 else generated_circuits

    def _create_cand_circuit(
        self, state: Circuit = None, gates_to_reorder: int = 0
    ) -> Circuit:
        if state is None:
            m = len(self.circuit.data)
            # we want to smaple m gates with replacement from the original circuit -> only using the qubits involved in the gates
            # -> preserves empirical qubit dependent 1q/2q node/edge distribution
            # -> to ensure variation we sample the operation / params in the next step
            idx = self.rng.integers(0, m, size=m)

            gate_list = []
            for i in idx:
                q = self.circuit.data[i].qubits

                if len(q) == 2:
                    _, operation, clbits = self.circuit.sample_2q_gate(self.rng)
                elif len(q) == 1:
                    _, operation, clbits = self.circuit.sample_1q_gate(self.rng)
                elif len(q) >= 3:
                    _, operation, clbits = self.circuit.sample_gt2q_gate(self.rng)
                    print(
                        bcolors.WARNING
                        + f"Warning: Sampled {operation.name} gate with >= 3 qubits"
                        + bcolors.ENDC
                    )
                else:
                    raise ValueError("Gate with no qubits?")

                gate_list.append(CircuitInstruction(operation.copy(), q, clbits))

            state = self.circuit

        else:
            # reorder gates_to_reorder gates in the current state
            gate_list = self._reorder_gatelist(
                list(state.data.copy()), gates_to_reorder
            )

        qc = self._build_qc_from_gate_list(gate_list, state)
        return Circuit.from_qiskit(qc)

    def _reorder_gatelist(self, gate_list: list, gates_to_reorder: int) -> list:
        if gates_to_reorder <= 0 or gates_to_reorder > len(gate_list):
            print(
                bcolors.WARNING
                + "Warning: gates_to_reorder out of bounds"
                + bcolors.ENDC
            )
            return gate_list

        src_idx = self.rng.choice(
            range(len(gate_list)), size=gates_to_reorder, replace=False
        )
        new_idx = self.rng.choice(
            range(len(gate_list)), size=gates_to_reorder, replace=False
        )
        moved = [gate_list[i] for i in src_idx]
        # self.rng.shuffle(moved)
        placement = dict(zip(new_idx, moved))

        src_set = set(src_idx)
        rest_iter = (g for i, g in enumerate(gate_list) if i not in src_set)

        out = []
        for i in range(len(gate_list)):
            if i in placement:  # all the move items
                out.append(placement[i])
            else:  # original order for non moved
                out.append(next(rest_iter))
        return out

    def _build_qc_from_gate_list(
        self, data: list, src_circuit: Circuit
    ) -> QuantumCircuit:
        qc = QuantumCircuit(src_circuit.num_qubits, src_circuit.num_clbits)

        for item in data:
            # CircuitInstruction instances
            op, qargs, cargs = item.operation, item.qubits, item.clbits
            # Map source Bits -> integer positions
            q_idx = [src_circuit.find_bit(q).index for q in qargs]
            c_idx = [src_circuit.find_bit(c).index for c in cargs] if cargs else []

            qc.append(op, q_idx, c_idx)

        return qc

    def _check_inputs(self, circuit, target_override, objectives, obj_weights, seed):
        if not isinstance(circuit, Circuit):
            raise TypeError("circuit must be an instance of Circuit class.")
        if objectives is None or len(objectives) == 0:
            raise ValueError("At least one objective function must be provided.")
        if obj_weights is None or len(obj_weights) == 0:
            obj_weights = [1.0 / len(objectives) for _ in objectives]
        elif len(obj_weights) != len(objectives):
            raise ValueError("obj_weights must have the same length as objectives.")
        if target_override is not None and not isinstance(target_override, dict):
            raise TypeError("target_override must be a dictionary.")
        else:
            if target_override is not None:
                all_required = set()
                for obj in objectives:
                    all_required.update(obj.requires)
                for key in target_override.keys():
                    if key not in all_required:
                        print(
                            bcolors.WARNING
                            + f"Warning: target_override contains key '{key}' which is not required by any objective and therefore unused for optimization."
                            + bcolors.ENDC
                        )
        if not math.isclose(sum(obj_weights), 1.0):
            print(
                f"{bcolors.WARNING}Warning: obj_weights do not sum to 1. Proceeding anyway.{bcolors.ENDC}"
            )

        self.circuit = circuit
        self.target_override = target_override
        self.objectives = objectives
        self.obj_weights = obj_weights
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    def _calc_loss(
        self,
        cand_circuit: Circuit,
    ) -> float:
        loss = 0.0
        for obj, weight in zip(self.objectives, self.obj_weights):
            loss += weight * obj.score(cand_circuit, self.target_feats)
        return loss

    def _get_gold_standard(self) -> dict:
        target_feats = {}
        if self.target_override is not None:
            target_feats.update(self.target_override)
        for obj in self.objectives:
            target_feats.update(obj.add_required_features(self.circuit, target_feats))

        return target_feats

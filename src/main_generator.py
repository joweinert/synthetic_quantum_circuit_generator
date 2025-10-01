from circuit import Circuit
from generator import QuantumCircuitGenerator


if __name__ == "__main__":

    circuit = Circuit.from_mqt_bench("dj", 100, desc="Deutsch-Jozsa with 100 qubits")
    generator = QuantumCircuitGenerator(
        op_min=0.04, op_ratio=0.5, temp_init=0.01, alpha=0.85, delta=2e-4
    )

    circuit_dict = generator.generate_circuits(circuit, n=1)
    print(circuit_dict)

from src.circuit import Circuit, to_flattened_feature_dict
import numpy as np


TEST_CFG_DJ_5 = {
    "mqt_bench_name": "dj",
    "n_qubits": 5,
    ### SIZE FEATS
    "num_qubits": 5,
    "size": 36,
    "num_gates": 36,
    "num_1q_gates": 32,
    "num_2q_gates": 4,
    "pct_2q_gates": 4 / 36,
    "depth": 11,
    ### IG FEATS
    "ig_aspl": 2 - 2 / 5,  # -> aspl of a star is 2 minus 2 divided by n
    # mean is 4/10, 6 zeros and 4 ones (upper triangle)
    "ig_std_adj_mat": np.sqrt((6 * (0 - 4 / 10) ** 2 + 4 * (1 - 4 / 10) ** 2) / 10),
    "ig_diameter": 2,
    "ig_max_betweenness": 1,  # -> middle node in star (normalized!)
    "ig_avg_degree": (4 + 4 * 1) / 5,
    "ig_max_degree": 4,
    "ig_std_degree": np.sqrt((1 * (4 - 8 / 5) ** 2 + 4 * (1 - 8 / 5) ** 2) / 5),
    # streangths are same as degrees since all edge weights are 0 or 1
    "ig_avg_strength": (4 + 4 * 1) / 5,
    "ig_max_strength": 4,
    "ig_std_strength": np.sqrt((1 * (4 - 8 / 5) ** 2 + 4 * (1 - 8 / 5) ** 2) / 5),
    # four cliques of size 2 (each edge is a clique) nothing bigger in a star (formnally: largest complete subgraph is an edge)
    "ig_max_cliques_num": 4,
    "ig_max_cliques_size": 2,
    # no triangles -> transitivity 0, avg clustering coeff 0
    "ig_transitivity": 0,
    "ig_avg_clustering_coef": 0,
    # any edge removal disconnects a leaf node, removing the center disconnects all
    "ig_vertex_connectivity": 1,
    "ig_edge_connectivity": 1,
    # coreness of all nodes is 1 in a star
    "ig_avg_coreness": 1,
    "ig_min_coreness": 1,
    "ig_max_coreness": 1,
    "ig_std_coreness": 0,
    # well here I cheated for min max and validate that the other result from taking the 4 leaves with the same value
    "ig_max_pagerank": 0.475674483783958,
    "ig_min_pagerank": 0.131081379054010,
    "ig_std_pagerank": np.sqrt(
        (
            1
            * (0.475674483783958 - (0.475674483783958 + 4 * 0.131081379054010) / 5) ** 2
            + 4
            * (0.131081379054010 - (0.475674483783958 + 4 * 0.131081379054010) / 5) ** 2
        )
        / 5
    ),
    "ig_normalized_hhi_pagerank": (
        4 * (0.131081379054010) ** 2 + (0.475674483783958) ** 2 - 1 / 5
    )
    / (1 - 1 / 5),
    ### GDG FEATS
    "gdg_critical_path_length": 10,
    "gdg_num_critical_paths": 2,
    "gdg_mean_path_length": (2 * (8 + 9 + 9 + 10) + (8 + 8 + 9) + (6 + 7) + 6)
    / 14,  # =8.285714285714286
    "gdg_std_path_length": np.sqrt(
        (
            4 * (8 - 8.285714285714286) ** 2
            + 5 * (9 - 8.285714285714286) ** 2
            + 2 * (10 - 8.285714285714286) ** 2
            + 2 * (6 - 8.285714285714286) ** 2
            + (7 - 8.285714285714286) ** 2
        )
        / 14
    ),
    "gdg_percentage_gates_in_critical_path": (10 + 1) / 36,
    ### DENSITY FEATS
    "density_score": (((2 * 4 + 32) / 11) - 1) / (5 - 1),
    "idling_score": (11 * 5 - (9 + 8 + 9 + 7 + 7)) / (11 * 5),
    # ### REPETITIVE SUBCIRCUIT FEATS
    # "repetitive_subcircuit_count": 2,
    # "repetitive_subcircuit_size": 3,
}

TESTS = [TEST_CFG_DJ_5]


def plot_rustowrkx_dag(c: Circuit):
    import matplotlib.pyplot as plt
    from rustworkx.visualization import mpl_draw
    import rustworkx as rx

    G: rx.PyDiGraph = c._to_rustworkx(c.dag_qiskit)

    mpl_draw(G, with_labels=True)

    plt.show()
    plt.close()


if __name__ == "__main__":
    for test in TESTS:
        c = Circuit.from_mqt_bench(test.pop("mqt_bench_name"), test.pop("n_qubits"))

        features = to_flattened_feature_dict(c)

        print("Testing circuit:", c.name, c.n_qubits)

        c.plot_ig(f"tests/{c.name}_{c.n_qubits}_ig.png")
        c.plot_ig(f"tests/{c.name}_{c.n_qubits}_ig_unweighted.png", weighted=False)
        c.plot_gdg(f"tests/{c.name}_{c.n_qubits}_dag.png")
        plot_rustowrkx_dag(c)

        print(c.ig_adj_mat)
        print(c.ig_adj_mat_2q)

        for key, expected_value in test.items():
            actual_value = features[key]
            try:
                np.testing.assert_allclose(
                    actual_value, expected_value, rtol=0, atol=1e-12
                )
            except AssertionError:
                print(f"Feature {key} expected {expected_value}, got {actual_value}")

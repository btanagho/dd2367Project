from simulator import SparseQuantumSimulator, DenseQuantumSimulator
from gates import Gate, H, X, SWAP,Z,Y,S, T,RX,RZ,RY, CP, CNOT, Toffoli
import numpy as np
import time
import tracemalloc
import matplotlib.pyplot as plt

def run_qft(QuantumSim, n, uniform_init=False, randomize=False):
    # INIT STATE
    def make_init_state(m):
        if uniform_init:
            amp = 1 / np.sqrt(2**m)
            if QuantumSim is SparseQuantumSimulator:
                st = {j: amp + 0.0j for j in range(2**m)}
            elif QuantumSim is DenseQuantumSimulator:
                st = [amp + 0.0j for _ in range(2**m)]
        else:
            if QuantumSim is SparseQuantumSimulator:
                st = {0: 1.0+0.0j}
            elif QuantumSim is DenseQuantumSimulator:
                st = [1.0+0.0j] + [0.0+0.0j for _ in range(2**m-1)]
        return st

    # QFT gates
    def make_qft_gates(m):
        gates = []
        for j in reversed(range(m)):
            gates.append(H(j))
            for k in range(j-1, -1, -1):
                angle = -np.pi / (2 ** (j - k))
                gates.append(CP(k, j, angle))
        for i in range(m // 2):
            gates.append(SWAP(i, m - 1 - i))
        return gates

    def make_random_gates(m, depth=2):
        gates = []
        for _ in range(depth):
            for q in range(m):
                gates.append(H(q))
            for _ in range(m):
                a, b = np.random.choice(m, size=2, replace=False)
                gates.append(CNOT(a, b))
        return gates

    st = make_init_state(n)
    sim = QuantumSim(number_qubits=n, states=st)

    gates = []
    if randomize:
        gates += make_random_gates(n, depth=max(2, n//2))
    gates += make_qft_gates(n)

    # Apply gates
    for g in gates:
        g.apply(sim)

    sim.circ_plot(filename=f"qft_n{n}_uniform{uniform_init}_rand{randomize}")
    return sim

def run_adder(QuantumSim, n):
    def MAJ(a, b, c):
        return [CNOT(c, b), CNOT(c, a), Toffoli(a, b, c)]

    def UMA(a, b, c):
        return [Toffoli(a, b, c), CNOT(c, a), CNOT(a, b)]

    def make_adder_gates(n):
        gates = []
        carry = 2 * n

        gates += MAJ(0, n + 0, carry)
        for i in range(1, n):
            gates += MAJ(i, n + i, n + i - 1)

        for i in reversed(range(1, n)):
            gates += UMA(i, n + i, n + i - 1)
        gates += UMA(0, n + 0, carry)

        return gates

    def make_init_state(m):
        amp = 1 / np.sqrt(2**m)
        
        if QuantumSim is SparseQuantumSimulator:
            st = {j: amp + 0.0j for j in range(2**m)}
        elif QuantumSim is DenseQuantumSimulator:
            st = [amp + 0.0j for i in range(2**m)]
        
        return st

    total_qubits = 2 * n + 1

    st = make_init_state(total_qubits)

    sim = QuantumSim(number_qubits=total_qubits, states=st)

    adder_gates = make_adder_gates(n)
    sim.apply_circuit(adder_gates)

    sim.circ_plot(filename="adder")

    return sim

# Report runtime, memory, and maximum non-zero count

def max_nonzero_count(states):
    # For sparse: states is a dict, for dense: states is a list
    if isinstance(states, dict):
        return len([v for v in states.values() if abs(v) > 1e-12])
    else:
        return len([v for v in states if abs(v) > 1e-12])
    
def Evaluate(Sim_class, n=[8,10,12], circuit_method=None, uniform_init=False, randomize=False):
    results = {m: {"runtime": None, "memory": None, "max_nz_count": None, "density": None} for m in n}

    for m in n:
        print(Sim_class, m)
        tracemalloc.start()
        t0 = time.time()
        sim = circuit_method(Sim_class, m, uniform_init=uniform_init, randomize=randomize)
        t1 = time.time()
        mem = tracemalloc.get_traced_memory()[1] / (1024**2)  # MB
        tracemalloc.stop()

        if isinstance(sim.states, dict):
            nz = len([v for v in sim.states.values() if abs(v) > 1e-12])
        else:
            nz = len([v for v in sim.states if abs(v) > 1e-12])
        density = nz / (2**m)

        results[m] = {"runtime": t1-t0, "memory": mem, "max_nz_count": nz, "density": density}

    return results

import os
import matplotlib.pyplot as plt

def compare_sims(results_dict, save_dir="plots", filename="simulator_comparison.png"):
    """
    Compare simulator performance across metrics and qubit counts.

    Args:
        results_dict (dict): Mapping from simulator name to results dict (from Evaluate_on_QFT).
        save_dir (str): Directory where the plot should be saved.
        filename (str): Output filename for the saved figure.
    """

    # Extract metric names dynamically from first entry 
    sample_sim = next(iter(results_dict.values()))
    metrics = list(next(iter(sample_sim.values())).keys())  

    num_metrics = len(metrics)
    n_cols = min(2, num_metrics)
    n_rows = (num_metrics + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    axs = axs.flatten()

    for i, metric in enumerate(metrics):
        ax = axs[i]
        for sim_name, sim_results in results_dict.items():
            qubits = sorted(sim_results.keys())
            values = [sim_results[q][metric] for q in qubits]
            ax.plot(qubits, values, marker='o', label=sim_name)

        ax.set_xlabel("Number of qubits (n)")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} vs Number of Qubits")
        ax.grid(True)
        ax.legend()

    # Hide empty subplots if metrics < n_rows*n_cols
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()

    # Save the plot 
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")

    # Optionally display
    plt.show()


def main():
    #QFT
    num_qubits = [4, 6, 8, 10]
    res_dense = Evaluate(DenseQuantumSimulator, n=num_qubits, circuit_method=run_qft,
                        uniform_init=True, randomize=True)
    res_sparse = Evaluate(SparseQuantumSimulator, n=num_qubits, circuit_method=run_qft,
                        uniform_init=True, randomize=True)
    print("Dense:", res_dense)
    print("Sparse:", res_sparse)
    
    compare_sims({
        "Dense NumPy Vector": res_dense,
        "Sparse Hashmap": res_sparse
    }, filename="qft_comparison.png")

    #ADDER
    num_qubits = [i*2 for i in range(3)]
    res_dense = Evaluate(DenseQuantumSimulator,n=num_qubits,circuit_method=run_adder)
    res_sparse = Evaluate(SparseQuantumSimulator,n=num_qubits,circuit_method=run_adder)
    print("Dense:", res_dense)
    print("Sparse:", res_sparse)
    
    compare_sims({
        "Dense NumPy Vector": res_dense,
        "Sparse Hashmap": res_sparse
    }, filename="adder_comparison.png")

if __name__=="__main__":
    main()

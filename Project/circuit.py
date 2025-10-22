from simulator import SparseQuantumSimulator, DenseQuantumSimulator
from gates import Gate, H, X, SWAP, CSWAP, Z,Y,S, T,RX,RZ,RY, CP, CNOT, Toffoli
import numpy as np
import time
import tracemalloc
import matplotlib.pyplot as plt
import os

def make_prep_gates(m):
    gates = []
    for i in range(m):
      gates.append(H(i))
    return gates

def make_qft_gates(n):
    # Dense-friendly 
    '''
    Constructs a quantum circuit for the n-qubit Quantum Fourier Transform (QFT).

    The QFT is a key quantum algorithm that transforms a quantum state into its Fourier basis, analogous to the classical discrete Fourier transform but implemented efficiently on a quantum computer.

    The circuit is built as follows:
      - For each qubit (starting from the last), apply a Hadamard gate.
      - Then, for each pair of qubits, apply a series of controlled-phase (CP) gates with decreasing angles (\(-\pi/2, -\pi/4, ...\)), entangling the qubits and encoding phase information.
      - Finally, swap qubits to reverse their order, as the QFT outputs in bit-reversed order.

    The resulting circuit uses \(O(n^2)\) gates and is the standard textbook construction for QFT[1][3][10].

    Args:
        n (int): Number of qubits for the QFT circuit.

    Returns:
        gates (list): List of gate objects (Hadamard, controlled-phase, and SWAP) implementing the QFT.
    '''
    gates = make_prep_gates(n)
    gates.append(CP(1,1,np.pi))

    for j in reversed(range(n)):
        gates.append(H(j))
        for k in range(j-1, -1, -1):
            angle = -np.pi / (2 ** (j - k))
            gates.append(CP(k, j, angle))
    for i in range(n // 2):
        gates.append(SWAP(i, n - 1 - i))
    return gates

def make_ghz_gates(n):
    # Sparse-friendly 
    '''
    Constructs a quantum circuit to prepare an n-qubit GHZ (Greenberger-Horne-Zeilinger) state.

    The GHZ state is a maximally entangled state of the form:
        (|00...0> + |11...1>) / sqrt(2)

    The circuit applies a Hadamard gate to the first qubit to create superposition,
    then a chain of CNOT gates entangles all qubits such that the final state is
    a superposition of all qubits being 0 and all qubits being 1.

    Args:
        n (int): Number of qubits in the GHZ state.

    Returns:
        gates (list): List of gate objects (Hadamard and CNOT) to generate the GHZ state.
    '''
    gates = [H(0)]
    for i in range(n-1):
        gates.append(CNOT(i, i+1))
    return gates

#TODO please feel free to validate and change, just added the function here to test
def make_adder_gates(n):
    # Sparse-friendly
    '''
    Constructs a quantum circuit for an n-qubit quantum adder (half-adder for n=2).

    For n=2, the circuit adds two qubits (q0, q1) and stores the sum in q0 and the carry in an ancilla qubit (q2).
    The circuit uses CNOT and Toffoli (CCNOT) gates to compute the sum and carry bits.

    Args:
        n (int): Number of qubits (for a half-adder, n=3: two inputs and one ancilla for carry).

    Returns:
        gates (list): List of gate objects (CNOT, Toffoli) implementing the quantum adder.
    '''
    gates = []
    # For a half-adder: q0, q1 are inputs, q2 is ancilla for carry
    # Sum: CNOT(q1, q0)
    gates.append(CNOT(1, 0))
    # Carry: Toffoli(q0, q1, 2) -- you need to implement a Toffoli gate class
    gates.append(Toffoli(0, 1, 2))
    return gates
    
def make_swap_test_gates(n):
    # Dense-friendly
    '''
    Constructs a quantum circuit for the SWAP test, which estimates the overlap between two n-qubit quantum states.

    The SWAP test uses an ancilla qubit (qubit 0), applies a Hadamard gate to it, then applies controlled-SWAP gates between corresponding pairs of qubits in the two states, and finally applies another Hadamard to the ancilla before measurement.

    Args:
        n (int): Number of qubits in each state to compare (total qubits = 2n + 1, including ancilla).

    Returns:
        gates (list): List of gate objects (Hadamard, controlled-SWAP) implementing the SWAP test.
    '''
    gates = []
    # Ancilla is qubit 0, first state is qubits 1..n, second state is qubits n+1..2n
    gates.append(H(0))
    for i in range(n):
        gates.append(CSWAP(0, 1 + i, 1 + n + i))
    gates.append(H(0))
    return gates


# SWAP Test Runner - needs its own runner since requires 2n + 1 qubits instead of the usual n
def run_swap_test(QuantumSim, n):
    total_qubits = 2 * n + 1
    if QuantumSim is SparseQuantumSimulator:
        st = {0: 1.0 + 0.0j}
    elif QuantumSim is DenseQuantumSimulator:
        st = [1.0 + 0.0j] + [0.0 + 0.0j for _ in range(2**total_qubits - 1)]
    else:
        raise ValueError("Unknown simulator type")
    sim = QuantumSim(number_qubits=total_qubits, states=st)
    gates = make_swap_test_gates(n)
    sim.apply_circuit(gates)
    return sim

# Generalized Circuit Runner 
def run_circuit(QuantumSim, n, circuit_method):
    if QuantumSim is SparseQuantumSimulator:
        st = {0: 1.0 + 0.0j}
    elif QuantumSim is DenseQuantumSimulator:
        st = [1.0 + 0.0j] + [0.0 + 0.0j for _ in range(2**n - 1)]
    else:
        raise ValueError("Unknown simulator type")
    sim = QuantumSim(number_qubits=n, states=st)
    gates = circuit_method(n)
    sim.apply_circuit(gates)
    return sim


# Nonzero Count Helper 
def max_nonzero_count(states):
    if isinstance(states, dict):
        return len([v for v in states.values() if abs(v) > 1e-8])
    else:
        return len([v for v in states if abs(v) > 1e-8])

# General Evaluation Function for Standard Circuits
def Evaluate(Sim_class, n_list=[8,10,12], circuit_method=None):
    results = {m: {"runtime": None, "memory": None, "max_nonzero_count": None} for m in n_list}
    for m in n_list:
        print(f"Evaluating {Sim_class.__name__} with n={m} using {circuit_method.__name__}")
        tracemalloc.start()
        t0 = time.time()
        sim = run_circuit(Sim_class, m, circuit_method)
        t1 = time.time()
        mem = tracemalloc.get_traced_memory()[1] / (1024**2)  # MB
        tracemalloc.stop()
        nz = sim.non_zero_counts
        runtime = t1 - t0
        results[m] = {"runtime": runtime, "memory": mem, "max_nonzero_count": nz}
    return results

# Specialized Evaluation Function for SWAP Test
def EvaluateSwapTest(Sim_class, n_list=[8,10,12]):
    results = {m: {"runtime": None, "memory": None, "max_nonzero_count": None} for m in n_list}
    for m in n_list:
        print(f"Evaluating {Sim_class.__name__} SWAP test with n={m}")
        tracemalloc.start()
        t0 = time.time()
        sim = run_swap_test(Sim_class, m)
        t1 = time.time()
        mem = tracemalloc.get_traced_memory()[1] / (1024**2)  # MB
        tracemalloc.stop()
        nz = sim.non_zero_counts
        runtime = t1 - t0
        results[m] = {"runtime": runtime, "memory": mem, "max_nonzero_count": nz}
    return results

def compare_sims(results_dict, save_dir="plots", filename="simulator_comparison.png",
                 title_prefix="", use_log_scale=True):
    """
    Compare simulator performance across metrics and qubit counts.
    Layout: 3 columns (one per metric), legend shown only on the last plot.
    """

    import matplotlib.pyplot as plt
    import os

    units = {
        "runtime": "Runtime (seconds)",
        "memory": "Memory (MB)",
        "max_nonzero_count": "Maximum Non-zero Amplitudes"
    }

    # Base colors for circuit types
    base_colors = {
        "QFT": "#1f77b4",   # blue
        "GHZ": "#2ca02c",   # green
        "Adder": "#ff7f0e", # orange
        "SWAP": "#9467bd"   # purple
    }

    # Extract metrics dynamically
    sample_sim = next(iter(results_dict.values()))
    metrics = list(next(iter(sample_sim.values())).keys())

    # Force exactly 3 columns (one row)
    n_cols = len(metrics)
    n_rows = 1
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5), squeeze=False)
    axs = axs.flatten()

    for i, metric in enumerate(metrics):
        ax = axs[i]
        y_label = units.get(metric, metric.replace("_", " ").title())

        for sim_name, sim_results in results_dict.items():
            qubits = sorted(sim_results.keys())
            values = [sim_results[q][metric] for q in qubits]

            # Identify circuit type and simulator type
            circuit_type = next((k for k in base_colors.keys() if k in sim_name), None)
            color = base_colors.get(circuit_type, "#7f7f7f")
            linestyle = "-" if "Dense" in sim_name else "--"

            ax.plot(qubits, values, linestyle=linestyle, color=color,
                    marker="o", label=sim_name, markersize=5, alpha=0.9)

        ax.set_xlabel("Number of qubits (n)")
        ax.set_ylabel(y_label)
        ax.set_title(f"{title_prefix} {y_label} vs Number of Qubits" if title_prefix else y_label)

        if use_log_scale:
            min_val = min([v for sim in results_dict.values()
                           for v in [sim[q][metric] for q in sorted(sim.keys())]])
            if min_val > 0:
                ax.set_yscale("log")

        ax.grid(True, linestyle="--", alpha=0.7)

        # ✅ Only put legend on the *last* subplot
        if i == len(metrics) - 1:
            ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=9)

    # Clean layout
    fig.tight_layout(rect=[0, 0, 0.88, 1])

    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")
    plt.show()
    plt.close(fig)


# Main Script #TODO Swap Test takes super long for some reason, so I've commented it out
def main():
    num_qubits = [i*2 for i in range(2,7)]
    res_dense_qft = Evaluate(DenseQuantumSimulator, n_list=num_qubits, circuit_method=make_qft_gates)
    res_sparse_qft = Evaluate(SparseQuantumSimulator, n_list=num_qubits, circuit_method=make_qft_gates)
    res_dense_ghz = Evaluate(DenseQuantumSimulator, n_list=num_qubits, circuit_method=make_ghz_gates)
    res_sparse_ghz = Evaluate(SparseQuantumSimulator, n_list=num_qubits, circuit_method=make_ghz_gates)
    res_dense_adder = Evaluate(DenseQuantumSimulator, n_list=num_qubits, circuit_method=make_adder_gates)
    res_sparse_adder = Evaluate(SparseQuantumSimulator, n_list=num_qubits, circuit_method=make_adder_gates)
    """res_dense_swap = EvaluateSwapTest(DenseQuantumSimulator, n_list=num_qubits)
    res_sparse_swap = EvaluateSwapTest(SparseQuantumSimulator, n_list=num_qubits)"""
    compare_sims({
        "Dense QFT": res_dense_qft,
        "Sparse QFT": res_sparse_qft,
        "Dense GHZ": res_dense_ghz,
        "Sparse GHZ": res_sparse_ghz,
        "Dense Adder": res_dense_adder,
        "Sparse Adder": res_sparse_adder,
        #"Dense SWAP Test": res_dense_swap,
        #"Sparse SWAP Test": res_sparse_swap
    }, filename="simulator_comparison_all.png")
    # plot only QFT
    compare_sims({
        "Dense QFT": res_dense_qft,
        "Sparse QFT": res_sparse_qft
    }, filename="simulator_comparison_qft.png", title_prefix="QFT")
    # plot only GHZ
    compare_sims({
        "Dense GHZ": res_dense_ghz,
        "Sparse GHZ": res_sparse_ghz
    }, filename="simulator_comparison_ghz.png", title_prefix="GHZ")
    # plot only Adder
    compare_sims({
        "Dense Adder": res_dense_adder,
        "Sparse Adder": res_sparse_adder
    }, filename="simulator_comparison_adder.png", title_prefix="Adder")
    # plot only SWAP
    """compare_sims({
         "Dense SWAP Test": res_dense_swap,
         "Sparse SWAP Test": res_sparse_swap
     }, filename="simulator_comparison_swap.png", title_prefix="SWAP Test")"""

if __name__ == "__main__":
    main()




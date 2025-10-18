from simulator import SparseQuantumSimulator, DenseQuantumSimulator
from gates import Gate, H, X, SWAP,Z,Y,S, T,RX,RZ,RY, CP
import numpy as np
import time
import tracemalloc
import matplotlib.pyplot as plt

def run_qft(QuantumSim, n):
  # PREP
  def make_prep_gates(m):
    gates = []
    for i in range(m):
      gates.append(H(i))
    gates.append(CP(1, 1, np.pi))
    return gates

  # QFT
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

  # INIT STATE
  def make_init_state(m):      
      if QuantumSim is SparseQuantumSimulator:
          st = {0:1.0+0.0j} 
      elif QuantumSim is DenseQuantumSimulator:
          st = [1.0+0.0j]+[0.0+0.0j for i in range(2**m -1)]
      return st
  
  PREP = make_prep_gates(n)
  QFT = make_qft_gates(n)
  st = make_init_state(n)
  
  sim = QuantumSim(number_qubits=n, states=st)
  
  sim.apply_circuit(PREP)
  sim.apply_circuit(QFT)

  return sim

# Report runtime, memory, and maximum non-zero count

def max_nonzero_count(states):
    # For sparse: states is a dict, for dense: states is a list
    if isinstance(states, dict):
        return len([v for v in states.values() if abs(v) > 1e-12])
    else:
        return len([v for v in states if abs(v) > 1e-12])
    
def Evaluate(Sim_class,n=[8,10,12], circuit_method=None):
    results= {m:{"runtime":None, "memory":None, "max_nz_count":None} for m in n}
    for m in n:
        print(Sim_class,m)
        tracemalloc.start()
        t0 = time.time()
        sim = circuit_method(Sim_class, m)
        t1 = time.time()
        mem = tracemalloc.get_traced_memory()[1] / (1024**2)  # MB
        tracemalloc.stop()
        nz = max_nonzero_count(sim.states)
        runtime=t1-t0
        results[m]={"runtime":runtime, "memory":mem, "max_nz_count":nz}

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
    num_qubits = [i*2 for i in range(8)]
    res_dense = Evaluate(DenseQuantumSimulator,n=num_qubits,circuit_method=run_qft)
    res_sparse = Evaluate(SparseQuantumSimulator,n=num_qubits,circuit_method=run_qft)
    print("Dense:", res_dense)
    print("Sparse:", res_sparse)
    compare_sims({
        "Dense NumPy Vector": res_dense,
        "Sparse Hashmap": res_sparse
    })

if __name__=="__main__":
    main()

from simulator import SparseQuantumSimulator, DenseQuantumSimulator
from gates import Gate, H, X, SWAP,Z,Y,S, T,RX,RZ,RY, CP
import numpy as np

def run_qft():

    # Prep
    h1 = H(0)
    h2 = H(1)
    h3 = H(2)
    h4 = H(3)

    cp1 = CP(1, 1, np.pi)

    # QFT
    def make_qft_gates(n):
      gates = []
      for j in reversed(range(n)):
          gates.append(H(j))
          for k in range(j-1, -1, -1):
              angle = -np.pi / (2 ** (j - k))
              gates.append(CP(k, j, angle))
              print(f"CP({k},{j},{angle})")
      for i in range(n // 2):
          gates.append(SWAP(i, n - 1 - i))
      return gates

    PREP = [h1, h2, h3, h4, cp1]
    QFT = make_qft_gates(4)
    
    st = {0:1.0+0.0j} 
    sqs = SparseQuantumSimulator(number_qubits=4, states=st)
    st = [1.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j,
           0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j,
             0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j,
               0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j]
    dqs = DenseQuantumSimulator(number_qubits=4, states=st)

    dqs.apply_circuit(PREP)
    sqs.apply_circuit(PREP)

    dqs.apply_circuit(QFT)
    sqs.apply_circuit(QFT)

    return sqs,dqs


run_qft()
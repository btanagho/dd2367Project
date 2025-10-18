from simulator import SparseQuantumSimulator, DenseQuantumSimulator
from gates import Gate, H, X, SWAP,Z,Y,S

def hadamard_test():
    st = {0:1.0+0.0j} 
    sqs = SparseQuantumSimulator(number_qubits=3, states=st)
    st = [1.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j]
    dqs = DenseQuantumSimulator(number_qubits=3, states=st)
    h1 = H(0)
    h2 = H(1)
    h3 = H(2)

    circuit = [h1, h2, h3]

    sqs.apply_circuit(circuit)
    dqs.apply_circuit(circuit)

    print(sqs.states)
    print(dqs.states)

def flip_test():
    st = {2:0.8+0.0j, 7:0.6+0.0j} 
    sqs = SparseQuantumSimulator(number_qubits=3, states=st)
    st = [0.0+0.0j, 0.0+0.0j, 0.8+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.6+0.0j]
    dqs = DenseQuantumSimulator(number_qubits=3, states=st)

    x1 = X(0)
    x2 = X(1)
    x3 = X(2)

    circuit = [x1, x2, x3]

    sqs.apply_circuit(circuit)
    dqs.apply_circuit(circuit)

    print(sqs.states)
    print(dqs.states)

def swap_test():
    st = {1:0.8+0.0j, 6:0.6+0.0j} 
    sqs = SparseQuantumSimulator(number_qubits=3, states=st)
    st = [0.0+0.0j, 0.8+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.6+0.0j, 0.0+0.0j]
    dqs = DenseQuantumSimulator(number_qubits=3, states=st)

    s1 = SWAP(0, 2)

    circuit = [s1]

    sqs.apply_circuit(circuit)
    dqs.apply_circuit(circuit)

    print(sqs.states)
    print(dqs.states)


def z_gate_test():
    print("\n--- Z Gate Test ---")
    # Initial state: amplitudes on |010> (2) and |111> (7)
    st = {2: 0.8+0.0j, 7: 0.6+0.0j}
    sqs = SparseQuantumSimulator(number_qubits=3, states=st)
    st_dense = [0.0+0.0j]*8
    st_dense[2] = 0.8+0.0j
    st_dense[7] = 0.6+0.0j
    dqs = DenseQuantumSimulator(number_qubits=3, states=st_dense)

    z0 = Z(0)
    z1 = Z(1)
    z2 = Z(2)

    circuit = [z0, z1, z2]

    sqs.apply_circuit(circuit)
    dqs.apply_circuit(circuit)

    print("Sparse:", sqs.states)
    print("Dense:", dqs.states)


def s_gate_test():
    print("\n--- S Gate Test ---")
    # Initial state: amplitudes on |001> (1) and |111> (7)
    st = {1: 0.8+0.0j, 7: 0.6+0.0j}
    sqs = SparseQuantumSimulator(number_qubits=3, states=st)
    st_dense = [0.0+0.0j]*8
    st_dense[1] = 0.8+0.0j
    st_dense[7] = 0.6+0.0j
    dqs = DenseQuantumSimulator(number_qubits=3, states=st_dense)

    s0 = S(0)
    s1 = S(1)
    s2 = S(2)

    circuit = [s0, s1, s2]

    sqs.apply_circuit(circuit)
    dqs.apply_circuit(circuit)

    print("Sparse:", sqs.states)
    print("Dense:", dqs.states)


def y_gate_test():
    print("\n--- Y Gate Test ---")
    # Initial state: amplitudes on |000> (0) and |101> (5)
    st = {0: 0.8+0.0j, 5: 0.6+0.0j}
    sqs = SparseQuantumSimulator(number_qubits=3, states=st)
    st_dense = [0.0+0.0j]*8
    st_dense[0] = 0.8+0.0j
    st_dense[5] = 0.6+0.0j
    dqs = DenseQuantumSimulator(number_qubits=3, states=st_dense)

    y0 = Y(0)
    y1 = Y(1)
    y2 = Y(2)

    circuit = [y0, y1, y2]

    sqs.apply_circuit(circuit)
    dqs.apply_circuit(circuit)

    print("Sparse:", sqs.states)
    print("Dense:", dqs.states)


if __name__=="__main__":
    hadamard_test()
    flip_test()
    swap_test()
    z_gate_test()
    s_gate_test()
    y_gate_test()


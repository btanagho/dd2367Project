from simulator import SparseQuantumSimulator, DenseQuantumSimulator
from gates import Gate, H, X, SWAP

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

hadamard_test()
flip_test()
swap_test()

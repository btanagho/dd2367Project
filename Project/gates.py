import numpy as np

class Gate():
    # X, Y, Z, H, S, T, RX/RY/RZ, CX/CZ/CP, SWAP

    def __init__(self):
        pass

    # def apply_gate():
    #     pass

    def apply(self, simulator):
        if getattr(simulator, 'is_dense', False):
            return self.apply_dense_gate(simulator)
        else:
            return self.apply_sparse_gate(simulator)


class H(Gate):
    ''' 
        For a given state, e.g 001 with qubit_index = 2, do:
        |001> = (|001> + |101>) / sqrt(2)
        |101> = (|001> - |101>) / sqrt(2)
    '''
    
    def __init__(self,  target_qubit):
        super().__init__()
        self.target_qubit = target_qubit

    def apply_sparse_gate(self, state):
        inv_sqrt2 = 1 / np.sqrt(2)
        old_states = state.states
        new_states = {}

        for index, amp in old_states.items():
            partner_index = index ^ (1 << self.target_qubit)
            bit = (index >> self.target_qubit) & 1

            if bit == 0:
                ts1 = amp * inv_sqrt2
                ts2 = amp * inv_sqrt2
            else:
                ts1 = -amp * inv_sqrt2
                ts2 = amp * inv_sqrt2

            new_states[index] = new_states.get(index, 0) + ts1
            new_states[partner_index] = new_states.get(partner_index, 0) + ts2
        
        state.states = new_states
        return state

    def apply_dense_gate(self, simulator):
        n = simulator.n
        state_vec = np.array(simulator.states)
        for i in range(len(state_vec)):
            bit = (i >> self.target_qubit) & 1
            partner = i ^ (1 << self.target_qubit)
            if i < partner:
                amp_i = state_vec[i]
                amp_p = state_vec[partner]
                state_vec[i] = (amp_i + amp_p) / np.sqrt(2)
                state_vec[partner] = (amp_i - amp_p) / np.sqrt(2)
        simulator.states = list(state_vec)
        return simulator


class X(Gate):
    """
    Represents the Pauli-X (bit-flip) gate

    So it acts like a classical NOT operation:
    X|0> = |1>
    X|1> = |0>
    
    On a multi-qubit system, the target qubits bit is flipped 
    in the computational basis for every basis state amplitude
    """

    def __init__(self, target_qubit):
        super().__init__()
        self.target_qubit = target_qubit


    def apply_sparse_gate(self, state):
        mask = 1<< self.target_qubit
        visited = set()

        for idx in list(state.states.keys()):
            if idx in visited:
                continue
            flipped_idx = idx ^ mask
            visited.add(idx)
            visited.add(flipped_idx)

            amp_i = state.get_amplitude(idx)
            amp_j = state.get_amplitude(flipped_idx)

            # swap amplitudes
            state.set_amplitude(idx, amp_j)
            state.set_amplitude(flipped_idx, amp_i)

        return state

    def apply_dense_gate(self, simulator):
        n = simulator.n
        state_vec = np.array(simulator.states)
        mask = 1 << self.target_qubit
        for i in range(len(state_vec)):
            j = i ^ mask
            if i < j:
                state_vec[i], state_vec[j] = state_vec[j], state_vec[i]
        simulator.states = list(state_vec)
        return simulator
    
class SWAP(Gate):
    """
    The SWAP gate exchanges the quantum states of two qubits, swapping their values in every basis state.
    This operation is reversible and preserves the overall quantum state.
    """
    def __init__(self, qubit1: int, qubit2: int):
        super().__init__()
        self.qubit1 = qubit1
        self.qubit2 = qubit2
        
    def apply_sparse_gate(self, state):
        old_states = state.states
        new_states = {}

        def swap_bits(basis, qubit1, qubit2):
            # Extract the two bits
            bit1 = (basis >> qubit1) & 1
            bit2 = (basis >> qubit2) & 1

            # If bits are the same, no change
            if bit1 == bit2:
                return basis

            # Create a bit mask where bits at qubit1 and qubit2 are 1
            bit_mask = (1 << qubit1) | (1 << qubit2)

            # Toggle both bits (flip them)
            swapped = basis ^ bit_mask
            return swapped

        for basis, amp in list(old_states.items()):
            swapped_basis = swap_bits(basis, self.qubit1, self.qubit2)
            new_states[swapped_basis] = new_states.get(swapped_basis, 0) + amp
        
        state.states = new_states
        return state
    
    def apply_dense_gate(self, simulator):
        n = simulator.n
        state_vec = np.array(simulator.states)
        for i in range(len(state_vec)):
            bit1 = (i >> self.qubit1) & 1
            bit2 = (i >> self.qubit2) & 1
            if bit1 != bit2:
                j = i ^ ((1 << self.qubit1) | (1 << self.qubit2))
                if i < j:
                    state_vec[i], state_vec[j] = state_vec[j], state_vec[i]
        simulator.states = list(state_vec)
        return simulator
    


class Z(Gate):
    """
    Applis 180 degree shift on |1>
    Pauli-Z gate:
        Z|0> = |0>
        Z|1> = -|1>
    """
    def __init__(self, target_qubit):
        super().__init__()
        self.target_qubit = target_qubit

    def apply_sparse_gate(self, state):
        for idx, amp in list(state.states.items()):
            bit = (idx >> self.target_qubit) & 1
            if bit == 1:
                state.set_amplitude(idx, -amp)
        return state

    def apply_dense_gate(self, simulator):
        state_vec = np.array(simulator.states, dtype=complex)
        for i in range(len(state_vec)):
            bit = (i >> self.target_qubit) & 1
            if bit == 1:
                state_vec[i] *= -1
        simulator.states = list(state_vec)
        return simulator

class S(Gate):
    """
    Applis 90 degree shift on |1>
    Phase (S) gate:
        S|0> = |0>
        S|1> = i|1>
    """
    def __init__(self, target_qubit):
        super().__init__()
        self.target_qubit = target_qubit

    def apply_sparse_gate(self, state):
        for idx, amp in list(state.states.items()):
            bit = (idx >> self.target_qubit) & 1
            if bit == 1:
                state.set_amplitude(idx, amp * 1j)
        return state

    def apply_dense_gate(self, simulator):
        state_vec = np.array(simulator.states, dtype=complex)
        for i in range(len(state_vec)):
            bit = (i >> self.target_qubit) & 1
            if bit == 1:
                state_vec[i] *= 1j
        simulator.states = list(state_vec)
        return simulator



class Y(Gate):
    """
    Pauli-Y gate:
        Y|0> = i|1>
        Y|1> = -i|0>
    """
    def __init__(self, target_qubit: int):
        super().__init__()
        self.target_qubit = target_qubit

    def apply_sparse_gate(self, state):
        mask = 1 << self.target_qubit
        visited = set()

        for i in list(state.states.keys()):
            if i in visited:
                continue
            j = i ^ mask
            visited.add(i); visited.add(j)

            ai = state.get_amplitude(i)
            aj = state.get_amplitude(j)

            # Apply [ [0, -i], [i, 0] ] to the pair (ai, aj)
            state.set_amplitude(i, (-1j) * aj)
            state.set_amplitude(j, (+1j) * ai)
        return state

    def apply_dense_gate(self, simulator):
        state_vec = np.array(simulator.states, dtype=complex)
        mask = 1 << self.target_qubit
        for i in range(len(state_vec)):
            j = i ^ mask
            if i < j:
                ai = state_vec[i]
                aj = state_vec[j]
                state_vec[i] = (-1j) * aj
                state_vec[j] = (+1j) * ai
        simulator.states = list(state_vec)
        return simulator


class T(Gate):
    """
    Applies 45° (π/4) phase shift on |1>.
        T|0> = |0>
        T|1> = e^{iπ/4}|1>
    """
    def __init__(self, target_qubit):
        super().__init__()
        self.target_qubit = target_qubit
        self.phase = np.exp(1j * np.pi / 4)

    def apply_sparse_gate(self, state):
        for idx, amp in list(state.states.items()):
            bit = (idx >> self.target_qubit) & 1
            if bit == 1:
                state.set_amplitude(idx, amp * self.phase)
        return state

    def apply_dense_gate(self, simulator):
        state_vec = np.array(simulator.states, dtype=complex)
        for i in range(len(state_vec)):
            bit = (i >> self.target_qubit) & 1
            if bit == 1:
                state_vec[i] *= self.phase
        simulator.states = list(state_vec)
        return simulator

    


class RX(Gate):
    """
    RX rotation: cos(θ/2)I - i sin(θ/2)X
    """
    def __init__(self, target_qubit, theta):
        super().__init__()
        self.target_qubit = target_qubit
        self.theta = theta

    def apply_sparse_gate(self, state):
        c = np.cos(self.theta / 2)
        s = -1j * np.sin(self.theta / 2)
        mask = 1 << self.target_qubit
        visited = set()
        for i in list(state.states.keys()):
            if i in visited:
                continue
            j = i ^ mask
            visited.add(i); visited.add(j)
            ai = state.get_amplitude(i)
            aj = state.get_amplitude(j)
            state.set_amplitude(i, c*ai + s*aj)
            state.set_amplitude(j, s*ai + c*aj)
        return state

    def apply_dense_gate(self, simulator):
        c = np.cos(self.theta / 2)
        s = -1j * np.sin(self.theta / 2)
        state_vec = np.array(simulator.states, dtype=complex)
        mask = 1 << self.target_qubit
        for i in range(len(state_vec)):
            j = i ^ mask
            if i < j:
                ai = state_vec[i]
                aj = state_vec[j]
                state_vec[i] = c*ai + s*aj
                state_vec[j] = s*ai + c*aj
        simulator.states = list(state_vec)
        return simulator


class RY(Gate):
    """
    RY rotation: cos(θ/2)I - i sin(θ/2)Y
    """
    def __init__(self, target_qubit, theta):
        super().__init__()
        self.target_qubit = target_qubit
        self.theta = theta

    def apply_sparse_gate(self, state):
        c = np.cos(self.theta / 2)
        s = np.sin(self.theta / 2)
        mask = 1 << self.target_qubit
        visited = set()
        for i in list(state.states.keys()):
            if i in visited:
                continue
            j = i ^ mask
            visited.add(i); visited.add(j)
            ai = state.get_amplitude(i)
            aj = state.get_amplitude(j)
            state.set_amplitude(i, c*ai - s*aj)
            state.set_amplitude(j, s*ai + c*aj)
        return state

    def apply_dense_gate(self, simulator):
        c = np.cos(self.theta / 2)
        s = np.sin(self.theta / 2)
        state_vec = np.array(simulator.states, dtype=complex)
        mask = 1 << self.target_qubit
        for i in range(len(state_vec)):
            j = i ^ mask
            if i < j:
                ai = state_vec[i]
                aj = state_vec[j]
                state_vec[i] = c*ai - s*aj
                state_vec[j] = s*ai + c*aj
        simulator.states = list(state_vec)
        return simulator



class RZ(Gate):
    """
    RZ rotation: e^{-iθ/2}|0> + e^{iθ/2}|1>
    """
    def __init__(self, target_qubit, theta):
        super().__init__()
        self.target_qubit = target_qubit
        self.theta = theta

    def apply_sparse_gate(self, state):
        phase0 = np.exp(-1j * self.theta / 2)
        phase1 = np.exp(1j * self.theta / 2)
        for idx, amp in list(state.states.items()):
            bit = (idx >> self.target_qubit) & 1
            state.set_amplitude(idx, amp * (phase1 if bit else phase0))
        return state

    def apply_dense_gate(self, simulator):
        phase0 = np.exp(-1j * self.theta / 2)
        phase1 = np.exp(1j * self.theta / 2)
        state_vec = np.array(simulator.states, dtype=complex)
        for i in range(len(state_vec)):
            bit = (i >> self.target_qubit) & 1
            state_vec[i] *= phase1 if bit else phase0
        simulator.states = list(state_vec)
        return simulator


class CP(Gate):
    """
    Controlled-Phase gate:
        Adds e^{iφ} phase if both control and target qubits = 1.
    """
    def __init__(self, control_qubit, target_qubit, phi):
        super().__init__()
        self.control = control_qubit
        self.target = target_qubit
        self.phi = phi
        self.phase = np.exp(1j * phi)

    def apply_sparse_gate(self, state):
        for idx, amp in list(state.states.items()):
            control_bit = (idx >> self.control) & 1
            target_bit = (idx >> self.target) & 1
            if control_bit == 1 and target_bit == 1:
                state.set_amplitude(idx, amp * self.phase)
        return state

    def apply_dense_gate(self, simulator):
        state_vec = np.array(simulator.states, dtype=complex)
        for i in range(len(state_vec)):
            control_bit = (i >> self.control) & 1
            target_bit = (i >> self.target) & 1
            if control_bit == 1 and target_bit == 1:
                state_vec[i] *= self.phase
        simulator.states = list(state_vec)
        return simulator

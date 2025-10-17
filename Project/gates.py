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
        old_states = state.states
        inv_sqrt2 = 1/np.sqrt(2)
        for index, amp in list(old_states.items()):
            partner_index = index ^ (1 << self.target_qubit)
            bit = (index >> self.target_qubit) & 1

            if bit == 0:
                ts1 = amp * inv_sqrt2
                ts2 = amp * inv_sqrt2
            else:
                ts1 = -amp * inv_sqrt2
                ts2 = amp * inv_sqrt2

            state.set_amplitude(index, ts1)
            state.set_amplitude(partner_index, 
            state.states.get(partner_index, 0) + ts2)
        
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
    


        
           

       
    



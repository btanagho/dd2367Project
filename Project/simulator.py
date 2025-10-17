import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

from gates import Gate

class GenericQuantumSimulator():
    epsilon = 1e-12

    def __init__(self):
        pass
    
    def apply_circuit(self, circuit : list[Gate]):
        pass

class SparseQuantumSimulator(GenericQuantumSimulator):

    def __init__(self, number_qubits:int = 5, states: dict[int,complex] =None):
        self.n = number_qubits

        if states is not None:
            if not isinstance(states, dict):
                raise TypeError(f"'states' must be a dict[int, complex], got {type(states).__name__}")
            
            for key, value in states.items():
                if not isinstance(key, int):
                    raise TypeError(f"All keys in 'states' must be int, but got {type(key).__name__} ({key})")
                if not isinstance(value, complex):
                    raise TypeError(f"All values in 'states' must be complex, but got {type(value).__name__} ({value})")

            self.states = {index: value for index, value in states.items() if abs(value) >= self.epsilon}
        
    #TODO this doesn't take into account complex numbers and should be changed
    def get_amplitude(self, base_index):
        return self.states.get(base_index, 0.0+0.0j)

    #This is a bit hardcodey but if used properly should work 
    def set_amplitude(self, base_index, value):
        if abs(value) < self.epsilon:
            if base_index in self.states:
                del self.states[base_index]
        else:
            self.states[base_index] = value
    
    def collapse(self):
        states = list(self.states.keys())
        probs  = [abs(self.states[i])**2 for i in states]
        
        chosen_index = random.choices(states, weights=probs)[0]

        self.states = {chosen_index: 1.0+0.0j}

        return chosen_index

    def apply_circuit(self, circuit : list[Gate]):
        for gate in circuit:
            gate.apply_gate(self)

    def circ_plot(self, states=None, max_columns=6, amplitude_scale=0.6):
        if states is None:
            states = self.state

        basis_states = 2**self.n
        amplitudes = {k: np.absolute(v) for k, v in self.state.items()}
        phases = {k: np.angle(v) for k, v in self.state.items()}

        rows = (basis_states + max_columns - 1) // max_columns  
        cols = min(basis_states, max_columns)

        fig, axs = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4), squeeze=False)

        for row in range(rows):
            for col in range(cols):
                idx = row * max_columns + col
                if idx >= basis_states:
                    axs[row][col].axis('off')
                    continue
                    
                amplitude = amplitudes.get(idx, 0)
                phase = phases.get(idx, 0)

            # Outer circle with custom color depending on amplitude threshold
            outer_color = 'gray' if amplitude > 0.001 else 'lightgray'
            circleExt = patches.Circle((0.5, 0.5), 0.5, color=outer_color, alpha=0.15)
            axs[row][col].add_patch(circleExt)
            
            # Inner circle scaled by amplitude
            circleInt = patches.Circle((0.5, 0.5), amplitude * amplitude_scale, color='navy', alpha=0.4)
            axs[row][col].add_patch(circleInt)
            axs[row][col].set_aspect('equal')
            
            state_label = "|" + format(idx, f'0{basis_states}b') + ">" 
            axs[row][col].set_title(state_label, fontsize=9)
            
            # Phase vector in red
            angle = phase + np.pi / 2
            xl = [0.5, 0.5 + amplitude * amplitude_scale * math.cos(angle)]
            yl = [0.5, 0.5 + amplitude * amplitude_scale * math.sin(angle)]
            axs[row][col].plot(xl, yl, 'r')
            axs[row][col].axis('off')
    
        plt.tight_layout()
        plt.show()

    def plot_multiple_states(self, diff_states: list[dict[int, complex]]):
        for i, state in enumerate(diff_states):
            print(f"Plotting state {i+1}/{len(diff_states)}")
            self.circ_plot(state=state)


class DenseQuantumSimulator(GenericQuantumSimulator):
    def __init__(self, number_qubits:int = 5, states: list[complex] = None):
        self.n = number_qubits

        self.states = []

        if states is not None:
            for i in range(len(states)):
                if not isinstance(states[i], complex):
                    raise TypeError(f"All values in 'states' must be complex, but got {type(states[i]).__name__} ({states[i]})")
                self.states = states
        
    def get_amplitude(self, base_index):
        return self.states[base_index]

    def set_amplitude(self, base_index, value):
        self.states[base_index] = value
    
    def collapse(self):
        chosen_index = random.choices(range(self.n), weights=self.states)[0]
        #TODO do this properly (returning array with only one value on 1.0+0.0j)
        return chosen_index

    #TODO call a different apply_gate specific for dense
    def apply_circuit(self, circuit : list[Gate]):
        for gate in circuit:
            #gate.apply_gate(self)
            pass


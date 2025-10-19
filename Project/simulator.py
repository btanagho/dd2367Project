import numpy as np
import os
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

    is_dense = False

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
        i = 0
        for gate in circuit:
            gate.apply(self)
            filename = f"sparse_{i}"
            i += 1
            #self.circ_plot(filename=filename)


    def circ_plot(self, states=None, max_columns=6, amplitude_scale=0.6, folder="plots", filename=None):
        if states is None:
            states = self.states

        basis_states = 2**self.n
        amplitudes = {k: np.absolute(v) for k, v in self.states.items()}
        phases = {k: np.angle(v) for k, v in self.states.items()}

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

                outer_color = 'gray' if amplitude > 0.001 else 'lightgray'
                circleExt = patches.Circle((0.5, 0.5), 0.5, color=outer_color, alpha=0.15)
                axs[row][col].add_patch(circleExt)

                circleInt = patches.Circle((0.5, 0.5), amplitude * amplitude_scale, color='navy', alpha=0.4)
                axs[row][col].add_patch(circleInt)
                axs[row][col].set_aspect('equal')

                state_label = "|" + format(idx, f'0{self.n}b') + ">"
                axs[row][col].set_title(state_label, fontsize=9)

                angle = phase + np.pi / 2
                xl = [0.5, 0.5 + amplitude * amplitude_scale * math.cos(angle)]
                yl = [0.5, 0.5 + amplitude * amplitude_scale * math.sin(angle)]
                axs[row][col].plot(xl, yl, 'r')
                axs[row][col].axis('off')

        if filename:
            # Make sure folder exists
            os.makedirs(folder, exist_ok=True)
            # Build full path with .png extension
            save_path = os.path.join(folder, f"sqs-{filename}.png")
            plt.savefig(save_path)  # overwrites if exists
            plt.close()  # free memory
        else:
            plt.show()


    def plot_multiple_states(self, diff_states: list[dict[int, complex]]):
        for i, state in enumerate(diff_states):
            print(f"Plotting state {i+1}/{len(diff_states)}")
            self.circ_plot(state=state)


class DenseQuantumSimulator(GenericQuantumSimulator):

    is_dense = True

    #TODO proposed creation of DenseQuantumSimulator, existing code below - had to create this to test the gates, but please feel free to change and adapt
    def __init__(self, number_qubits:int = 5, states: list[complex] = None):
        self.n = number_qubits
        # State vector (dense); use all 0s state by default
        if states is not None:
            if len(states) != 2 ** self.n:
                raise ValueError(f"States list must be of length 2^{self.n}")
            for v in states:
                if not isinstance(v, complex):
                    raise TypeError(f"All values in 'states' must be complex, but got {type(v).__name__} ({v})")
            self.states = list(states)
        else:
            self.states = [0.0+0.0j] * (2 ** self.n)
            self.states[0] = 1.0 + 0.0j # |0...0>

    def get_amplitude(self, base_index):
        return self.states[base_index]
    
    def set_amplitude(self, base_index, value):
        self.states[base_index] = value

    def collapse(self):
        # Compute measurement probabilities
        probs = [abs(a)**2 for a in self.states]
        indices = list(range(len(self.states)))
        chosen_index = random.choices(indices, weights=probs)[0]
        # Collapse state to basis index 'chosen_index'
        self.states = [0.0+0.0j] * len(self.states)
        self.states[chosen_index] = 1.0+0.0j
        return chosen_index

    def apply_circuit(self, circuit : list[Gate]):
        i = 0
        for gate in circuit:
            gate.apply(self)
            filename = f"dense_{i}"
            i += 1
            #self.circ_plot(filename=filename)

    def circ_plot(self, states=None, max_columns=6, amplitude_scale=0.6, folder="plots", filename=None):
        # supports both explicit state vector or current
        if states is None:
            amplitudes = [abs(v) for v in self.states]
            phases = [np.angle(v) for v in self.states]
        else:
            amplitudes = [abs(v) for v in states]
            phases = [np.angle(v) for v in states]

        basis_states = 2 ** self.n
        rows = (basis_states + max_columns - 1) // max_columns
        cols = min(basis_states, max_columns)
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4), squeeze=False)

        for row in range(rows):
            for col in range(cols):
                idx = row * max_columns + col
                if idx >= basis_states:
                    axs[row][col].axis('off')
                    continue
                amplitude = amplitudes[idx]
                phase = phases[idx]
                outer_color = 'gray' if amplitude > 0.001 else 'lightgray'
                circleExt = patches.Circle((0.5, 0.5), 0.5, color=outer_color, alpha=0.15)
                axs[row][col].add_patch(circleExt)
                circleInt = patches.Circle((0.5, 0.5), amplitude * amplitude_scale, color='navy', alpha=0.4)
                axs[row][col].add_patch(circleInt)
                axs[row][col].set_aspect('equal')
                state_label = "|" + format(idx, f'0{self.n}b') + ">"
                axs[row][col].set_title(state_label, fontsize=9)
                angle = phase + np.pi / 2
                xl = [0.5, 0.5 + amplitude * amplitude_scale * math.cos(angle)]
                yl = [0.5, 0.5 + amplitude * amplitude_scale * math.sin(angle)]
                axs[row][col].plot(xl, yl, 'r')
                axs[row][col].axis('off')

        plt.tight_layout()

        if filename:
            # Make sure folder exists
            os.makedirs(folder, exist_ok=True)
            # Build full path with .png extension
            save_path = os.path.join(folder, f"dqs-{filename}.png")
            plt.savefig(save_path)  # overwrites if exists
            plt.close()  # free memory
        else:
            plt.show()

    def plot_multiple_states(self, diff_states: list[dict[int, complex]]):
        for i, state in enumerate(diff_states):
            print(f"Plotting state {i+1}/{len(diff_states)}")
            self.circ_plot(state=state)

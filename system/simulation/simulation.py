from typing import Tuple

import qutip
from scipy.constants import e as C_e, hbar as C_hbar, physical_constants
import numpy as np

from system.hamiltonians.hamiltonians import load_hamiltonian
from system.states import States, Basis
from system.transformations.utils import load_transformation, transform_basis
from timer import timer


class Simulation:
    def __init__(
            self,
            n: int,
            hamiltonian: str,
            dc_field: Tuple[int, int],
            rf_freq: float,
            rf_energy: float,
            t: float,
            timesteps: int,
    ):
        """

        :param n: Principal quantum number
        :param hamiltonian: String describing the pre-generated atomic Hamiltonian data
        :param dc_field: Initial value for strength of DC field in units of V / m.
        :param rf_freq: F_{RF} in units of GHz
        :param rf_energy: E_{RF} in units of V / m
        :param t: Protocol duration in units of microseconds
        :param timesteps: Number of equally-spaced points in time at which the system state is calculated
        """
        self.n = n
        self.hamiltonian = hamiltonian
        self.dc_field = dc_field
        self.rf_freq = rf_freq
        self.rf_energy = rf_energy
        self.t = t
        self.timesteps = timesteps

    def setup(self):
        """
        Calculates Hamiltonians in the n1n2mlms basis, keeping only the n1 == 0 or 1 states

        :return:
        """
        with timer("Generating states"):
            states = States(self.n, basis=Basis.N1_N2_ML_MS).states

        with timer("Loading Hamiltonian"):
            mat_1, mat_2, mat_2_minus, mat_2_plus = load_hamiltonian(self.hamiltonian)
            mat_2_combination = mat_2_plus + mat_2_minus  # Units of a0 e
            mat_2_combination *= C_e * physical_constants["Bohr radius"][0] / C_hbar
            mat_2_combination *= 1e-9  # Convert Hz to GHz

        with timer("Loading transformations"):
            transform_1 = load_transformation(self.n, Basis.N_L_J_MJ, Basis.N_L_ML_MS)
            transform_2 = load_transformation(self.n, Basis.N_L_ML_MS, Basis.N1_N2_ML_MS)

        with timer("Applying transformation to nlmlms"):
            mat_1 = transform_basis(mat_1, transform_1)
            mat_2 = transform_basis(mat_2, transform_1)
            mat_2_combination = transform_basis(mat_2_combination, transform_1)

        with timer("Applying transformation to n1n2mlms"):
            mat_1 = transform_basis(mat_1, transform_2)
            mat_2 = transform_basis(mat_2, transform_2)
            mat_2_combination = transform_basis(mat_2_combination, transform_2)

        with timer("Applying state filters"):
            indices_to_keep = []
            for i, (n1, n2, _ml, _ms) in enumerate(states):
                if _ms > 0 and (n1 == 0 or n1 == 1) and _ml >= 0:
                    # Only keep n1 == 0 or 1, remove ml and ms degenerate states (ignored for this simulation)
                    indices_to_keep.append(i)

            # Sort indices, first in increasing n1, then in increasing ml.
            indices_to_keep = sorted(indices_to_keep, key=lambda x: (states[x][0], states[x][2]))

            # Filter matrices to only keep rows/columns pertaining to relevant states
            mat_1 = mat_1[indices_to_keep, :][:, indices_to_keep]
            mat_2 = mat_2[indices_to_keep, :][:, indices_to_keep]
            mat_2_combination = mat_2_combination[indices_to_keep, :][:, indices_to_keep]

            # Filter states, only keeping relevant states
            self.states = np.array(states)[indices_to_keep]

            self.states_count = len(self.states)
            print(f"Filtered states to {self.states_count}")

        self.mat_1 = mat_1
        self.mat_2 = mat_2
        self.mat_2_combination = mat_2_combination

    def run(self):
        t_list = np.linspace(0, self.t * 1000, self.timesteps + 1)  # Use self.t (in ms) to creat t_list in ns
        # print(t_list.min(), t_list.max(), "timeeeeeeeee")
        initial_state = qutip.basis(self.states_count, 3)
        self.results = qutip.mesolve(
            self.get_hamiltonian, initial_state, t_list, c_ops=[],
            options=qutip.solver.Options(store_states=True, nsteps=2000),
            progress_bar=True
        )

    def get_hamiltonian(self, t: float, *args):
        """
        Calculates the Hamiltonian as a function of time, used as an argument in qutip.mesolve().
        :param t: Time (in nanoseconds)
        :param args:
        :return: qutip.Qobj containing the system Hamiltonian at the specified time
        """
        dc_field = self.dc_field[0] + t / 1000 / self.t * (self.dc_field[1] - self.dc_field[0])

        hamiltonian = self.mat_1 + dc_field * self.mat_2
        eigenvalues = np.diagonal(hamiltonian)
        s = 3  # Index for zero-energy state: n1=0, ml=3

        detunings = np.zeros(self.states_count)
        for i in range(2 * self.n - 1):
            if i >= self.states_count:
                break
            if i < self.n:
                detunings[i] = (i - s) * self.rf_freq - (eigenvalues[i] - eigenvalues[s])
            else:
                detunings[i] = self.rf_freq - (eigenvalues[i - self.n + 1] - eigenvalues[i]) + detunings[i - self.n + 1]

        # Create Hamiltonian with RF couplings. Diagonal elements from detunings calculated above.
        hamiltonian_with_rf = self.rf_energy * self.mat_2_combination + np.diagflat(detunings)
        return qutip.Qobj(hamiltonian_with_rf)


if __name__ == '__main__':
    hamiltonian = "56_rubidium87"
    sim = Simulation(
        n=56,
        hamiltonian=hamiltonian,
        dc_field=(185, 140),
        rf_freq=195e6 / 1e9,
        rf_energy=25,
        t=1,
        timesteps=1000,
    )
    sim.setup()
    sim.run()

    import pickle
    with open(f"{hamiltonian}.pkl", "wb") as f:
        pickle.dump(sim, f)

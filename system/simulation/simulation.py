from typing import Sequence, Union

import numpy as np
import qutip
from scipy.constants import e as C_e, hbar as C_hbar, physical_constants
from scipy.interpolate import interp1d

from system.hamiltonians.hamiltonians import load_hamiltonian
from system.simulation.utils import tukey
from system.states import States, Basis
from system.transformations.utils import load_transformation, transform_basis
from timer import timer


class Simulation:
    def __init__(
            self,
            n: int,
            hamiltonian: str,
            dc_field: Sequence[float],
            rf_freq: float,
            rf_field: Union[float, Sequence[float]],
            magnetic_field: Union[float, Sequence[float]],
            t: float,
            timesteps: int,
    ):
        """

        :param n: Principal quantum number
        :param hamiltonian: String describing the pre-generated atomic Hamiltonian data
        :param dc_field: Strength of DC field in units of V / m.
        :param rf_freq: F_{RF} in units of GHz
        :param rf_field: E_{RF} in units of V / m
        :param magnetic_field: B_z in units of Tesla
        :param t: Protocol duration in units of microseconds
        :param timesteps: Number of equally-spaced points in time at which the system state is calculated
        """
        self.n = n
        self.hamiltonian = hamiltonian
        self.dc_field = dc_field
        self.rf_freq = rf_freq
        self.rf_field = rf_field
        self.magnetic_field = magnetic_field
        self.t = t
        self.timesteps = timesteps

    def setup(self, keep_mat_2_minus: bool = True, keep_n1_1: bool = True):
        """
        Calculates Hamiltonians in the n1n2mlms basis, keeping only the n1 == 0 or 1 states

        :return:
        """
        with timer("Generating states"):
            states = States(self.n, basis=Basis.N1_N2_ML_MS).states

        with timer("Loading Hamiltonian"):
            mat_1, mat_1_zeeman, mat_2, mat_2_minus, mat_2_plus = load_hamiltonian(self.hamiltonian)
            if keep_mat_2_minus:
                mat_2_combination = mat_2_plus + mat_2_minus  # Units of a0 e
            else:
                mat_2_combination = mat_2_plus  # Units of a0 e
            mat_2_combination *= C_e * physical_constants["Bohr radius"][0] / C_hbar
            mat_2_combination *= 1e-9  # Convert Hz to GHz

        with timer("Loading transformations"):
            transform_1 = load_transformation(self.n, Basis.N_L_J_MJ, Basis.N_L_ML_MS)
            transform_2 = load_transformation(self.n, Basis.N_L_ML_MS, Basis.N1_N2_ML_MS)

        with timer("Applying transformation to nlmlms"):
            mat_1 = transform_basis(mat_1, transform_1)
            mat_1_zeeman = transform_basis(mat_1_zeeman, transform_1)
            mat_2 = transform_basis(mat_2, transform_1)
            mat_2_combination = transform_basis(mat_2_combination, transform_1)

        with timer("Applying transformation to n1n2mlms"):
            mat_1 = transform_basis(mat_1, transform_2)
            mat_1_zeeman = transform_basis(mat_1_zeeman, transform_2)
            mat_2 = transform_basis(mat_2, transform_2)
            mat_2_combination = transform_basis(mat_2_combination, transform_2)

        with timer("Applying state filters"):
            indices_to_keep = []
            for i, (n1, n2, _ml, _ms) in enumerate(states):
                if _ms > 0 and (n1 == 0 or n1 == 1) and _ml >= 0:
                    # Only keep n1 == 0 or 1, remove ml and ms degenerate states (ignored for this simulation)
                    indices_to_keep.append(i)

            # Sort indices, first in increasing n1, then in increasing ml.
            indices_to_keep = sorted(indices_to_keep, key=lambda i: (states[i][0], states[i][2]))

            # Filter matrices to only keep rows/columns pertaining to relevant states
            mat_1 = mat_1[indices_to_keep, :][:, indices_to_keep]
            mat_1_zeeman = mat_1_zeeman[indices_to_keep, :][:, indices_to_keep]
            mat_2 = mat_2[indices_to_keep, :][:, indices_to_keep]
            mat_2_combination = mat_2_combination[indices_to_keep, :][:, indices_to_keep]

            # Filter states, only keeping relevant states
            self.states = np.array(states)[indices_to_keep]

            self.states_count = len(self.states)
            print(f"Filtered states to {self.states_count}")

        self.mat_1 = mat_1
        self.mat_1_zeeman = mat_1_zeeman
        self.mat_2 = mat_2
        self.mat_2_combination = mat_2_combination

        if not keep_n1_1:
            for i in range(self.n):
                for j in range(self.n, self.states_count):
                    self.mat_2_combination[i, j] = 0
                    self.mat_2_combination[j, i] = 0

        import matplotlib.pyplot as plt
        plt.imshow(self.mat_2_combination)
        plt.colorbar()
        plt.show()

        # if not keep_n1_1:
        #     # Remove couplings from n1=0 to n1=1

    def run(self, initial_state_index: int = 3):
        self.dc_field_calculator = self.get_calculator(self.dc_field)
        tukey_timesteps = 5000
        # rf_window = tukey(tukey_timesteps, 0.3)
        rf_window = tukey(tukey_timesteps, 0.1)

        def window_fn(t):
            return rf_window(t / 1000 / self.t * tukey_timesteps)

        self.rf_field_calculator = self.get_calculator(
            self.rf_field, window_fn=window_fn,
            x=np.array([0, 20, 50, 100, self.t * 1000]),
            # x=np.array([0, 10, 20, 50, 100, self.t * 1000]),
            # x=np.array([0, 10, 20, 40, 60, self.t * 1000]),
            # x=np.array([0, 10, 20, 50, self.t * 1000]),
            # x=np.array([0, 10, 20, 30, self.t * 1000]),
        )
        self.magnetic_field_calculator = self.get_calculator(self.magnetic_field)

        t_list = np.linspace(0, self.t * 1000, self.timesteps + 1)  # Use self.t (in ms) to creat t_list in ns
        # print(t_list.min(), t_list.max(), "timeeeeeeeee")
        initial_state = qutip.basis(self.states_count, initial_state_index)
        self.initial_state_index = initial_state_index
        self.results = qutip.mesolve(
            self.get_hamiltonian,
            # self.get_hamiltonian_FULL,
            initial_state, t_list, c_ops=[],
            options=qutip.solver.Options(store_states=True, nsteps=20000),
            progress_bar=True
        )

    def get_hamiltonian(self, t: float, *args):
        """
        Calculates the Hamiltonian as a function of time, used as an argument in qutip.mesolve().
        :param t: Time (in nanoseconds)
        :param args:
        :return: qutip.Qobj containing the system Hamiltonian at the specified time
        """
        dc_field = self.dc_field_calculator(t)
        rf_field = self.rf_field_calculator(t)
        magnetic_field = self.magnetic_field_calculator(t)

        hamiltonian = self.mat_1 + magnetic_field * self.mat_1_zeeman + dc_field * self.mat_2
        eigenvalues = np.diagonal(hamiltonian)
        s = self.initial_state_index  # Index for zero-energy state: For s = 3: n1 = 0, ml = 3

        detunings = np.zeros(self.states_count)
        for i in range(2 * self.n - 1):
            if i >= self.states_count:
                break
            if i < self.n:
                detunings[i] = (i - s) * self.rf_freq - (eigenvalues[i] - eigenvalues[s])
            else:
                detunings[i] = self.rf_freq - (eigenvalues[i - self.n + 1] - eigenvalues[i]) + detunings[i - self.n + 1]

        detunings *= -1
        # Create Hamiltonian with RF couplings. Diagonal elements from detunings calculated above.
        hamiltonian_with_rf = rf_field * self.mat_2_combination + np.diagflat(detunings)
        return qutip.Qobj(hamiltonian_with_rf)

    def get_hamiltonian_FULL(self, t: float, *args):
        """
        Calculates the Hamiltonian as a function of time, used as an argument in qutip.mesolve().
        :param t: Time (in nanoseconds)
        :param args:
        :return: qutip.Qobj containing the system Hamiltonian at the specified time
        """
        dc_field = self.dc_field_calculator(t)
        rf_field = self.rf_field_calculator(t) * np.cos(t * 2 * np.pi * self.rf_freq) * 2
        magnetic_field = self.magnetic_field_calculator(t)

        hamiltonian = self.mat_1 + magnetic_field * self.mat_1_zeeman + dc_field * self.mat_2
        eigenvalues = np.diagonal(hamiltonian)
        s = self.initial_state_index  # Index for zero-energy state: For s = 3: n1 = 0, ml = 3

        detunings = np.zeros(self.states_count)
        for i in range(2 * self.n - 1):
            if i >= self.states_count:
                break
            detunings[i] = -(eigenvalues[i] - eigenvalues[s])
            continue

            # if i < self.n:
            #     detunings[i] = -(eigenvalues[i] - eigenvalues[s])
            # else:
            #     detunings[i] = -(eigenvalues[i - self.n + 1] - eigenvalues[i]) + detunings[i - self.n + 1]

            # detunings[i] =  -e[i - n + 1] + e[i] -e[i - n + 1] + e[s]
            #     = -2 * e[i - n + 1] + e[i] + e[s]
        detunings *= -1
        # Create Hamiltonian with RF couplings. Diagonal elements from detunings calculated above.
        hamiltonian_with_rf = rf_field * self.mat_2_combination + np.diagflat(detunings)
        return qutip.Qobj(hamiltonian_with_rf)

    def get_calculator(self, value: Union[float, Sequence[float]], window_fn=None, x=None):
        """Gets a function that takes t (in nanoseconds) and returns a value.

        Used for generating the DC field, RF energy, and magnetic field from parameters.
        :param value: Parameter to generate a function for.
            Either a float or a sequence of floats
        :return:
        """
        if window_fn is None:
            if isinstance(value, float) or isinstance(value, int):
                return lambda t: value
            else:
                kind = 'quadratic' if len(value) > 2 else 'linear'
                interp_x = np.linspace(0, self.t * 1000, len(value)) if x is None else x
                return interp1d(
                    interp_x,
                    value,
                    kind=kind,
                    bounds_error=False,
                    fill_value=0,
                )
        else:
            if isinstance(value, float):
                return lambda t: value * window_fn(t)
            else:
                kind = 'quadratic' if len(value) > 2 else 'linear'
                interp_x = np.linspace(0, self.t * 1000, len(value)) if x is None else x
                interp = interp1d(
                    interp_x,
                    value,
                    kind=kind,
                    bounds_error=False,
                    fill_value=0,
                )
                return lambda t: interp(t) * window_fn(t)

    def diagnostic_run(self, initial_state_index: int = 3):
        self.dc_field_calculator = self.get_calculator(self.dc_field)
        tukey_timesteps = 5000
        rf_window = tukey(tukey_timesteps, 0.3)

        def window_fn(t):
            return rf_window(t / 1000 / self.t * tukey_timesteps)

        self.rf_field_calculator = self.get_calculator(self.rf_field, window_fn=window_fn)
        self.magnetic_field_calculator = self.get_calculator(self.magnetic_field)

        t_list = np.linspace(0, self.t * 1000, self.timesteps + 1)  # Use self.t (in ms) to creat t_list in ns
        initial_state = qutip.basis(self.states_count, initial_state_index)
        self.initial_state_index = initial_state_index
        self.results = qutip.mesolve(
            self.get_hamiltonian, initial_state, t_list, c_ops=[],
            options=qutip.solver.Options(store_states=True, nsteps=2000),
            progress_bar=True
        )

        import matplotlib.pyplot as plt
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        dc_fields = []
        rf_energies = []
        magnetic_fields = []
        for t in t_list:
            dc_fields.append(self.dc_field_calculator(t))
            rf_energies.append(self.rf_field_calculator(t))
            magnetic_fields.append(self.magnetic_field_calculator(t))
        ax1.plot(t_list, np.array(dc_fields) / 100)  # Convert V/m to V/cm
        ax2.plot(t_list, np.array(rf_energies) * 10)  # Factor of 10 to convert V/m to mV/cm
        ax3.plot(t_list, np.array(magnetic_fields) * 10_000)  # Convert from Tesla to Gauss

        ax1.set_ylabel(r"$E_{\mathrm{d.c.}}$  [V $\mathrm{cm}^{-1}$]")
        ax2.set_ylabel(r"$E_{\mathrm{RF}}$  [mV $\mathrm{cm}^{-1}$]")
        ax3.set_ylabel("$B$ [G]")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    hamiltonian = "56_rubidium87"
    hamiltonian = "51_rubidium87"
    # hamiltonian = "56_strontium88"
    # sim = Simulation(
    #     n=51,
    #     hamiltonian=hamiltonian,
    #     dc_field=(379.2, 88.6, 254.3),
    #     rf_freq=162.8e6 / 1e9,
    #     rf_energy=4.094,
    #     magnetic_field=14.33 / 10_000,  # 1 Tesla = 10 000 Gauss
    #     t=0.1,
    #     timesteps=1000,
    # )
    x_opt = [164.28348402, 292.68459828, 3.48248663, 3., 4.60444935]
    x_opt = [250.32064413, 195.53905344,   3.8380068,    4.36148669 ,  3.48767457 ,  4.05001848  , 4.3267684 ,   4.45284118]
    x_opt = [183.98916679, 205.61242751,   4.1904532,    3.74984  ,    4.93255689  , 4.39462206,   4.98157147,   4.64596374]
    x_opt = [247.18429938  , 3.       ,    3.       ,    3.   ,        3.        ,   3.        ]
    x_opt = [232.53711169  , 1.          , 2.56343786  , 3.44037653  , 2.44891319  , 3.55233269]
    x_opt = [101.1623506  ,  1.   ,        1.    ,       4.22247235 ,  5.      ,     4.52998722]
    # x_opt = [239.07180192 ,  1.78212756  , 1.     ,      4.4060104 ,   2.45513773 ,  3.95135493]
    x_opt = [107.33676222 ,  4.1682727 ,   2.77443768 ,  4.2625094  ,  4.02863142  , 4.33061969]
    x_opt =  [218.83690122 ,  2.27798751 ,  2.9758423   , 0.5       ,   5.     ,      5.   ,     ]
    x_opt =  [225.07912179 ,  1.65003376 ,  2.24265907,   4.76662172  , 1.64502995 ,  4.0307883 ]
    # dc_field = x_opt[0:2]
    dc_field =(250, x_opt[0])
    rf_freq = 250e6 / 1e9
    # rf_field = x_opt[2:8]
    rf_field = x_opt[1:]
    magnetic_field = 0
    # magnetic_field = 30 / 10_000
    print(len(x_opt))
    sim = Simulation(
        n=51,
        hamiltonian=hamiltonian,
        dc_field=dc_field,
        rf_freq=rf_freq,
        rf_field=rf_field,
        magnetic_field=magnetic_field,
        t=0.2,
        timesteps=5000,
    )
    sim.setup(keep_mat_2_minus=False, keep_n1_1=True)
    sim.run(initial_state_index=3)

    import pickle
    from system.simulation.utils import get_time_str

    del sim.dc_field_calculator
    del sim.rf_field_calculator
    del sim.magnetic_field_calculator
    with open(f"{hamiltonian}_{get_time_str()}.pkl", "wb") as f:
        pickle.dump(sim, f)

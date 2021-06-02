import functools
from typing import Sequence, Union, List

import numpy as np
import qutip
from matplotlib import pyplot as plt
from scipy.constants import e as C_e, hbar as C_hbar, physical_constants
from scipy.interpolate import interp1d

from system.hamiltonians.hamiltonians import load_hamiltonian
from system.hamiltonians.utils import plot_matrices, diagonalise_for_n1n2
from system.simulation.utils import tukey
from system.states import States, Basis
from system.transformations.utils import load_transformation, transform_basis
from timer import timer


class Simulation:
    def __init__(
            self,
            n: int,
            hamiltonian: str,
            dc_field: Union[float, Sequence[float]],
            rf_freq: Union[float, Sequence[float]],
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

        self.mat_1 = None
        self.mat_1_zeeman = None
        self.mat_2 = None
        self.mat_2_combination = None

        self.dc_field_calculator = None
        self.rf_field_calculator = None
        self.rf_freq_calculator = None
        self.magnetic_field_calculator = None

        self._hamiltonian_n1n2 = None
        self.states_count = None
        self.results = None

        self._last_hamiltonian_dc_field = None  # Caching for fixed-dc protocols.

    def setup(self):
        """Sets up matrices for Hamiltonian construction."""
        with timer("Generating states"):
            if self.hamiltonian.endswith("relevant"):
                self.states = States(self.n, basis=Basis.N_L_ML_MS_RELEVANT)
                print("Loaded relevant N L ML MS states.")
            else:
                self.states = States(self.n, basis=Basis.N_L_ML_MS)
                print("Loaded N L ML MS states.")

        with timer("Loading Hamiltonian"):
            mat_1, mat_1_zeeman, mat_2, mat_2_minus, mat_2_plus = load_hamiltonian(self.hamiltonian)
            mat_2_combination = mat_2_plus + mat_2_minus

        # # Set zero-energy state (detuning)
        # zero_index = 18
        # mat_1 = np.diagflat(np.diagonal(mat_1) - mat_1[zero_index, zero_index])
        #
        # if True:
        #     # Zero out low L states.
        #     for matrix in [mat_1, mat_1_zeeman, mat_2, mat_2_minus, mat_2_plus]:
        #         matrix[:zero_index, :zero_index] = 0

        with timer("Loading transformations"):
            transform_1 = load_transformation(self.n, Basis.N_L_J_MJ_RELEVANT, Basis.N_L_ML_MS_RELEVANT)

        with timer("Applying transformation to nlmlms"):
            mat_1 = transform_basis(mat_1, transform_1)
            mat_1_zeeman = transform_basis(mat_1_zeeman, transform_1)
            mat_2 = transform_basis(mat_2, transform_1)
            # mat_2_plus = transform_basis(mat_2_plus, transform_1)
            # mat_2_minus = transform_basis(mat_2_minus, transform_1)
            mat_2_combination = transform_basis(mat_2_combination, transform_1)

        self.mat_1 = mat_1
        self.mat_1_zeeman = mat_1_zeeman
        self.mat_2 = mat_2
        # self.mat_2_plus = mat_2_plus
        # self.mat_2_minus = mat_2_minus
        self.mat_2_combination = mat_2_combination

    def new_run(self):
        self.new_setup_run()
        t_list = np.linspace(0, self.t * 1000, self.timesteps + 1)  # Use self.t (in ms) to creat t_list in ns
        # TODO: Remove test.
        # initial_state = qutip.basis(self.states_count, 0)
        initial_state = qutip.basis(self.states_count, 3)
        self.results = qutip.mesolve(
            self.new_get_hamiltonian,
            # self.new_get_hamiltonian_no_rwa,
            initial_state, t_list, c_ops=[],
            options=qutip.solver.Options(store_states=True, nsteps=20000),
            progress_bar=True
        )

    def new_setup_run(self):
        self.dc_field_calculator = self.get_calculator(self.dc_field)
        self.rf_freq_calculator = self.get_calculator(self.rf_freq)

        tukey_timesteps = 5000
        rf_window = tukey(tukey_timesteps, 0.1)

        def window_fn(t):
            return rf_window(t / 1000 / self.t * tukey_timesteps)

        self.rf_field_calculator = self.get_calculator(
            self.rf_field, window_fn=window_fn,
            # self.rf_field,  # TODO: Remove test for
        )
        self.magnetic_field_calculator = self.get_calculator(self.magnetic_field)

        self._calculate_hamiltonian_n1n2(self.dc_field_calculator(0))

    def _calculate_hamiltonian_n1n2(self, dc_field: float):
        if self._last_hamiltonian_dc_field == dc_field:
            return
        self._last_hamiltonian_dc_field = dc_field

        hamiltonian = self.mat_1 + dc_field * self.mat_2
        eigenvalues, eigenstates, transformation = diagonalise_for_n1n2(self.states, hamiltonian)
        self._hamiltonian_n1n2 = transformation @ hamiltonian @ transformation.T
        # self._mat_2_plus_n1n2 = (transformation @ self.mat_2_plus @ transformation.T)
        self._mat_2_combination_n1n2 = np.abs(transformation @ self.mat_2_combination @ transformation.T)
        self._mat_2_combination_n1n2 = (self._mat_2_combination_n1n2 + self._mat_2_combination_n1n2.T) / 2

        # Clean up couplings
        # Only keep diagonal quadrants for mat_2_plus, zero off-diagonal quadrants.

        # TODO: Remove test.
        # self._mat_2_plus_n1n2[:2, 2:] = 0
        # self._mat_2_plus_n1n2[2:, :2] = 0
        # self._mat_2_combination_n1n2[:2, 2:] = 0
        # self._mat_2_combination_n1n2[2:, :2] = 0

        # self._mat_2_plus_n1n2[:self.n, self.n:] = 0
        # self._mat_2_plus_n1n2[self.n:, :self.n] = 0
        self._mat_2_combination_n1n2[:self.n, self.n:] = 0
        self._mat_2_combination_n1n2[self.n:, :self.n] = 0

        # # Only keep off-diagonal quadrants for mat_2_minus, zero diagonal quadrants.
        # mat_2_minus[:self.n, :self.n] = 0
        # mat_2_minus[self.n:, self.n:] = 0

        self._transformation_n1n2 = transformation

        self.states_count = len(self._hamiltonian_n1n2)
        # print(f"Filtered states to {self.states_count}")

    def new_get_hamiltonian(self, t: float, *args):
        """
        Calculates the Hamiltonian as a function of time, used as an argument in qutip.mesolve().
        :param t: Time (in nanoseconds)
        :param args:
        :return: qutip.Qobj containing the system Hamiltonian at the specified time
        """
        dc_field = self.dc_field_calculator(t)
        rf_field = self.rf_field_calculator(t)
        rf_freq = self.rf_freq_calculator(t)
        # magnetic_field = self.magnetic_field_calculator(t)

        self._calculate_hamiltonian_n1n2(dc_field)

        _eigenvalues = np.diagonal(self._hamiltonian_n1n2)

        # TODO: Remove test.
        # s = 0
        s = 3
        detunings = np.zeros(len(_eigenvalues))
        for i in range(2 * self.n - 1):
            # TODO: Remove test.
            # if i >= len(_eigenvalues):
            #     continue

            if i < self.n:
                # TODO: Remove test for non-RWA matching.
                detunings[i] = (i - s) * rf_freq - (_eigenvalues[i] - _eigenvalues[s])
                # detunings[i] = (i - s) * 2 * np.pi * self.rf_freq - (_eigenvalues[i] - _eigenvalues[s])
            else:
                # TODO: Remove test for non-RWA matching.
                detunings[i] = (i - self.n + 1 - s) * rf_freq - (_eigenvalues[i] - _eigenvalues[s])
                # detunings[i] = (i - self.n + 1 - s) * 2 * np.pi * self.rf_freq - (_eigenvalues[i] - _eigenvalues[s])

        # detunings *= -1
        detunings *= -1 * 2 * np.pi

        # Construct a new Hamiltonian using the no-RF detunings along the diagonal.
        hamiltonian_with_rf = rf_field * self._mat_2_combination_n1n2 / 2 + np.diagflat(detunings)
        _hamiltonian = qutip.Qobj(hamiltonian_with_rf)
        return _hamiltonian

    def new_get_hamiltonian_no_rwa(self, t: float, *args):
        """
        Calculates the Hamiltonian as a function of time, used as an argument in qutip.mesolve().
        :param t: Time (in nanoseconds)
        :param args:
        :return: qutip.Qobj containing the system Hamiltonian at the specified time
        """
        # dc_field = self.dc_field_calculator(t)
        rf_field = self.rf_field_calculator(t) * np.cos(t * 2 * np.pi * self.rf_freq)
        # magnetic_field = self.magnetic_field_calculator(t)

        _eigenvalues = np.diagonal(self._hamiltonian_n1n2) * -1 * 2 * np.pi

        # TODO: Remove test.
        # s = 0
        s = 3
        _eigenvalues -= _eigenvalues[s]

        hamiltonian_with_rf = rf_field * self._mat_2_combination_n1n2 + np.diagflat(_eigenvalues)
        return qutip.Qobj(hamiltonian_with_rf)

    def old_setup(self):
        """
        Calculates Hamiltonians in the n1n2mlms basis, keeping only the n1 == 0 or 1 states

        :return:
        """
        with timer("Generating states"):
            states = States(self.n, basis=Basis.N1_N2_ML_MS).states

        with timer("Loading Hamiltonian"):
            mat_1, mat_1_zeeman, mat_2, mat_2_minus, mat_2_plus = load_hamiltonian(self.hamiltonian)

        # Set zero-energy state (detuning)
        zero_index = 18
        mat_1 = np.diagflat(np.diagonal(mat_1) - mat_1[zero_index, zero_index])

        if True:
            # Zero out low L states.
            for matrix in [mat_1, mat_1_zeeman, mat_2, mat_2_minus, mat_2_plus]:
                matrix[:zero_index, :zero_index] = 0

        with timer("Loading transformations"):
            transform_1 = load_transformation(self.n, Basis.N_L_J_MJ, Basis.N_L_ML_MS)
            transform_2 = load_transformation(self.n, Basis.N_L_ML_MS, Basis.N1_N2_ML_MS)
            # transform_1 = load_transformation(self.n, Basis.N_L_J_MJ_STARK_MAP_THING, Basis.N_L_ML_MS_STARK_MAP_THING)
            # transform_2 = load_transformation(self.n, Basis.N_L_ML_MS_STARK_MAP_THING, Basis.N1_N2_ML_MS)

        with timer("Applying transformation to nlmlms"):
            mat_1 = transform_basis(mat_1, transform_1)
            mat_1_zeeman = transform_basis(mat_1_zeeman, transform_1)
            mat_2 = transform_basis(mat_2, transform_1)
            mat_2_plus = transform_basis(mat_2_plus, transform_1)
            mat_2_minus = transform_basis(mat_2_minus, transform_1)

        with timer("Applying transformation to n1n2mlms"):
            mat_1 = transform_basis(mat_1, transform_2)
            mat_1_zeeman = transform_basis(mat_1_zeeman, transform_2)
            mat_2 = transform_basis(mat_2, transform_2)
            mat_2_plus = transform_basis(mat_2_plus, transform_2)
            mat_2_minus = transform_basis(mat_2_minus, transform_2)

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
            mat_2_plus = mat_2_plus[indices_to_keep, :][:, indices_to_keep]
            mat_2_minus = mat_2_minus[indices_to_keep, :][:, indices_to_keep]

            # Filter states, only keeping relevant states
            self.states = np.array(states)[indices_to_keep]

            self.states_count = len(self.states)
            print(f"Filtered states to {self.states_count}")

        # Clean up couplings
        # Only keep diagonal quadrants for mat_2_plus, zero off-diagonal quadrants.
        mat_2_plus[:self.n, self.n:] = 0
        mat_2_plus[self.n:, :self.n] = 0
        # Only keep off-diagonal quadrants for mat_2_minus, zero diagonal quadrants.
        mat_2_minus[:self.n, :self.n] = 0
        mat_2_minus[self.n:, self.n:] = 0

        # plot_matrices([mat_2_plus, mat_2_minus])
        if keep_mat_2_minus:
            mat_2_combination = mat_2_plus + mat_2_minus
        else:
            mat_2_combination = mat_2_plus
        self.mat_1 = mat_1
        self.mat_1_zeeman = mat_1_zeeman
        self.mat_2 = mat_2
        self.mat_2_plus = mat_2_plus
        self.mat_2_minus = mat_2_minus
        self.mat_2_combination = mat_2_combination

        if not keep_n1_1:
            self.mat_2_combination[:self.n, self.n:] = 0
            self.mat_2_combination[self.n:, :self.n] = 0

        # plot_matrices([mat_1, mat_1_zeeman, mat_2, mat_2_plus, mat_2_minus, mat_2_combination])
        # import matplotlib.pyplot as plt
        # plt.imshow(self.mat_2_combination)
        # plt.colorbar()
        # plt.show()

    def old_run(self, initial_state_index: int = 3):
        self.setup_run(initial_state_index)
        t_list = np.linspace(0, self.t * 1000, self.timesteps + 1)  # Use self.t (in ms) to creat t_list in ns
        # print(t_list.min(), t_list.max(), "timeeeeeeeee")
        initial_state = qutip.basis(self.states_count, initial_state_index)
        self.results = qutip.mesolve(
            # self.get_hamiltonian,
            # self.get_hamiltonian_full_1,
            self.get_hamiltonian_full_2,
            initial_state, t_list, c_ops=[],
            options=qutip.solver.Options(store_states=True, nsteps=20000),
            progress_bar=True
        )

    def old_setup_run(self, initial_state_index):
        self.dc_field_calculator = self.get_calculator(self.dc_field)
        tukey_timesteps = 5000
        # rf_window = tukey(tukey_timesteps, 0.3)
        rf_window = tukey(tukey_timesteps, 0.1)

        # rf_window = tukey(tukey_timesteps, 0.8)
        # rf_window = tukey(tukey_timesteps, 0.02)

        def window_fn(t):
            return rf_window(t / 1000 / self.t * tukey_timesteps)

        self.rf_field_calculator = self.get_calculator(
            self.rf_field, window_fn=window_fn,
            # x=np.array([0, 20, 50, 100, self.t * 1000]),
            # x=np.array([0, 10, 20, 50, 100, self.t * 1000]),
            # x=np.array([0, 10, 20, 40, 60, self.t * 1000]),
            # x=np.array([0, 10, 20, 50, self.t * 1000]),
            # x=np.array([0, 10, 20, 30, self.t * 1000]),
        )
        self.magnetic_field_calculator = self.get_calculator(self.magnetic_field)
        self.initial_state_index = initial_state_index

    def old_get_hamiltonian(self, t: float, *args):
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
        # plot_matrices([self.mat_1, dc_field * self.mat_2, np.diagflat(detunings), rf_field * self.mat_2_combination])
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
            progress_bar=True,
        )

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

    def plot_hamiltonian(self, ts: List[float], initial_state_index: int):
        self.setup_run(initial_state_index)
        for i, t in enumerate(ts):
            qobj = self.get_hamiltonian(t / 1000)
            hamiltonian = np.real(qobj.full())
            plt.figure(i)
            plt.imshow(hamiltonian)
            plt.colorbar()
        plt.show()



if __name__ == '__main__':
    # hamiltonian = "56_rubidium87"
    # hamiltonian = "51_rubidium87"
    hamiltonian = "51_rubidium87_relevant"
    # hamiltonian = "_51_rubidium87"
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
    x_opt = [250.32064413, 195.53905344, 3.8380068, 4.36148669, 3.48767457, 4.05001848, 4.3267684, 4.45284118]
    x_opt = [183.98916679, 205.61242751, 4.1904532, 3.74984, 4.93255689, 4.39462206, 4.98157147, 4.64596374]
    x_opt = [247.18429938, 3., 3., 3., 3., 3.]
    x_opt = [232.53711169, 1., 2.56343786, 3.44037653, 2.44891319, 3.55233269]
    x_opt = [101.1623506, 1., 1., 4.22247235, 5., 4.52998722]
    # x_opt = [239.07180192 ,  1.78212756  , 1.     ,      4.4060104 ,   2.45513773 ,  3.95135493]
    x_opt = [107.33676222, 4.1682727, 2.77443768, 4.2625094, 4.02863142, 4.33061969]
    x_opt = [218.83690122, 2.27798751, 2.9758423, 0.5, 5., 5., ]
    x_opt = [225.07912179, 1.65003376, 2.24265907, 4.76662172, 1.64502995, 4.0307883]
    x_opt = [154.19593645, 2.32488819, 0.5, 4.65512591, 5., 5.]
    x_opt = [173.01562685, 4.67583474, 0.74703419, 5., 0.98299447, 5.]
    x_opt = [204.3333129, 4.40068715, 0.5, 5., 0.5, 5.]
    x_opt = [206.64451663, 1.83646413, 5., 2.71537548, 2.04219031, 4.57671419]
    x_opt = [113.28796163, 0.5, 5., 5., 5., 5., 5., 5.]
    x_opt = [195.00256478, 0.5, 0.96078382, 5., 5., 3.72955619, 3.3752865, 2.18375995]
    x_opt = [203.19123954, 0.5, 4.75148917, 2.04259386, 2.67388021, 1.1724471, 5., 5., 2.79701756, 4.6594936]
    x_opt = [208.4025186, 0.5, 0.5, 0.5, 5., 5., 5., 3.75605121]
    x_opt = [241.01725763 ,235.82679791   ,1.78421868   ,4.61832604,   5.     ,      4.43368171  , 0.5      ,    1.28699744 ,  1.93644116]
    x_opt = [247.16910072, 217.11269375 , 1.89720091 ,  0.5    ,      5.     ,      5.        ,   5.       ,    5.  ,         4.44835669]


    # dc_field = x_opt[0:2]
    # dc_field = (230, x_opt[0])
    rf_freq = 230e6 / 1e9
    # rf_field = x_opt[2:]
    # rf_field = x_opt[1:]

    # rf_field = [0.56758042, 1.43077844, 5.      ,   4.6977703 , 4.33251981]
    # rf_field = [0.5    ,   1.40637859, 4.95560548 ,5.    ,     3.15624901]
    # rf_field = [5.       ,  0.5    ,    5.      ,   5.,         4.77343123]
    # rf_field = [0.5    ,    3.03856308 ,3.48458768 ,5.   ,      1.67365681]
    # dc_field = (234.5, 234.5)
    # rf_field = [1.43269537, 0.96644762, 5.       ,  4.98124798, 3.90756422]
    rf_field = [4.26738891, 4.56526325 ,4.85637474, 0.5     ,   5.        ]
    rf_field = [2.39128913, 0.80504716, 4.480454 ,  3.58821682, 4.51838017]
    rf_field = [4.6, 4.6]
    rf_field = [4.35850331 ,4.42511603, 4.97649871, 4.98473686, 4.10886309]
    rf_field = [2.928174  , 4.85094192, 4.80427191, 4.06389658 ,4.72874359]
    dc_field = 230

    dc_field = [250, 200]
    rf_field = (4.0, 4.6, 4.6, 4.6, 4.0)

    magnetic_field = 0
    # magnetic_field = 30 / 10_000
    # print(len(x_opt))
    sim = Simulation(
        n=51,
        hamiltonian=hamiltonian,
        dc_field=dc_field,
        rf_freq=rf_freq,
        rf_field=rf_field,
        magnetic_field=magnetic_field,
        # t=0.08,
        # t=0.15,
        t=1.5,
        timesteps=5000,
    )
    sim.setup()
    sim.new_run()

    # sim.setup(keep_mat_2_minus=False, keep_n1_1=True)
    # sim.plot_hamiltonian(ts=[0, 0.01, 0.05, 0.08], initial_state_index=2)
    # sim.plot_hamiltonian(ts=[0.05], initial_state_index=2)
    # sim.run(initial_state_index=3)

    import pickle
    from system.simulation.utils import get_time_str

    # del sim.dc_field_calculator
    del sim.rf_field_calculator
    del sim.magnetic_field_calculator
    with open(f"saved_simulations/{hamiltonian}_{get_time_str()}_.pkl", "wb") as f:
        pickle.dump(sim, f)

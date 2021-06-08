from typing import Sequence, Union

import numpy as np
import qutip
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
            t: float,
            timesteps: int,
    ):
        """

        :param n: Principal quantum number
        :param hamiltonian: String describing the pre-generated atomic Hamiltonian data
        :param dc_field: Strength of DC field in units of V / m.
        :param rf_freq: F_{RF} in units of GHz
        :param rf_field: E_{RF} in units of V / m
        :param t: Protocol duration in units of microseconds
        :param timesteps: Number of equally-spaced points in time at which the system state is calculated
        """
        self.n = n
        self.hamiltonian = hamiltonian
        self.dc_field = dc_field
        self.rf_freq = rf_freq
        self.rf_field = rf_field
        self.t = t
        self.timesteps = timesteps

        self.mat_1 = None
        self.mat_1_zeeman = None
        self.mat_2 = None
        self.mat_2_combination = None

        self.dc_field_calculator = None
        self.rf_field_calculator = None
        self.rf_freq_calculator = None

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
        self.setup_run()
        t_list = np.linspace(0, self.t * 1000, self.timesteps + 1)  # Use self.t (in ms) to creat t_list in ns
        # TODO: Remove test.
        # initial_state = qutip.basis(self.states_count, 0)
        initial_state = qutip.basis(self.states_count, 3)
        self.results = qutip.mesolve(
            self.get_hamiltonian,
            # self.get_hamiltonian_no_rwa,
            initial_state, t_list, c_ops=[],
            options=qutip.solver.Options(store_states=True, nsteps=20000),
            progress_bar=True
        )

    def setup_run(self):
        self.dc_field_calculator = self.get_calculator(self.dc_field)
        self.rf_freq_calculator = self.get_calculator(self.rf_freq)

        tukey_timesteps = 5000
        rf_window = tukey(tukey_timesteps, 0.1)  # Define Tukey window width (alpha = 0.1)

        def window_fn(t):
            return rf_window(t / 1000 / self.t * tukey_timesteps)

        self.rf_field_calculator = self.get_calculator(
            self.rf_field, window_fn=window_fn,
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
        # self._mat_2_combination_n1n2[:2, 2:] = 0
        # self._mat_2_combination_n1n2[2:, :2] = 0
        self._mat_2_combination_n1n2[:self.n, self.n:] = 0
        self._mat_2_combination_n1n2[self.n:, :self.n] = 0

        self._transformation_n1n2 = transformation
        self.states_count = len(self._hamiltonian_n1n2)

    def get_hamiltonian(self, t: float, *args):
        """
        Calculates the Hamiltonian as a function of time, used as an argument in qutip.mesolve().
        :param t: Time (in nanoseconds)
        :param args:
        :return: qutip.Qobj containing the system Hamiltonian at the specified time
        """
        dc_field = self.dc_field_calculator(t)
        rf_field = self.rf_field_calculator(t)
        rf_freq = self.rf_freq_calculator(t)

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

    def get_hamiltonian_no_rwa(self, t: float, *args):
        """
        Calculates the Hamiltonian as a function of time, used as an argument in qutip.mesolve().
        :param t: Time (in nanoseconds)
        :param args:
        :return: qutip.Qobj containing the system Hamiltonian at the specified time
        """
        rf_field = self.rf_field_calculator(t) * np.cos(t * 2 * np.pi * self.rf_freq)

        _eigenvalues = np.diagonal(self._hamiltonian_n1n2) * -1 * 2 * np.pi

        # TODO: Remove test.
        # s = 0
        s = 3
        _eigenvalues -= _eigenvalues[s]

        hamiltonian_with_rf = rf_field * self._mat_2_combination_n1n2 + np.diagflat(_eigenvalues)
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


if __name__ == '__main__':
    hamiltonian = "51_rubidium87_relevant"

    rf_freq = 230e6 / 1e9

    dc_field = [250, 200]
    rf_field = (4.0, 4.6, 4.6, 4.6, 4.0)

    sim = Simulation(
        n=51,
        hamiltonian=hamiltonian,
        dc_field=dc_field,
        rf_freq=rf_freq,
        rf_field=rf_field,
        t=1.5,
        timesteps=5000,
    )
    sim.setup()
    sim.new_run()


    import pickle
    from system.simulation.utils import get_time_str

    # del sim.dc_field_calculator
    del sim.rf_field_calculator
    del sim.magnetic_field_calculator
    with open(f"saved_simulations/{hamiltonian}_{get_time_str()}_.pkl", "wb") as f:
        pickle.dump(sim, f)

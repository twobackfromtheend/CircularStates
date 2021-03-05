from functools import partial
from typing import Tuple

import numpy as np

# Flush prints immediately.
from optimal_control.utils import optimise
from system.simulation.simulation import Simulation
from timer import timer

print = partial(print, flush=True)

np.set_printoptions(linewidth=200, precision=None, floatmode='maxprec')

with timer("Setting up simulation"):
    # hamiltonian = "56_rubidium87"
    hamiltonian = "51_rubidium87"
    # hamiltonian = "56_strontium88"
    n = 51
    sim = Simulation(
        n=n,
        hamiltonian=hamiltonian,
        dc_field=(185, 140),
        rf_freq=175e6 / 1e9,
        rf_energy=0.5,
        magnetic_field=15 / 10_000,
        t=0.1,
        timesteps=1000,
    )
    sim.setup()

with timer("Getting f"):
    # def figure_of_merit(sim: Simulation):
    #     """
    #     Calculates the average ml.
    #     :param sim:
    #     :return:
    #     """
    #     system = sim.results.states[-1]
    #
    #     ml_average = 0
    #     system_populations = np.abs(system.data.toarray()) ** 2
    #     for j in range(sim.states_count):
    #         n1, n2, ml, ms = sim.states[j]
    #         state_population = system_populations[j]
    #         if state_population > 0:
    #             ml_average += state_population * ml
    #     return ml_average

    def figure_of_merit(sim: Simulation):
        """
        Calculates the overlap
        :param sim:
        :return:
        """
        system = sim.results.states[-1]
        return np.abs(system.data.toarray()[n - 1]) ** 2


    def f(inputs: np.ndarray):
        """
        :param inputs: 2-dimensional array
        :return: 2-dimentional array, one-evaluation per row
        """
        outputs = []
        for input_ in inputs:
            # dc_0, dc_1, dc_2, rf_freq, rf_energy, magnetic_field = input_
            # dc_0, dc_1, dc_2, rf_freq, rf_energy_0, rf_energy_1, rf_energy_2, magnetic_field = input_
            # dc_0, dc_1, dc_2, rf_freq, rf_energy_0, rf_energy_1, rf_energy_2, rf_energy_3, rf_energy_4, magnetic_field = input_
            dc_0, dc_1, dc_2, dc_3, rf_freq, rf_energy_0, rf_energy_1, rf_energy_2, rf_energy_3, magnetic_field_0, magnetic_field_1, magnetic_field_2, magnetic_field_3 = input_
            sim.dc_field = (dc_0, dc_1, dc_2, dc_3)
            sim.rf_freq = rf_freq
            sim.rf_energy = (rf_energy_0, rf_energy_1, rf_energy_2, rf_energy_3)
            sim.magnetic_field = (magnetic_field_0, magnetic_field_1, magnetic_field_2, magnetic_field_3)

            # sim.run(initial_state_index=3)
            sim.diagnostic_run(initial_state_index=3)
            output = -figure_of_merit(sim)
            print(f"FoM: {output} for input: {input_}")
            outputs.append(output)
        return np.array(outputs)


def get_domain(
        dc_field: Tuple[float, float],
        rf_freq: Tuple[float, float],
        rf_energy: Tuple[float, float],
        magnetic_field: Tuple[float, float]
):
    return [
        {
            'name': f'dc_0',
            'type': 'continuous',
            'domain': dc_field,
        },
        {
            'name': f'dc_1',
            'type': 'continuous',
            'domain': dc_field,
        },
        {
            'name': f'dc_2',
            'type': 'continuous',
            'domain': dc_field,
        },
        {
            'name': f'dc_3',
            'type': 'continuous',
            'domain': dc_field,
        },
        {
            'name': f'rf_freq',
            'type': 'continuous',
            'domain': rf_freq,
        },
        {
            'name': f'rf_energy_0',
            'type': 'continuous',
            'domain': rf_energy,
        },
        {
            'name': f'rf_energy_1',
            'type': 'continuous',
            'domain': rf_energy,
        },
        {
            'name': f'rf_energy_2',
            'type': 'continuous',
            'domain': rf_energy,
        },
        {
            'name': f'rf_energy_3',
            'type': 'continuous',
            'domain': rf_energy,
        },
        {
            'name': f'magnetic_field_0',
            'type': 'continuous',
            'domain': magnetic_field,
        },
        {
            'name': f'magnetic_field_1',
            'type': 'continuous',
            'domain': magnetic_field,
        },
        {
            'name': f'magnetic_field_2',
            'type': 'continuous',
            'domain': magnetic_field,
        },
        {
            'name': f'magnetic_field_3',
            'type': 'continuous',
            'domain': magnetic_field,
        },
    ]

    # TODO
    # Try for ml=1 and 2 initial state
    # Fix e_rf axis label on plot
    # Fix rf_energy envelope plot (remove manually plotted envelope, just plot cos(omega_rf t)
    # Add envelope for B?

    # Add envelope for E_RF
    # Create plot with E_RF time-dependent amplitude (and envelope),
    # Add B field to top panel of plot.

    # TODO
    # Try switch off B field, fix rf_freq = 230 MHz
    # Try 3 time slices instead of 4 per control
    # Try ml = 2
    # Rename rf_energy to rf_field
    # Use 1ms duration
    # Move conversion of mat_2_plus and mat_2_minus to GHz into hamiltonian.py, to mirror mat_1 and mat_2.
    # rename Simulation.t to t_p
    # Add option to limit sigma plus and minus to be selective on n1

    # Look at m_s taking + and - instead of discarding negatives





domain = get_domain(
    dc_field=(50, 400),  # 234.5 V / m =  2.346 V / cm
    rf_freq=(150e6 / 1e9, 250e6 / 1e9),
    rf_energy=(0.01, 5.0),  # 4.6 V / m  = 46 mV / cm
    magnetic_field=(0, 30 / 10_000),
)

max_iter = 1000
exploit_iter = 50

with timer(f"Optimising f"):
    bo = optimise(
        f,
        domain,
        max_iter=max_iter,
        exploit_iter=exploit_iter,
    )

    print("x_opt", bo.x_opt)
    print("fx_opt", bo.fx_opt)

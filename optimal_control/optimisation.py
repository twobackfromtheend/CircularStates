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
    # hamiltonian = "51_rubidium87"
    hamiltonian = "51_rubidium87_relevant"
    # hamiltonian = "_51_rubidium87"
    # hamiltonian = "56_strontium88"
    n = 51
    sim = Simulation(
        n=n,
        hamiltonian=hamiltonian,
        dc_field=None,
        rf_freq=None,
        rf_field=None,
        # rf_freq=200e6 / 1e9,
        # rf_field=0.5,
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
            # dc_0, dc_1, dc_2, rf_freq, rf_field, magnetic_field = input_
            # dc_0, dc_1, dc_2, rf_freq, rf_field_0, rf_field_1, rf_field_2, magnetic_field = input_
            # dc_0, dc_1, dc_2, rf_freq, rf_field_0, rf_field_1, rf_field_2, rf_field_3, rf_field_4, magnetic_field = input_
            # dc_0, dc_1, dc_2, dc_3, rf_freq, rf_field_0, rf_field_1, rf_field_2, rf_field_3, magnetic_field_0, magnetic_field_1, magnetic_field_2, magnetic_field_3 = input_
            # dc_0, dc_1, dc_2, rf_freq, rf_field_0, rf_field_1, rf_field_2 = input_
            # dc_0, dc_1, rf_field_0, rf_field_1, rf_field_2 = input_
            # dc_0, dc_1, rf_field_0, rf_field_1, rf_field_2, rf_field_3, rf_field_4, rf_field_5 = input_
            # sim.dc_field = (dc_0, dc_1, dc_2, dc_3)
            # sim.rf_freq = rf_freq
            # sim.rf_field = (rf_field_0, rf_field_1, rf_field_2, rf_field_3)
            # sim.magnetic_field = (magnetic_field_0, magnetic_field_1, magnetic_field_2, magnetic_field_3)
            # sim.dc_field = (dc_0, dc_1)
            # dc_0, dc_1, rf_field_0, rf_field_1, rf_field_2, rf_field_3, rf_field_4, rf_field_5 = input_
            # dc_1, rf_field_0, rf_field_1, rf_field_2, rf_field_3, rf_field_4 = input_
            # dc_1, rf_field_0, rf_field_1, rf_field_2, rf_field_3, rf_field_4,  rf_field_5, rf_field_6 = input_
            # dc_0, dc_1, rf_field_0, rf_field_1, rf_field_2, rf_field_3, rf_field_4,  rf_field_5, rf_field_6 = input_
            # dc_1, rf_field_0, rf_field_1, rf_field_2, rf_field_3, rf_field_4,  rf_field_5, rf_field_6, rf_field_7, rf_field_8 = input_
            # dc_1, rf_field_0, rf_field_1, rf_field_2, rf_field_3, rf_field_4, t = input_
            # sim.dc_field = (234.5, 234.5)
            # sim.dc_field = (230, 230)
            # sim.dc_field = (235, dc_1)
            # sim.dc_field = (dc_0, dc_1)
            # rf_field_0, rf_field_1, rf_field_2, rf_field_3, rf_field_4 = input_
            # sim.dc_field = (300, 300)
            # sim.rf_field = (rf_field_0, rf_field_1, rf_field_2)
            # sim.rf_field = (rf_field_0, rf_field_1, rf_field_2, rf_field_3, rf_field_4)
            # sim.rf_field = (rf_field_0, rf_field_1, rf_field_2, rf_field_3, rf_field_4, rf_field_5)
            # sim.rf_field = (rf_field_0, rf_field_1, rf_field_2, rf_field_3, rf_field_4, rf_field_5, rf_field_6)
            # sim.rf_field = (rf_field_0, rf_field_1, rf_field_2, rf_field_3, rf_field_4, rf_field_5, rf_field_6, rf_field_7, rf_field_8)
            # sim.rf_freq = 230e6 / 1e9
            # sim.rf_freq = 240e6 / 1e9
            # sim.magnetic_field = 0
            # sim.magnetic_field = 30 / 10_000
            # sim.t = t

            # sim.dc_field = 230
            # sim.dc_field = 250
            # sim.dc_field = 210
            # rf_field_0, rf_field_1, rf_field_2, rf_field_3, rf_field_4 = input_
            # rf_freq_0, rf_freq_1, rf_freq_2, rf_field_0, rf_field_1, rf_field_2, rf_field_3, rf_field_4 = input_
            # sim.rf_field = (rf_field_0, rf_field_1, rf_field_2, rf_field_3, rf_field_4)
            # sim.rf_freq = 230e6 / 1e9
            # sim.rf_freq = (rf_freq_0, rf_freq_1, rf_freq_2)
            # sim.magnetic_field = 30 / 10_000
            # sim.magnetic_field = 0

            # dc_0, dc_1, rf_freq_0, rf_freq_1, rf_freq_2, rf_field_0, rf_field_1, rf_field_2, rf_field_3 = input_
            # sim.dc_field = (dc_0, dc_1)
            # sim.rf_field = (rf_field_0, rf_field_1, rf_field_2, rf_field_3)
            # sim.rf_freq = (rf_freq_0, rf_freq_1, rf_freq_2)
            # sim.magnetic_field = 0

            dc_0, dc_1, rf_freq, rf_field_0, rf_field_1, rf_field_2, rf_field_3 = input_
            sim.dc_field = (dc_0, dc_1)
            sim.rf_field = (rf_field_0, rf_field_1, rf_field_2, rf_field_3)
            sim.rf_freq = rf_freq
            # sim.magnetic_field = 0

            sim.new_run()
            # sim.diagnostic_run(initial_state_index=3)
            output = -figure_of_merit(sim)
            print(f"FoM: {output} for input: {input_}")
            print(f"FoM: {output} for input: {input_.tolist()}")
            outputs.append(output)
        return np.array(outputs)


def get_domain(
        dc_field: Tuple[float, float],
        rf_freq: Tuple[float, float],
        rf_field: Tuple[float, float],
        # magnetic_field: Tuple[float, float]
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
        # {
        #     'name': f'dc_2',
        #     'type': 'continuous',
        #     'domain': dc_field,
        # },
        # {
        #     'name': f'dc_3',
        #     'type': 'continuous',
        #     'domain': dc_field,
        # },
        {
            'name': f'rf_freq',
            'type': 'continuous',
            'domain': rf_freq,
        },
        # {
        #     'name': f'rf_freq_0',
        #     'type': 'continuous',
        #     'domain': rf_freq,
        # },
        # {
        #     'name': f'rf_freq_1',
        #     'type': 'continuous',
        #     'domain': rf_freq,
        # },
        # {
        #     'name': f'rf_freq_2',
        #     'type': 'continuous',
        #     'domain': rf_freq,
        # },
        {
            'name': f'rf_field_0',
            'type': 'continuous',
            'domain': rf_field,
        },
        {
            'name': f'rf_field_1',
            'type': 'continuous',
            'domain': rf_field,
        },
        {
            'name': f'rf_field_2',
            'type': 'continuous',
            'domain': rf_field,
        },
        {
            'name': f'rf_field_3',
            'type': 'continuous',
            'domain': rf_field,
        },
        # {
        #     'name': f'rf_field_4',
        #     'type': 'continuous',
        #     'domain': rf_field,
        # },
        # {
        #     'name': f'rf_field_5',
        #     'type': 'continuous',
        #     'domain': rf_field,
        # },
        # {
        #     'name': f'rf_field_6',
        #     'type': 'continuous',
        #     'domain': rf_field,
        # },
        # {
        #     'name': f'rf_field_7',
        #     'type': 'continuous',
        #     'domain': rf_field,
        # },
        # {
        #     'name': f'rf_field_8',
        #     'type': 'continuous',
        #     'domain': rf_field,
        # },
        # {
        #     'name': f't',
        #     'type': 'continuous',
        #     'domain': (0.05, 0.1),
        # },
        # {
        #     'name': f'rf_field_5',
        #     'type': 'continuous',
        #     'domain': rf_field,
        # },
        # {
        #     'name': f'magnetic_field_0',
        #     'type': 'continuous',
        #     'domain': magnetic_field,
        # },
        # {
        #     'name': f'magnetic_field_1',
        #     'type': 'continuous',
        #     'domain': magnetic_field,
        # },
        # {
        #     'name': f'magnetic_field_2',
        #     'type': 'continuous',
        #     'domain': magnetic_field,
        # },
        # {
        #     'name': f'magnetic_field_3',
        #     'type': 'continuous',
        #     'domain': magnetic_field,
        # },
    ]


domain = get_domain(
    dc_field=(200, 300),  # 234.5 V / m =  2.346 V / cm
    # dc_field=(215, 255),  # 234.5 V / m =  2.346 V / cm
    rf_freq=(180e6 / 1e9, 280e6 / 1e9),
    # rf_field=(0.5, 5.0),  # 4.6 V / m  = 46 mV / cm
    rf_field=(0.5, 50),
    # magnetic_field=(0, 30 / 10_000),
)


max_iter = 300
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


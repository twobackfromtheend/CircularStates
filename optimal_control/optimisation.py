from functools import partial
import numpy as np

# Flush prints immediately.
from optimal_control.utils import optimise
from system.simulation.simulation import Simulation
from timer import timer

print = partial(print, flush=True)

np.set_printoptions(linewidth=200, precision=None, floatmode='maxprec')

with timer("Setting up simulation"):
    hamiltonian = "56_rubidium87"
    # hamiltonian = "56_strontium88"
    n = 56
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
    def figure_of_merit(sim: Simulation):
        system = sim.results.states[-1]

        ml_average = 0
        system_populations = np.abs(system.data.toarray()) ** 2
        for j in range(sim.states_count):
            n1, n2, ml, ms = sim.states[j]
            state_population = system_populations[j]
            if state_population > 0:
                ml_average += state_population * ml
        return ml_average


    def f(inputs: np.ndarray):
        """
        :param inputs: 2-dimensional array
        :return: 2-dimentional array, one-evaluation per row
        """
        outputs = []
        for input_ in inputs:
            dc_start, dc_end, rf_freq, rf_energy, magnetic_field = input_
            sim.dc_field = (dc_start, dc_end)
            sim.rf_freq = rf_freq
            sim.rf_energy = rf_energy
            sim.magnetic_field = magnetic_field

            sim.run()
            output = -figure_of_merit(sim)
            print(f"FoM: {output} for input: {input_}")
            outputs.append(output)
        return np.array(outputs)

domain = [
    {
        'name': f'dc_start',
        'type': 'continuous',
        'domain': [50, 300],
    },
    {
        'name': f'dc_end',
        'type': 'continuous',
        'domain': [50, 300],
    },
    {
        'name': f'rf_freq',
        'type': 'continuous',
        'domain': [150e6 / 1e9, 250e6 / 1e9],
    },
    {
        'name': f'rf_energy',
        'type': 'continuous',
        'domain': [0.01, 0.5],
    },
    {
        'name': f'magnetic_field',
        'type': 'continuous',
        'domain': [0, 30 / 10_000],
    },
]

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

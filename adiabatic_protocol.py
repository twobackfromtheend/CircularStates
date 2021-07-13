import numpy as np
import qutip

from system.simulation.simulation import Simulation


def custom_run(sim):
    sim.setup_run()

    sim.dc_field_calculator = sim.get_calculator((250, 210                                                                                           ))
    sim.rf_freq_calculator = sim.get_calculator(230e6 / 1e9)
    sim.rf_field_calculator = lambda t: 3 * np.sin(np.pi * t / 1000 / sim.t)

    t_list = np.linspace(0, sim.t * 1000, sim.timesteps + 1)  # Use self.t (in ms) to create t_list in ns
    initial_state = qutip.basis(sim.states_count, 3)
    # sim.debug_plots(skip_setup=True)

    sim.results = qutip.mesolve(
        sim.get_hamiltonian,
        initial_state, t_list, c_ops=[],
        options=qutip.solver.Options(store_states=True, nsteps=20000),
        progress_bar=True
    )


if __name__ == '__main__':
    hamiltonian = "51_rubidium87_relevant"

    rf_freq = 0
    dc_field = 0
    rf_field = (0, 0)

    sim = Simulation(
        n=51,
        hamiltonian=hamiltonian,
        dc_field=dc_field,
        rf_freq=rf_freq,
        rf_field=rf_field,
        t=2,
        timesteps=5000,
    )
    sim.setup()

    custom_run(sim)

    import pickle
    from system.simulation.utils import get_time_str

    del sim.dc_field_calculator
    del sim.rf_field_calculator
    del sim.rf_freq_calculator
    with open(f"system/simulation/saved_simulations/{hamiltonian}_{get_time_str()}_.pkl", "wb") as f:
        pickle.dump(sim, f)

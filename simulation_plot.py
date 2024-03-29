from typing import List

import matplotlib.pyplot as plt
import numpy as np
import qutip
from matplotlib.colors import LogNorm

from plots.presentation.utils import save_current_fig, setup_plot, setup_upmu
from system.simulation.simulation import Simulation
from system.states import States, Basis


def custom_run(sim):
    sim.new_run()
    # sim.setup_run()
    #
    # # _raw_dc_calculator = sim.get_calculator((270, 210))
    # # sim.dc_field_calculator = lambda t: _raw_dc_calculator(t).round(1)
    # # sim.rf_freq_calculator = sim.get_calculator(225e6 / 1e9)
    # # sim.rf_field_calculator = lambda t: 30 * np.sin(np.pi * t / 1000 / sim.t)
    #
    # t_list = np.linspace(0, sim.t * 1000, sim.timesteps + 1)  # Use self.t (in ms) to create t_list in ns
    # initial_state = qutip.basis(sim.states_count, 3)
    #
    # sim.results = qutip.mesolve(
    #     sim.get_hamiltonian,
    #     initial_state, t_list, c_ops=[],
    #     options=qutip.solver.Options(store_states=True, nsteps=20000),
    #     progress_bar=True
    # )
    print(f"fidelity: {np.abs(sim.results.states[-1].data.toarray()[sim.n - 1]) ** 2}")



def plot(file_name):
    with open(f"system/simulation/saved_simulations/{file_name}", "rb") as f:
        simulation: Simulation = pickle.load(f)

    print(simulation.dc_field)
    print(simulation.rf_field)
    print(simulation.rf_freq)
    print(simulation.t)

    systems: List[qutip.Qobj] = simulation.results.states

    states = States(51, Basis.N1_N2_ML_MS).states
    indices_to_keep = []
    for i, (n1, n2, ml, ms) in enumerate(states):
        if (n1 == 0 or n1 == 1) and ml >= 0 and ms > 0:
            indices_to_keep.append(i)
    indices_to_keep = sorted(indices_to_keep, key=lambda i: (states[i][0], states[i][2]))
    states = np.array(states)[indices_to_keep]

    state_mls = [state[2] for state in states]

    max_ml = int(max(state_mls))

    t_list = np.linspace(0, simulation.t, simulation.timesteps + 1)
    system_mls = []
    system_ml_averages = []
    system_n1s = []
    for i, t in enumerate(t_list):
        system = systems[i]

        ml_average = 0
        mls = np.zeros(max_ml + 1)
        n1s = np.zeros(2)
        system_populations = np.abs(system.data.toarray()) ** 2
        for j in range(simulation.states_count):
            n1, n2, ml, ms = states[j]
            state_population = system_populations[j]
            if state_population > 0:
                ml_average += state_population * ml
                if n1 == 0:
                    mls[int(ml)] += state_population
                n1s[int(n1)] += state_population
        system_mls.append(mls)
        system_ml_averages.append(ml_average)
        system_n1s.append(n1s)

    setup_plot()
    setup_upmu()

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(5, 6), sharex='all',
        gridspec_kw={
            'hspace': 0.2,
            'left': 0.15,
            'right': 0.85,
            'top': 0.96,
            'bottom': 0.1,
        }
    )

    rf_field = np.array([simulation.rf_field_calculator(t * 1000) for t in t_list])
    rf_freq = np.array([simulation.rf_freq_calculator(t * 1000) for t in t_list])

    e_rf_t, = ax1.plot(
        t_list,
        # np.sin(t_list / t_list[-1] * np.pi) * simulation.rf_field * 10,
        np.cos(t_list * rf_freq * 1000 * 2 * np.pi) * rf_field * 10,  # Factor of 10 to convert V/m to mV/cm
        c="C0",
        lw=3,
    )
    _ax1 = ax1.twinx()

    dc_field = np.array([simulation.dc_field_calculator(t * 1000) for t in t_list])
    e_dc_t, = _ax1.plot(
        t_list,
        dc_field / 100,
        # (simulation.dc_field[0] + t_list / t_list[-1] * (simulation.dc_field[1] - simulation.dc_field[0])) / 100,
        c="C1",
        lw=3,
    )

    ax1.set_ylabel(r"$E_{\mathrm{RF}}$  [mV $\mathrm{cm}^{-1}$]")
    _ax1.set_ylabel(r"$E_{\mathrm{d.c.}}$  [V $\mathrm{cm}^{-1}$]")

    ax1.yaxis.label.set_color(e_rf_t.get_color())
    _ax1.yaxis.label.set_color(e_dc_t.get_color())
    ax1.tick_params(axis='y', colors=e_rf_t.get_color())
    _ax1.tick_params(axis='y', colors=e_dc_t.get_color())

    # lines = [e_rf_t, e_dc_t]
    # ax1.legend(lines, [l.get_label() for l in lines])

    system_mls = np.array(system_mls).T
    system_mls = np.clip(system_mls, 1e-10, 1)

    im = ax2.imshow(
        system_mls,
        aspect='auto',
        cmap=plt.get_cmap('Blues'),
        # cmap=COLORMAP, norm=NORM,
        norm=LogNorm(vmin=1e-3, vmax=1, clip=True),
        origin='lower',
        extent=(0, t_list[-1], 0, max_ml)
    )
    # plt.colorbar(mappable=im, ax=ax2)

    ax2.set_ylim((0, max_ml - 1))
    ax2.set_ylabel("$m_l$, $n_1 = 0$")

    system_n1s = np.array(system_n1s).T

    # Initial state
    ax3.plot(
        t_list,
        system_mls[3],
        label=f"$c_{3}$, $n_1 = 0$",
        lw=3,
    )

    # Circular state
    ax3.plot(
        t_list,
        system_mls[-1],
        label="$c_{n - 1}$",
        lw=3,
    )
    print(f"c_n-1: {system_mls[-1][-1]}")

    # n1 = 0
    ax3.plot(
        t_list,
        system_n1s[0],
        '--',
        label="$\sum c$, $n_1 = 0$",
        lw=3,
    )

    # n1 = 1
    ax3.plot(
        t_list,
        system_n1s[1],
        '--',
        label="$\sum c$, $n_1 = 1$",
        lw=3,
    )

    ax3.legend(fontsize='x-small')

    ax3.set_ylim((0, 1))
    ax3.set_ylabel("State Population")
    ax3.set_xlabel(r"$t$ [$\upmu$s]")

    save_current_fig(f'_simulation_{file_name}')


if __name__ == '__main__':
    hamiltonian = "51_rubidium87_relevant"

    protocol_input = "[2.16044956e+02 2.48867577e+02 2.30014513e-01 3.29910052e+01 2.42069534e+01 1.68627039e+01 1.43755753e+01]"
    protocol_input = "[2.33565509e+02 2.24556634e+02 1.89730962e-01 3.69935025e+01 2.63170221e+01 1.06195813e+01 1.54011179e+01]"
    protocol_input = [float(i) for i in protocol_input[1:-1].split(" ")]
    protocol_input = [288.1068819042522, 213.48403072230826, 0.24731960052934127, 37.71830504289914, 43.06135181958834, 38.313629941466004, 27.97088984828705]

    dc_0, dc_1, rf_freq, rf_field_0, rf_field_1, rf_field_2, rf_field_3 = protocol_input
    dc_field = (dc_0, dc_1)
    rf_field = (rf_field_0, rf_field_1, rf_field_2, rf_field_3)
    rf_freq = rf_freq

    sim = Simulation(
        n=51,
        hamiltonian=hamiltonian,
        dc_field=dc_field,
        rf_freq=rf_freq,
        rf_field=rf_field,
        t=0.1,
        timesteps=10000,
    )
    sim.setup()

    custom_run(sim)

    # import dill as pickle
    # from system.simulation.utils import get_time_str
    #
    # file_name = f"{hamiltonian}_{get_time_str()}.pkl"
    # with open(f"system/simulation/saved_simulations/{file_name}", "wb") as f:
    #     pickle.dump(sim, f)
    #
    # plot(file_name)

import pickle
from typing import List

import numpy as np
import qutip
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from plots.presentation.utils import setup_plot, save_current_fig, setup_upmu
from system.simulation.simulation import Simulation
from system.simulation.utils import tukey

filename = "56_rubidium87_2021-01-16T19_05"
filename = "56_rubidium87_2021-01-19T12_01"
filename = "56_rubidium87_2021-01-19T13_44"
filename = "56_rubidium87_2021-01-19T13_58"
# filename = "56_rubidium87_2021-01-19T14_14"
filename = "56_rubidium87_2021-01-19T15_03"
filename = "56_rubidium87_2021-01-24T13_33"
filename = "56_rubidium87_2021-01-24T14_49"
filename = "56_rubidium87_2021-01-28T17_55"
filename = "56_rubidium87_2021-01-29T01_18"
filename = "51_rubidium87_2021-02-10T17_42"
filename = "51_rubidium87_2021-02-18T15_01"
filename = "51_rubidium87_2021-02-22T06_22"
filename = "51_rubidium87_2021-02-22T06_22"
filename = "51_rubidium87_2021-02-25T15_05"
filename = "51_rubidium87_2021-03-11T06_50"
filename = "51_rubidium87_2021-03-11T06_43"
filename = "51_rubidium87_2021-03-11T15_26"
filename = "51_rubidium87_2021-03-12T17_06"
filename = "51_rubidium87_2021-03-12T17_15"
filename = "51_rubidium87_2021-03-12T17_29"
filename = "51_rubidium87_2021-03-12T17_42"
filename = "51_rubidium87_2021-03-12T18_03"
filename = "51_rubidium87_2021-03-16T15_47"
filename = "51_rubidium87_2021-03-16T16_37"
filename = "51_rubidium87_2021-03-16T16_50"
filename = "51_rubidium87_2021-03-16T17_00"
filename = "51_rubidium87_2021-03-17T01_14"
filename = "51_rubidium87_2021-03-17T11_42"
filename = "51_rubidium87_2021-03-18T08_18"
filename = "51_rubidium87_2021-03-18T08_18"
filename = "51_rubidium87_2021-03-18T12_52"
filename = "51_rubidium87_2021-03-18T14_33"
filename = "51_rubidium87_2021-03-23T13_32"
filename = "51_rubidium87_2021-03-30T06_55"
filename = "51_rubidium87_2021-03-30T07_06"
filename = "51_rubidium87_2021-04-06T15_02"

with open(f"../../system/simulation/{filename}.pkl", "rb") as f:
    simulation: Simulation = pickle.load(f)

if not hasattr(simulation, 'rf_field'):
    # Migrate old name
    simulation.rf_field = simulation.rf_energy

print(simulation.dc_field)
print(simulation.rf_field)
print(simulation.rf_freq)
print(simulation.t)

systems: List[qutip.Qobj] = simulation.results.states

state_mls = [state[2] for state in simulation.states]
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
        n1, n2, ml, ms = simulation.states[j]
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

# ax1.plot(
#     t_list,
#     system_ml_averages,
# )


tukey_timesteps = 5000
rf_window = tukey(timesteps=tukey_timesteps, alpha=0.3)


def window_fn(t):
    return rf_window(t / 1000 / simulation.t * tukey_timesteps)


rf_field_calculator = simulation.get_calculator(simulation.rf_field, window_fn=window_fn)
rf_field = np.array([rf_field_calculator(t * 1000) for t in t_list])
e_rf_t, = ax1.plot(
    t_list,
    # np.sin(t_list / t_list[-1] * np.pi) * simulation.rf_field * 10,
    np.cos(t_list * simulation.rf_freq * 1000 * 2 * np.pi) * rf_field * 10,  # Factor of 10 to convert V/m to mV/cm
    c="C0",
    lw=3,
)
_ax1 = ax1.twinx()

dc_field_calculator = simulation.get_calculator(simulation.dc_field)
dc_field = np.array([dc_field_calculator(t * 1000) for t in t_list])
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
    # norm=LogNorm(vmin=1e-3, vmax=1, clip=True),
    origin='lower',
    extent=(0, t_list[-1], 0, max_ml)
)
# plt.colorbar(mappable=im, ax=ax2)

ax2.set_ylim((0, max_ml - 1))
ax2.set_ylabel("$m_l$, $n_1 = 0$")

system_n1s = np.array(system_n1s).T

ax3.plot(
    t_list,
    system_mls[simulation.initial_state_index],
    label=f"$c_{simulation.initial_state_index}$, $n_1 = 0$",
    lw=3,
)
ax3.plot(
    t_list,
    system_mls[-1],
    label="$c_{n - 1}$",
    lw=3,
)
ax3.plot(
    t_list,
    system_n1s[0],
    '--',
    label="$\sum c$, $n_1 = 0$",
    lw=3,
)
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

# plt.tight_layout()

save_current_fig(f'_simulation_{filename}')

# plt.show()

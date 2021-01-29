import pickle
from typing import List

import numpy as np
import qutip
from matplotlib.colors import LogNorm

from system.simulation.simulation import Simulation

# with open("56_rubidium87.pkl", "rb") as f:
# with open("56_rubidium87_2021-01-14T15_09.pkl", "rb") as f:
# with open("56_rubidium87_2021-01-14T15_23.pkl", "rb") as f:
# with open("56_rubidium87_2021-01-14T15_36.pkl", "rb") as f:
# with open("56_rubidium87_2021-01-14T16_09.pkl", "rb") as f:
# with open("56_rubidium87_2021-01-14T16_26.pkl", "rb") as f:
# with open("56_rubidium87_2021-01-14T19_58.pkl", "rb") as f:
# with open("56_rubidium87_2021-01-14T20_23.pkl", "rb") as f:
# with open("56_rubidium87_2021-01-16T12_01.pkl", "rb") as f:
# with open("56_rubidium87_2021-01-16T19_05.pkl", "rb") as f:
# with open("56_rubidium87_2021-01-17T20_18.pkl", "rb") as f:
# with open("56_rubidium87_2021-01-17T20_49.pkl", "rb") as f:
# with open("56_rubidium87_2021-01-24T13_33.pkl", "rb") as f:
with open("56_rubidium87_2021-01-28T17_47.pkl", "rb") as f:
    simulation: Simulation = pickle.load(f)

print(simulation.dc_field)
print(simulation.rf_energy)

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
            mls[int(ml)] += state_population
            n1s[int(n1)] += state_population
    system_mls.append(mls)
    system_ml_averages.append(ml_average)
    system_n1s.append(n1s)

import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(4, 6), sharex='all')

ax1.plot(
    t_list,
    system_ml_averages,
)

system_mls = np.array(system_mls).T
system_mls = np.clip(system_mls, 1e-10, 1)

ax2.imshow(
    system_mls,
    aspect='auto',
    # cmap=COLORMAP, norm=NORM,
    norm=LogNorm(vmin=1e-3, vmax=1, clip=True),
    origin='lower',
    extent=(0, t_list[-1], 0, max_ml)
)

system_n1s = np.array(system_n1s).T

ax3.plot(
    t_list,
    system_mls[3],
    label="$c_3$, $n_1 = 0$"
)
ax3.plot(
    t_list,
    system_mls[-1],
    label="$c_{H}$"  # Highest state populated
)
ax3.plot(
    t_list,
    system_n1s[1],
    label="$\sum c$, $n_1 = 1$"
)
ax3.legend()

ax1.set_ylim((0, max_ml - 1))
ax2.set_ylim((0, max_ml - 1))
ax3.set_ylim((0, 1))

ax1.set_ylabel("Average $m_l$")
ax2.set_ylabel("$m_l$")
ax3.set_ylabel("State Population")
plt.tight_layout()
plt.show()

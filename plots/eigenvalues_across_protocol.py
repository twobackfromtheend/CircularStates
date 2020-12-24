import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from system.hamiltonians.hamiltonians import load_hamiltonian
from system.states import States, Basis
from system.transformations.utils import load_transformation, transform_basis
from timer import timer

n = 56

with timer("Generating states"):
    states_n_l_ml_ms = States(n, basis=Basis.N_L_ML_MS).states
    states = States(n, basis=Basis.N1_N2_ML_MS).states

for i, (_n, _l, _ml, _ms) in enumerate(states_n_l_ml_ms):
    if _l == 3 and _ml == 3:
        state_index_F_ml_3 = i
        break
else:
    raise RuntimeError("Could not find index for F ml=3 state.")

with timer("Loading Hamiltonian"):
    # mat_1, mat_2, mat_2_minus, mat_2_plus = load_hamiltonian(f"{n}_rubidium")
    mat_1, mat_2, mat_2_minus, mat_2_plus = load_hamiltonian(f"{n}_rubidium87")
    mat_2_combination = mat_2_plus + mat_2_minus
    # mat_2_combination *= C_e / C_h * 1e-9
    # mat_2_combination *= C_e * physical_constants["Bohr radius"][0]
    mat_2_combination *= 1e-9

with timer("Loading transformations"):
    transform_1 = load_transformation(n, Basis.N_L_J_MJ, Basis.N_L_ML_MS)
    transform_2 = load_transformation(n, Basis.N_L_ML_MS, Basis.N1_N2_ML_MS)

with timer("Applying transformation to nlmlms"):
    mat_1 = transform_basis(mat_1, transform_1)
    mat_2 = transform_basis(mat_2, transform_1)
    # mat_2_minus = transform_basis(mat_2_minus, transform_1)
    # mat_2_plus = transform_basis(mat_2_plus, transform_1)
    mat_2_combination = transform_basis(mat_2_combination, transform_1)

nlmlms_mat_1 = mat_1
nlmlms_mat_2 = mat_2


def get_zero_energy(dc_field):
    return nlmlms_mat_1[state_index_F_ml_3, state_index_F_ml_3] \
           + dc_field * nlmlms_mat_2[state_index_F_ml_3, state_index_F_ml_3]


with timer("Applying transformation to n1n2mlms"):
    mat_1 = transform_basis(mat_1, transform_2)
    mat_2 = transform_basis(mat_2, transform_2)
    # mat_2_minus = transform_basis(mat_2_minus, transform_2)
    # mat_2_plus = transform_basis(mat_2_plus, transform_2)
    mat_2_combination = transform_basis(mat_2_combination, transform_2)

with timer("Applying state filters"):
    indices_to_keep = []
    for i, (n1, n2, _ml, _ms) in enumerate(states):
        # if _ms > 0 and (n1 == 0 or n1 == 1) and _ml >= 0:
        # if _ms > 0 and n1 == 0 and _ml >= 0:
        if _ms > 0 and n1 == 0 and _ml >= 3:
            indices_to_keep.append(i)

    # Sort indices
    indices_to_keep = sorted(indices_to_keep, key=lambda x: (states[x][0], states[x][2]))

    mat_1 = mat_1[indices_to_keep, :][:, indices_to_keep]
    mat_2 = mat_2[indices_to_keep, :][:, indices_to_keep]
    # mat_2_minus = mat_2_minus[indices_to_keep, :][:, indices_to_keep]
    # mat_2_plus = mat_2_plus[indices_to_keep, :][:, indices_to_keep]
    mat_2_combination = mat_2_combination[indices_to_keep, :][:, indices_to_keep]
    states = np.array(states)[indices_to_keep]

    states_count = len(states)
    print(f"Filtered states to {states_count}")

timesteps = 500
# dc_fields = np.linspace(500, 100, timesteps)
# dc_fields = np.linspace(185, 140, timesteps)
dc_fields = np.linspace(200, 170, timesteps)  # V / m
tau_A = 5e-6  # s
t_p = 2 * tau_A
t = np.linspace(0, t_p, timesteps)  # s

rf_freq = 195e6 / 1e9  # GHz
e_rf = 25  # V / m

tau_A = 5e-6
e_rf_t = e_rf * np.cos(2 * np.pi * t / t_p - np.pi / 2)

energies = []
energies_with_rf = []
for i, dc_field in enumerate(tqdm(dc_fields)):
    hamiltonian = mat_1 + dc_field * mat_2

    eigenvalues = np.diagonal(hamiltonian)
    # eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

    # s = 0
    reference_zero_energy = get_zero_energy(dc_field)
    energy_diff = np.abs(eigenvalues - reference_zero_energy)
    s = np.argmin(energy_diff[:n])
    zero_energy = eigenvalues[s]
    # print(s, reference_zero_energy, energy_diff[s])
    detunings = np.zeros(states_count)
    for i in range(2 * n - 1):
        if i >= states_count:
            break
        if i < n:
            detunings[i] = (i - s) * rf_freq - (eigenvalues[i] - eigenvalues[s])
        else:
            detunings[i] = rf_freq - (eigenvalues[i - n + 1] - eigenvalues[i]) + detunings[i - n + 1]

    energies.append(detunings)

    # Construct a new Hamiltonian using the no-RF detunings along the diagonal.
    hamiltonian_with_rf = e_rf_t[i] * mat_2_combination + np.diagflat(detunings)

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_with_rf)
    energies_with_rf.append(eigenvalues)

energies = np.array(energies)
energies_with_rf = np.array(energies_with_rf)

fig, (ax1, ax2) = plt.subplots(1, 2, sharex='all', sharey='all')

for i in range(energies.shape[1]):
    ax1.scatter(
        t,
        energies[:, i],
        color='C0',
        alpha=0.1,
        s=3,
    )
for i in range(energies_with_rf.shape[1]):
    ax2.scatter(
        t,
        energies_with_rf[:, i],
        color='C0',
        alpha=0.1,
        s=3,
    )


def on_click(event):
    _x, _y = event.xdata, event.ydata
    if _x is None or _y is None:
        print("Ignoring click outside axes.")
        return

    if event.inaxes == ax1:
        dx2 = (t - _x) ** 2
        dy2 = (energies - _y) ** 2

        # Scale according to axes limits
        # If x and y axis scales differ significantly, a trivial calculation of difference will result in the smaller
        # axis scale being negligible.
        x_range = ax1.get_xlim()[1] - ax1.get_xlim()[0]
        y_range = ax1.get_ylim()[1] - ax1.get_ylim()[0]
        dx2 /= x_range ** 2
        dy2 /= y_range ** 2

        distances = np.sqrt(np.array([dx2 for _i in range(n)]).T + dy2)
        min_distance_i = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
        (t_i, e_i) = min_distance_i

        state = states[e_i]
        print(f"Click at {_x:.2f}, {_y:.2f}")
        print(f"\tFound point at {t[t_i]}, {energies[min_distance_i]}")
        print(f"\tn1: {state[0]}, n2: {state[1]}, ml: {state[2]}, ms: {state[3]}")

        components = transform_2[indices_to_keep[e_i]] ** 2

        print(f"\tContribution from:")
        sorting = np.argsort(components)
        for _i, index in enumerate(sorting[::-1]):
            contribution = components[index]
            if contribution < 0.01 or _i >= 5:
                break
            component_state = states_n_l_ml_ms[index]
            print(
                f"\t\tn: {component_state[0]}, l: {component_state[1]}, ml: {component_state[2]}, ms: {component_state[3]}"
                f" ({contribution:.5f})"
            )


cid = fig.canvas.mpl_connect('button_press_event', on_click)

plt.tight_layout()
plt.show()

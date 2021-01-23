from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from tqdm import tqdm

from scipy.constants import e as C_e, h as C_h, hbar as C_hbar, physical_constants

from system.hamiltonians.hamiltonians import load_hamiltonian
from system.hamiltonians.utils import plot_matrices
from system.states import States, Basis
from system.transformations.utils import load_transformation, transform_basis
from timer import timer

n = 56

with timer("Generating states"):
    states_n_l_ml_ms = States(n, basis=Basis.N_L_ML_MS).states
    states = States(n, basis=Basis.N1_N2_ML_MS).states

with timer("Loading Hamiltonian"):
    # mat_1, mat_2, mat_2_minus, mat_2_plus = load_hamiltonian(f"{n}_rubidium")
    mat_1, mat_2, mat_2_minus, mat_2_plus = load_hamiltonian(f"{n}_rubidium87")
    # mat_1, mat_2, mat_2_minus, mat_2_plus = load_hamiltonian(f"{n}_hydrogen")
    mat_2_combination = mat_2_plus + mat_2_minus  # Units of a0 e
    # mat_2_combination = mat_2_plus  # Units of a0 e
    mat_2_combination *= C_e * physical_constants["Bohr radius"][0] / C_hbar
    # Conversion from atomic units for dipole matrix elements to a Rabi freq in Hz
    mat_2_combination *= 1e-9  # Convert Hz to GHz

    # plot_matrices([mat_1, mat_2])

with timer("Loading transformations"):
    transform_1 = load_transformation(n, Basis.N_L_J_MJ, Basis.N_L_ML_MS)
    transform_2 = load_transformation(n, Basis.N_L_ML_MS, Basis.N1_N2_ML_MS)

with timer("Applying transformation to nlmlms"):
    mat_1 = transform_basis(mat_1, transform_1)
    mat_2 = transform_basis(mat_2, transform_1)
    mat_2_combination = transform_basis(mat_2_combination, transform_1)

with timer("Applying transformation to n1n2mlms"):
    mat_1 = transform_basis(mat_1, transform_2)
    mat_2 = transform_basis(mat_2, transform_2)
    mat_2_combination = transform_basis(mat_2_combination, transform_2)

with timer("Applying state filters"):
    indices_to_keep = []
    for i, (n1, n2, _ml, _ms) in enumerate(states):
        if _ms > 0 and n1 == 0 and _ml >= 0:
            # if _ms > 0 and n2 == 0 and _ml >= 0:
            indices_to_keep.append(i)

    # Sort indices
    # indices_to_keep = sorted(indices_to_keep, key=lambda i: (states[i][0], states[i][2]))
    indices_to_keep = sorted(indices_to_keep, key=lambda i: states[i][2])

    mat_1 = mat_1[indices_to_keep, :][:, indices_to_keep]
    mat_2 = mat_2[indices_to_keep, :][:, indices_to_keep]
    mat_2_combination = mat_2_combination[indices_to_keep, :][:, indices_to_keep]
    states = np.array(states)[indices_to_keep]

    states_count = len(states)
    print(f"Filtered states to {states_count}")

steps = 500
# dc_fields = np.linspace(140, 185, steps)  # V / m
dc_fields = np.linspace(185, 140, steps)  # V / m

rf_freq = 175e6 / 1e9  # GHz
e_rf = 0.1  # V / m

s = 3

energies = []
energies_with_rf = []
eigenvectors_with_rf = []
for i, dc_field in enumerate(tqdm(dc_fields)):
    hamiltonian = mat_1 + dc_field * mat_2

    eigenvalues = np.diagonal(hamiltonian)
    # eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

    zero_energy = eigenvalues[s]
    detunings = np.zeros(states_count)
    for i in range(2 * n - 1):
        if i >= states_count:
            break
        if i < n:
            detunings[i] = (i - s) * rf_freq - (eigenvalues[i] - eigenvalues[s])
        else:
            detunings[i] = rf_freq - (eigenvalues[i - n + 1] - eigenvalues[i]) + detunings[i - n + 1]

    detunings *= -1
    energies.append(detunings)

    # Construct a new Hamiltonian using the no-RF detunings along the diagonal.
    hamiltonian_with_rf = e_rf * mat_2_combination + np.diagflat(detunings)

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_with_rf)
    energies_with_rf.append(eigenvalues)
    eigenvectors_with_rf.append(eigenvectors)

energies = np.array(energies)
energies_with_rf = np.array(energies_with_rf)

fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3,
    sharex='all', sharey='all',
    figsize=(8, 5),
)


for i in range(energies.shape[1]):
    if i == s:
        color = 'C0'
        s_ = 3
        alpha = 0.7
        zorder = 3
    elif i == n - 1:
        color = 'C1'
        s_ = 3
        alpha = 0.7
        zorder = 3
    else:
        color = 'grey'
        s_ = 1
        alpha = 0.2
        zorder = 2
    ax1.scatter(
        dc_fields,
        energies[:, i],
        color=color,
        alpha=alpha,
        s=s_,
        zorder=zorder,
    )

eigenvectors_with_rf = np.array(eigenvectors_with_rf)  # [Timestep, eigenvectors, eigenvector]


# cmap = plt.get_cmap("RdYlBu")


def get_cmap_with_alpha(cmap: str, alpha_range: Tuple[float, float] = (0.3, 0.8)):
    # Colormap with alpha
    cmap = plt.get_cmap(cmap)
    cmap_with_alpha = cmap(np.arange(cmap.N))
    alphas = np.linspace(*alpha_range, cmap.N)
    BG = np.asarray([1., 1., 1., ])  # Background prevents banding in colorbar due to alpha

    for i in range(cmap.N):
        cmap_with_alpha[i, :-1] = cmap_with_alpha[i, :-1] * alphas[i] + BG * (1. - alphas[i])
    cmap_with_alpha = ListedColormap(cmap_with_alpha)
    return cmap_with_alpha


cmap_with_alpha = get_cmap_with_alpha("viridis_r")
for i in range(energies_with_rf.shape[1]):
    _eigenvectors = eigenvectors_with_rf[:, i, :]
    colors = []
    for eigenvector in eigenvectors_with_rf[:, :, i]:
        eigenvector_component_s = eigenvector[s] ** 2
        colors.append(cmap_with_alpha(eigenvector_component_s))

    ax2.scatter(
        dc_fields,
        energies_with_rf[:, i],
        # color='C0',
        color=colors,
        s=3,
    )

for i in range(energies_with_rf.shape[1]):
    _eigenvectors = eigenvectors_with_rf[:, i, :]
    colors = []
    for eigenvector in eigenvectors_with_rf[:, :, i]:
        eigenvector_component = eigenvector[n - 1] ** 2
        colors.append(cmap_with_alpha(eigenvector_component))

    ax3.scatter(
        dc_fields,
        energies_with_rf[:, i],
        # color='C0',
        color=colors,
        s=3,
    )

plt.colorbar(
    ScalarMappable(norm=Normalize(), cmap=cmap_with_alpha),
    ax=ax3,
)


def on_click(event):
    _x, _y = event.xdata, event.ydata
    if _x is None or _y is None:
        print("Ignoring click outside axes.")
        return

    if event.inaxes == ax1:
        dx2 = (dc_fields - _x) ** 2
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
        print(f"\tFound point at {dc_fields[t_i]}, {energies[min_distance_i]}")
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

plt.xlim(dc_fields[0], dc_fields[-1])
plt.tight_layout()
plt.show()

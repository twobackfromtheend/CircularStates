from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from tqdm import tqdm

from scipy.constants import e as C_e, h as C_h, hbar as C_hbar, physical_constants

from plots.presentation.utils import setup_plot, save_current_fig
from system.hamiltonians.hamiltonians import load_hamiltonian
from system.hamiltonians.utils import plot_matrices, diagonalise_by_ml, diagonalise_for_n1n2
from system.states import States, Basis
from system.transformations.utils import load_transformation, transform_basis
from timer import timer

plot_fig17a = False
plot_fig17b = True


n = 51

with timer("Generating states"):
    states = States(n, basis=Basis.N_L_ML_MS_RELEVANT)
    # states = States(n, basis=Basis.N_L_ML_MS)

with timer("Loading Hamiltonian"):
    mat_1, mat_1_zeeman, mat_2, mat_2_minus, mat_2_plus = load_hamiltonian(f"{n}_rubidium87_relevant")
    # mat_1, mat_1_zeeman, mat_2, mat_2_minus, mat_2_plus = load_hamiltonian(f"{n}_rubidium87")

with timer("Loading transformations"):
    transform_1 = load_transformation(n, Basis.N_L_J_MJ_RELEVANT, Basis.N_L_ML_MS_RELEVANT)
    # transform_1 = load_transformation(n, Basis.N_L_J_MJ, Basis.N_L_ML_MS)

with timer("Applying transformation to nlmlms"):
    mat_1 = transform_basis(mat_1, transform_1)
    mat_2 = transform_basis(mat_2, transform_1)
    mat_2_plus = transform_basis(mat_2_plus, transform_1)
    mat_2_minus = transform_basis(mat_2_minus, transform_1)

steps = 50
steps = 100
# dc_fields = np.linspace(140, 185, steps)  # V / m
# dc_fields = np.linspace(185, 140, steps)  # V / m
# dc_fields = np.linspace(100, 350, steps)  # V / m
dc_fields = np.linspace(350, 100, steps)  # V / m
# dc_fields = np.linspace(250, 200, steps)  # V / m

rf_freq = 230e6 / 1e9  # GHz
rf_field = 4.6  # V / m

s = 3

energies = []
energies_with_rf = []
eigenvectors_with_rf = []
for i, dc_field in enumerate(tqdm(dc_fields)):
    hamiltonian = mat_1 + dc_field * mat_2
    eigenvalues, eigenstates, transformation = diagonalise_for_n1n2(states, hamiltonian)
    hamiltonian_n1n2 = transformation @ hamiltonian @ transformation.T

    _eigenvalues = np.diagonal(hamiltonian_n1n2).copy()

    zero_energy = _eigenvalues[s]
    detunings = np.zeros(len(_eigenvalues))
    for i in range(2 * n - 1):
        if i < n:
            detunings[i] = (i - s) * rf_freq - (_eigenvalues[i] - _eigenvalues[s])
        else:
            # detunings[i] = rf_freq - (eigenvalues[i - n + 1] - eigenvalues[i]) + detunings[i - n + 1]
            detunings[i] = (i - n + 1 - s) * rf_freq - (_eigenvalues[i] - _eigenvalues[s])

    detunings *= -1
    energies.append(detunings)

    # Construct a new Hamiltonian using the no-RF detunings along the diagonal.
    mat_2_plus_n1n2 = transformation @ mat_2_plus @ transformation.T
    mat_2_minus_n1n2 = transformation @ mat_2_minus @ transformation.T
    hamiltonian_with_rf = rf_field * (mat_2_plus_n1n2 + mat_2_minus_n1n2) / 2 + np.diagflat(detunings)

    # plot_matrices([mat_2_plus_n1n2, transformation @ mat_2_minus @ transformation.T])
    # plt.show()
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_with_rf)
    energies_with_rf.append(eigenvalues)
    eigenvectors_with_rf.append(eigenvectors)

energies = np.array(energies)

setup_plot()
# fig, (ax1, ax2) = plt.subplots(
#     1, 2,
#     sharex='all', sharey='all',
#     figsize=(6, 3.5),
# )

if plot_fig17a:
    for i in range(energies.shape[1]):
        # Highlighted first and last state for n1 = 0
        # if i == s:
        #     color = 'C0'
        #     s_ = 3
        #     alpha = 0.7
        #     zorder = 3
        # elif i == n - 1:
        #     color = 'C1'
        #     s_ = 3
        #     alpha = 0.7
        #     zorder = 3
        # else:
        #     color = '#222222'
        #     s_ = 1
        #     alpha = 0.1
        #     zorder = 2

        # n1 = 0 only
        # color = '#1B9E77'
        # s_ = 3
        # alpha = 0.7
        # zorder = 3
        # if i >= n:
        #     continue

        # n1 = 1 only
        # color = '#D95F02'
        # s_ = 3
        # alpha = 0.7
        # zorder = 3
        # if i < n:
        #     continue

        if i < n:
            color = '#1B9E77'
            s_ = 3
            alpha = 0.3
            zorder = 3
        else:
            color = '#D95F02'
            s_ = 3
            alpha = 0.3
            zorder = 2
        # if i < n:
        #     marker = 'x'
        #     color = plt.get_cmap('viridis')(i / n)
        # else:
        #     continue
        plt.scatter(
            dc_fields / 100,
            energies[:, i],
            color=color,
            alpha=alpha,
            s=s_,
            zorder=zorder,
            # marker=marker,
            edgecolors='none',
        )

    plt.xlim(dc_fields[0] / 100, dc_fields[-1] / 100)
    plt.ylim(-3.5, 3.5)

    plt.ylabel('Energy [GHz]')
    plt.xlabel(r"$E_{\mathrm{d.c.}}$ [V $\mathrm{cm}^{-1}$]")
    plt.tight_layout(pad=0.5)

    save_current_fig('avoided_crossings_n1_01_coloured_small')
    plt.show()


if plot_fig17b:
    energies_with_rf = np.array(energies_with_rf)
    eigenvectors_with_rf = np.array(eigenvectors_with_rf)  # [Timestep, eigenvectors, eigenvector]

    for i in range(energies_with_rf.shape[1]):
        _eigenvectors = eigenvectors_with_rf[:, i, :]
        # colors = []
        for eigenvector in eigenvectors_with_rf[:, :, i]:
            eigenvector_component_s = eigenvector[s] ** 2
            eigenvector_component_max_ml = eigenvector[n - 1] ** 2
            # colors.append(cmap_with_alpha(eigenvector_component_s, eigenvector_component_max_ml))

        plt.scatter(
            dc_fields / 100,
            energies_with_rf[:, i],
            # color='C0',
            # color=colors,
            color='grey',
            s=3,
            alpha=0.2
        )

    plt.xlim(dc_fields[0] / 100, dc_fields[-1] / 100)
    plt.ylim(-3.5, 3.5)

    plt.ylabel('Energy [GHz]')
    plt.xlabel(r"$E_{\mathrm{d.c.}}$ [V $\mathrm{cm}^{-1}$]")
    plt.tight_layout(pad=0.5)

    save_current_fig('avoided_crossings_b')
    plt.show()

#
# plt.colorbar(
#     ScalarMappable(norm=Normalize(), cmap=cmap_with_alpha),
#     ax=ax3,
# )


plt.xlim(dc_fields[0] / 100, dc_fields[-1] / 100)
# plt.ylim(-1.5, 1.5)
plt.ylim(-3.5, 3.5)

ax1.set_ylabel('Energy [GHz]')
ax1.set_xlabel(r"$E_{\mathrm{d.c.}}$ [V $\mathrm{cm}^{-1}$]")
ax2.set_xlabel(r"$E_{\mathrm{d.c.}}$ [V $\mathrm{cm}^{-1}$]")

# ax1.set_xlabel(r"DC Field [V $\mathrm{cm}^{-1}$]")
# ax2.set_xlabel(r"DC Field [V $\mathrm{cm}^{-1}$]")

plt.tight_layout(pad=0.5)

# save_current_fig('avoided_crossings')

plt.show()

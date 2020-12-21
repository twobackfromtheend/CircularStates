import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from system.hamiltonians.hamiltonians import load_hamiltonian
from system.states import States, Basis
from system.transformations.utils import load_transformation, transform_basis

n = 35

print("Generating states")
states_n_l_ml_ms = States(n, basis=Basis.N_L_ML_MS).states
states = States(n, basis=Basis.N1_N2_ML_MS).states

for i, (_n, _l, _ml, _ms) in enumerate(states_n_l_ml_ms):
    if _l == 3 and _ml == 3:
        state_index_F_ml_3 = i
        break
else:
    raise RuntimeError("Could not find index for F ml=3 state.")

print("Loading Hamiltonian")
mat_1, mat_2, mat_2_minus, mat_2_plus = load_hamiltonian(f"{n}_rubidium")
mat_2_combination = mat_2_plus + mat_2_minus

print("Loading transformations")
transform_1 = load_transformation(n, Basis.N_L_J_MJ, Basis.N_L_ML_MS)
transform_2 = load_transformation(n, Basis.N_L_ML_MS, Basis.N1_N2_ML_MS)

print("Applying transformations")
mat_1 = transform_basis(mat_1, transform_1)
mat_2 = transform_basis(mat_2, transform_1)
mat_2_combination = transform_basis(mat_2_combination, transform_1)

nlmlms_mat_1 = mat_1
nlmlms_mat_2 = mat_2


def get_zero_energy(dc_field):
    return nlmlms_mat_1[state_index_F_ml_3, state_index_F_ml_3] \
           + dc_field * nlmlms_mat_2[state_index_F_ml_3, state_index_F_ml_3]


mat_1 = transform_basis(mat_1, transform_2)
mat_2 = transform_basis(mat_2, transform_2)
mat_2_combination = transform_basis(mat_2_combination, transform_2)

print("Applying state filters")

#
indices_to_keep = []
for i, (n1, n2, _ml, _ms) in enumerate(states):
    # if _ms > 0 and (n1 == 0 or n1 == 1) and _ml >= 0:
    if _ms > 0 and n1 == 0 and _ml >= 0:
        indices_to_keep.append(i)

# Sort indices
indices_to_keep = sorted(indices_to_keep, key=lambda x: (states[x][0], states[x][2]))

mat_1 = mat_1[indices_to_keep, :][:, indices_to_keep]
mat_2 = mat_2[indices_to_keep, :][:, indices_to_keep]
mat_2_minus = mat_2_minus[indices_to_keep, :][:, indices_to_keep]
mat_2_plus = mat_2_plus[indices_to_keep, :][:, indices_to_keep]
states = np.array(states)[indices_to_keep]

timesteps = 500
dc_fields = np.linspace(500, 100, timesteps)
t = np.linspace(0, 1e-6, timesteps)
rf_freq = 200e6 / 1e9


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
    detunings = np.zeros(2 * n - 1)
    # for i in range(2 * n - 1):
    for i in range(n):
        if i < n:
            detunings[i] = (i - s) * rf_freq - (eigenvalues[i] - eigenvalues[s])
        else:
            detunings[i] = rf_freq - (eigenvalues[i - n + 1] - eigenvalues[i]) + detunings[i - n + 1]

    energies.append(detunings[:n])

    # rf_field = np.cos(t[i] * 2 * np.pi * rf_freq) * 10
    # hamiltonian_with_rf = np.diagflat(detunings[:n])
    # hamiltonian_with_rf += rf_field * (mat_2_minus + mat_2_plus)
    #
    # # e_rf = np.cos(t[i] * 2 * np.pi * rf_freq * 1e9)
    # # ds = transform_basis(mat_2_plus + mat_2_minus, eigenvectors)
    # # hamiltonian += ds * e_rf
    # eigenvalues = []
    # for i in range(n):
    #     eigenvector = eigenvectors[:, i]
    #     energy = hamiltonian_with_rf @ eigenvector
    #     print(energy, energy.shape)
    # #     break
    # # break
    # energies_with_rf.append(eigenvalues)

energies = np.array(energies)
# energies_with_rf = np.array(energies_with_rf)

fig, (ax1, ax2) = plt.subplots(1, 2, sharex='all', sharey='all')

for i in range(energies.shape[1]):
    ax1.scatter(
        t,
        energies[:, i],
        color='C0',
        alpha=0.1,
        s=3,
    )
    # ax2.scatter(
    #     t,
    #     energies_with_rf[:, i],
    #     color='C0',
    #     alpha=0.1,
    #     s=3,
    # )

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from plots.presentation.utils import setup_plot, save_current_fig
from system.hamiltonians.hamiltonians import load_hamiltonian
from system.states import States, Basis
from system.transformations.utils import load_transformation, transform_basis
from timer import timer

n = 56

with timer("Generating states"):
    states_n_l_ml_ms = States(n, basis=Basis.N_L_ML_MS).states
    states = States(n, basis=Basis.N1_N2_ML_MS).states

with timer("Loading Hamiltonian"):
    mat_1, mat_2, mat_2_minus, mat_2_plus = load_hamiltonian("56_rubidium")
    mat_1_h, mat_2_h, mat_2_minus_h, mat_2_plus_h = load_hamiltonian("56_hydrogen")

with timer("Loading transformations"):
    transform_1 = load_transformation(n, Basis.N_L_J_MJ, Basis.N_L_ML_MS)
    transform_2 = load_transformation(n, Basis.N_L_ML_MS, Basis.N1_N2_ML_MS)

with timer("Applying transformations"):
    mat_2 = transform_basis(mat_2, transform_1)
    mat_2 = transform_basis(mat_2, transform_2)
    mat_2_h = transform_basis(mat_2_h, transform_1)
    mat_2_h = transform_basis(mat_2_h, transform_2)

with timer("Applying state filters"):
    indices_to_keep = []
    for i, (n1, n2, _ml, _ms) in enumerate(states):
        if _ms > 0 and _ml >= 0:
            indices_to_keep.append(i)

    mat_2 = mat_2[indices_to_keep, :][:, indices_to_keep]
    mat_2_h = mat_2_h[indices_to_keep, :][:, indices_to_keep]
    states = np.array(states)[indices_to_keep]

dc_field = 100

x = []
y = []
y_h = []
for i, state in enumerate(tqdm(states)):
    n1, n2, _ml, _ms = state
    x.append(_ml)
    y.append(dc_field * mat_2[i, i])
    y_h.append(dc_field * mat_2_h[i, i])

setup_plot()

plt.scatter(
    x, y,
    alpha=0.8,
    edgecolors='none',
    marker='_',
    s=12,
)

plt.scatter(
    x, y_h,
    alpha=0.3,
    edgecolors='none',
    marker='_',
    # marker='$\mathbf{--}$',
    s=12,
    c='k',
)

plt.xlabel("$m_l$")
plt.ylabel(r"$\Delta E$ [GHz]")

plt.tight_layout(pad=0.5)
save_current_fig('ladder')
plt.show()

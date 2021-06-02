import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange

from plots.presentation.utils import setup_plot, save_current_fig
from system.hamiltonians.hamiltonians import load_hamiltonian
from system.hamiltonians.utils import diagonalise_by_ml
from system.states import States, Basis
from system.transformations.utils import load_transformation, transform_basis
from timer import timer

n = 51

with timer("Generating states"):
    # states = States(n, basis=Basis.N_L_ML_MS)
    states = States(n, basis=Basis.N_L_ML_MS_RELEVANT)

with timer("Loading Hamiltonian"):
    # mat_1, mat_1_zeeman, mat_2, mat_2_minus, mat_2_plus = load_hamiltonian("51_rubidium87")
    # test = load_hamiltonian("51_rubidium87_relevant")
    mat_1, mat_1_zeeman, mat_2, mat_2_minus, mat_2_plus = load_hamiltonian("51_rubidium87_relevant")

    mat_1_h, mat_1_zeeman_h, mat_2_h, mat_2_minus_h, mat_2_plus_h = load_hamiltonian("51_hydrogen")

with timer("Loading transformations"):
    # transform = load_transformation(n, Basis.N_L_J_MJ, Basis.N_L_ML_MS)
    transform = load_transformation(n, Basis.N_L_J_MJ_RELEVANT, Basis.N_L_ML_MS_RELEVANT)

with timer("Applying transformations"):
    mat_1 = transform_basis(mat_1, transform)
    mat_2 = transform_basis(mat_2, transform)
    mat_1_h = transform_basis(mat_1_h, transform)
    mat_2_h = transform_basis(mat_2_h, transform)

dc_field = 100
hamiltonian_with_field = dc_field * mat_2
hamiltonian_with_field_h = dc_field * mat_2_h
# hamiltonian_with_field = mat_1 + dc_field * mat_2
# hamiltonian_with_field_h = mat_1_h + dc_field * mat_2_h

x = []
y = []
y_h = []

# Get eigenvalues for Hamiltonians
eigenvalues_with_field, _ = diagonalise_by_ml(states, hamiltonian_with_field)
eigenvalues_with_field_h, _ = diagonalise_by_ml(states, hamiltonian_with_field_h)
for _ml in trange(n):
    _eigenvalues_with_field_ml = eigenvalues_with_field[_ml]
    _eigenvalues_with_field_h_ml = eigenvalues_with_field_h[_ml]
    for i in range(len(_eigenvalues_with_field_ml)):
        x.append(_ml)
        y.append(_eigenvalues_with_field_ml[i])
        y_h.append(_eigenvalues_with_field_h_ml[i])

# setup_plot(figsize=(7, 5))
setup_plot()

s = 16
s = 60
plt.scatter(
    x, y,
    alpha=0.8,
    edgecolors='none',
    # marker='_',
    marker='x',
    s=s,
)

plt.scatter(
    x, y_h,
    alpha=0.3,
    edgecolors='none',
    marker='_',
    # marker='$\mathbf{--}$',
    s=s,
    c='k',
)

plt.xlabel("$m_l$")
plt.ylabel(r"$\Delta E$ [GHz]")

# plt.tight_layout(pad=0.5)
# save_current_fig('ladder')

plt.xlim(-0.5, 5.5)
plt.ylim(4, 5)
plt.tight_layout()
save_current_fig('ladder_zoomed')

plt.show()

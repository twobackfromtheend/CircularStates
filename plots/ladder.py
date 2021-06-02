import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from system.hamiltonians.hamiltonians import load_hamiltonian
from system.states import States, Basis
from system.transformations.utils import load_transformation, transform_basis
from timer import timer

n = 51

with timer("Generating states"):
    states_n_l_ml_ms = States(n, basis=Basis.N_L_ML_MS).states
    states = States(n, basis=Basis.N1_N2_ML_MS).states

with timer("Loading Hamiltonian"):
    # mat_1, mat_1_zeeman, mat_2, mat_2_minus, mat_2_plus = load_hamiltonian("51_rubidium87")
    mat_1, mat_1_zeeman, mat_2, mat_2_minus, mat_2_plus = load_hamiltonian("_51_rubidium87")

with timer("Loading transformations"):
    transform_1 = load_transformation(n, Basis.N_L_J_MJ, Basis.N_L_ML_MS)
    transform_2 = load_transformation(n, Basis.N_L_ML_MS, Basis.N1_N2_ML_MS)

with timer("Applying transformations"):
    mat_2 = transform_basis(mat_2, transform_1)
    mat_2 = transform_basis(mat_2, transform_2)

with timer("Applying state filters"):
    indices_to_keep = []
    for i, (n1, n2, _ml, _ms) in enumerate(states):
        if _ms > 0 and _ml >= 0:
            indices_to_keep.append(i)

    mat_2 = mat_2[indices_to_keep, :][:, indices_to_keep]
    states = np.array(states)[indices_to_keep]

dc_field = 230

x = []
y = []
for i, state in enumerate(tqdm(states)):
    n1, n2, _ml, _ms = state

    x.append(_ml)
    E_from_field = dc_field * mat_2[i, i]
    y.append(E_from_field)

fig = plt.figure()
plt.scatter(
    x, y,
    alpha=0.3,
    edgecolors='none'
)
plt.xlabel("$m_l$")
plt.ylabel("$\Delta E$ [GHz]")


def on_click(event):
    _x, _y = event.xdata, event.ydata
    if _x is None or _y is None:
        print("Ignoring click outside axes.")
        return

    dx2 = (np.array(x) - _x) ** 2
    dy2 = (np.array(y) - _y) ** 2

    # Scale according to axes limits
    # If x and y axis scales differ significantly, a trivial calculation of difference will result in the smaller
    # axis scale being negligible.
    ax = plt.gca()
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    dx2 /= x_range ** 2
    dy2 /= y_range ** 2

    distances = np.sqrt(dx2 + dy2)
    min_distance_i = np.argmin(distances)
    state = states[min_distance_i]
    print(f"Click at {_x:.2f}, {_y:.2f}")
    print(f"\tFound point at {x[min_distance_i]}, {y[min_distance_i]}")
    print(f"\tn1: {state[0]}, n2: {state[1]}, ml: {state[2]}, ms: {state[3]}")

    components = transform_2[indices_to_keep[min_distance_i]] ** 2

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

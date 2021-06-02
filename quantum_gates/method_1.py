import numpy as np
from scipy.constants import hbar as C_hbar
from scipy.integrate import simps as simpson
from matplotlib import pyplot as plt
from qutip import Qobj, sesolve, Options, mesolve

from timer import timer

# C3 = 2.5451803588748686  # GHz micrometer^3
C6 = 5680.08385788903  # GHz micrometer^6
C6 = 19.783556483780124  # GHz micrometer^6
C6 = 15.552919921875  # GHz micrometer^6

C6 /= 4 ** 6
# C6 /= 2 ** 6
# C6 /= 3 ** 6
print(C6)
# C6 = 3.45
# C6 = 17
# C6 = 0.02

C6 = 0.01

V6 = C6  # C6 / r^6
V6 = 0.01

rcrt = np.zeros((2, 1))
rcrt[0] = 1

rc1t = np.zeros((2, 1))
rc1t[1] = 1

t = 1e-6
# t = 50e-6


def Omega(_t: float, *args) -> float:
    if _t <= 0 or _t >= t:
        return 0
    # return 1e6
    # return 10e6
    # print('hi')
    # return 10e6
    # return 1e11
    # scaling = 1 / 100
    # scaling = 1 / 10
    # scaling = 1
    # scaling = 38.5e6
    # scaling = 100
    # scaling = 0
    # scaling = 10e6
    scaling = 10e6
    # scaling = 0.575e6
    return np.sin(_t / t * np.pi) ** 2 * scaling
    return np.sin(_t / t * np.pi) * scaling
    # return np.sin(_t / t * np.pi) * scaling * 1e9


t_list = np.linspace(0, t, 500)

Omegas = np.array([Omega(_t) for _t in t_list])
# plt.plot(
#     t_list,
#     Omegas / 1e9
# )
# plt.show()

area = simpson(Omegas, t_list)
print(f"Area: {area}")

psi_0 = Qobj(rc1t)


def get_hamiltonian(_t, *args):
    hamiltonian = np.zeros((2, 2))
    hamiltonian += Omega(_t) / 2 * (rcrt @ rc1t.T + rc1t @ rcrt.T)
    # hamiltonian += V6 * 1e9 * rcrt @ rcrt.T
    # print(f"{Omega(_t):.5f} ({_t * 1e6:.3f})")
    hamiltonian = Qobj(hamiltonian)
    return hamiltonian


# time_independent_terms = Qobj(np.zeros((2, 2)))
time_independent_terms = Qobj(np.zeros((2, 2)) + V6 * 1e9 * rcrt @ rcrt.T)
# time_independent_terms = Qobj(np.zeros((2, 2)) - V6 * 1e9 * rcrt @ rcrt.T - 6.8e9 * rc1t @ rc1t.T)
# time_independent_terms = Qobj(np.zeros((2, 2)) + 100e6 * rcrt @ rcrt.T)
Omega_coeff_terms = Qobj((rcrt @ rc1t.T + rc1t @ rcrt.T) / 2)

with timer("Solving"):
    solver = sesolve(
        # get_hamiltonian,
        [
            time_independent_terms,
            [Omega_coeff_terms, Omega]
        ],
        psi_0,
        t_list,
        options=Options(store_states=True, nsteps=20000),
        progress_bar=True,
    )


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

i0 = []
i1 = []
for state in solver.states:
    _state = np.abs(state.data.toarray().flatten())
    i0.append(_state[0] ** 2)
    i1.append(_state[1] ** 2)

ax1.plot(
    t_list,
    np.array(Omegas) / 1e6,
    lw=3,
)
ax1.set_ylabel(r"$\Omega(t)$ [MHz]")

ax2.plot(
    t_list,
    i0,
    lw=3,
    label="$c_{rr}$",
)
ax2.plot(
    t_list,
    i1,
    lw=3,
    # label="$r_c r_t$",
    label="$c_{r1}$",
)
ax2.legend()
plt.show()

print("hi")

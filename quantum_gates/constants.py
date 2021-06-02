import numpy as np
from arc import Rubidium
from scipy.constants import epsilon_0 as C_epsilon_0, physical_constants, h as C_h, elementary_charge as C_e

atom = Rubidium()

# coupling of 59 D_{3/2} m_j = 3/2 -> 51 P_{1/2} m_j = 1/2
dpDME = atom.getDipoleMatrixElement(59, 2, 1.5, 1.5, 61, 1, 0.5, 0.5, -1)
# coupling of 59 D_{3/2} m_j = 3/2 -> 57 F_{5/2} m_j = 5/2
dfDME = atom.getDipoleMatrixElement(59, 2, 1.5, 1.5, 57, 3, 2.5, 2.5, +1)
c3 = 1 / (4.0 * np.pi * C_epsilon_0) * dpDME * dfDME * C_e ** 2 * \
     (physical_constants["Bohr radius"][0]) ** 2
print(c3 / C_h)
print("C_3 = %.3f GHz (mu m)^3 " % (abs(c3) / C_h * 1.e9))


c3 = atom.getC3term(
    n=59, l=2, j=1.5,
    n1=61,l1=1, j1=0.5,
    n2=57,l2=3, j2=2.5,
)
print(c3 / C_h)
print(c3 / C_h / 1e9 * 1e6 ** 3)

d1 = atom.getRadialMatrixElement(n1=59, l1=2, j1=1.5, n2=61,l2=1, j2=0.5, s=-1)
d2 = atom.getRadialMatrixElement(n1=59, l1=2, j1=1.5, n2=57,l2=3, j2=2.5, s=+1)
d1d2 = 1 / (4.0 * np.pi * C_epsilon_0) * d1 * d2 * C_e ** 2 * \
       (physical_constants["Bohr radius"][0]) ** 2
print(d1d2 / C_h)

dpDME = atom.getDipoleMatrixElement(59, 2, 1.5, 1.5, 61, 1, 0.5, 0.5, -1)
dpDME_ = atom.getRadialMatrixElement(59, 2, 1.5, 61, 1, 0.5, -1)

print(dpDME)
print(dpDME_)



c6 = atom.getC6term(
    # n=59, l=2, j=1.5,
    # n1=61, l1=1, j1=0.5,
    # n2=57, l2=3, j2=2.5,
    n=76, l=1, j=1.5,
    n1=76, l1=1, j1=0.5,
    n2=76, l2=2, j2=2.5,
)
print(c6 / C_h / 1e9 * 1e6 ** 6)
print(c6 / C_h)


def get_C6(nRyd: int):
    C6_GHz_micm2au = 1.0 / 1.448e-19

    c6_au_ns = -(nRyd ** 11) * (11.97 - 0.8486 * nRyd + 3.385e-3 * nRyd * nRyd) / C6_GHz_micm2au * 1e3
    # [MHz [micrometers]^6]

    return c6_au_ns * 1e-30  # Hz m^6

print("akalsdka")
print(get_C6(51))
print("50000")
print(get_C6(50))
print(get_C6(50) / 1e9 * 1e6 ** 6)
print(get_C6(51) / 1e9 * 1e6 ** 6)
print(get_C6(60) / 1e9 * 1e6 ** 6)
print(get_C6(70) / 1e9 * 1e6 ** 6)


print("ENERGY")
energy = atom.getEnergy(
    n=50,
    l=0,
    j=0.5,
    s=0.5
) * C_e / C_h * 1e-9
print(energy)

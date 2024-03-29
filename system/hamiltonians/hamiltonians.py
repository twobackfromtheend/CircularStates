import logging
from pathlib import Path
from typing import Tuple

import arc
import numpy as np
from arc.alkali_atom_functions import _EFieldCoupling
from scipy.constants import e as C_e, h as C_h, hbar as C_hbar, physical_constants
from tqdm import tqdm

from system.states import States, Basis

logger = logging.getLogger("hamiltonian_generator")

GENERATED_HAMILTONIANS_FOLDER = Path(__file__).parent / "generated_hamiltonians"

C_a_0 = physical_constants["Bohr radius"][0]


def generate_matrices(n: int, stark_map: arc.StarkMap, s=0.5):
    """
    Generates matrices (described in detail below) used to construct a Hamiltonian in the n, l, j, mj basis.
    This function is modelled after arc.calculations_atom_single.StarkMap.defineBasis

    mat_1:
        Atomic energies. Diagonal elements only.
        See arc.alkali_atom_functions.AlkaliAtom.getEnergy()
    mat_2:
        E field couplings
        See arc.calculations_atom_single.StarkMap._eFieldCouplingDivE()
    mat_2_minus and mat_2_plus:
        Couplings to an RF field
        See calculate_coupling() below.
    :param n:
    :param stark_map:
    :param s:
    :return:
    """
    global wignerPrecal
    wignerPrecal = True
    stark_map.eFieldCouplingSaved = _EFieldCoupling()

    # states = States(n, Basis.N_L_J_MJ).states
    states = States(n, Basis.N_L_J_MJ_RELEVANT).states

    dimension = len(states)
    print(f"Dimension: {dimension}", flush=True)

    mat_1 = np.zeros((dimension, dimension), dtype=np.double)
    mat_1_zeeman = np.zeros((dimension, dimension), dtype=np.double)
    mat_2 = np.zeros((dimension, dimension), dtype=np.double)
    mat_2_minus = np.zeros((dimension, dimension), dtype=np.double)
    mat_2_plus = np.zeros((dimension, dimension), dtype=np.double)

    pbar = tqdm(desc="Generating matrices", total=dimension ** 2)
    for ii in range(dimension):
        pbar.update(((dimension - ii) * 2 - 1))
        n1, l1, j1, mj1 = states[ii]

        ### mat_1
        atom_energy = stark_map.atom.getEnergy(
            n=n1,
            l=l1,
            j=j1,
            s=stark_map.s
        ) * C_e / C_h * 1e-9
        mat_1[ii][ii] = atom_energy

        zeeman_energy_shift = stark_map.atom.getZeemanEnergyShift(
            l=l1,
            j=j1,
            mj=mj1,
            magneticFieldBz=1,
            s=stark_map.s
        ) / C_h * 1e-9
        mat_1_zeeman[ii][ii] = zeeman_energy_shift

        for jj in range(ii + 1, dimension):
            n2, l2, j2, mj2 = states[jj]

            ### mat_2
            coupling_1 = stark_map._eFieldCouplingDivE(
                n1=n1, l1=l1,
                j1=j1, mj1=mj1,
                n2=n2, l2=l2,
                j2=j2, mj2=mj2,
                s=stark_map.s
            ) * 1.e-9 / C_h
            # Scaling (as is also done in the arc package) so this can be multiplied by an E field (in V/m) to yield units of GHz.
            mat_2[jj][ii] = coupling_1
            mat_2[ii][jj] = coupling_1
    pbar.close()

    pbar = tqdm(desc="Generating matrices 2", total=dimension)
    for ii in range(dimension):
        for jj in range(dimension):
            n1, l1, j1, mj1 = states[ii]
            n2, l2, j2, mj2 = states[jj]

            ### mat_2_minus and mat_2_plus
            coupling_2 = calculate_coupling(
                stark_map,
                n1, l1, j1, mj1,
                n2, l2, j2, mj2,
                -1,
                s,
            ) * C_e * C_a_0 * 1.e-9 / C_hbar
            mat_2_minus[ii][jj] = coupling_2

            coupling_3 = calculate_coupling(
                stark_map,
                n1, l1, j1, mj1,
                n2, l2, j2, mj2,
                1,
                s,
            ) * C_e * C_a_0 * 1.e-9 / C_hbar
            mat_2_plus[ii][jj] = coupling_3
        pbar.update(1)

    pbar.close()
    return states, (mat_1, mat_1_zeeman, mat_2, mat_2_minus, mat_2_plus)


def calculate_coupling(stark_map, n, l, j, mj, n2, l2, j2, mj2, q, s):
    """
    Calculates dipole matrix elements.

    See FIND LASER COUPLINGS section (i.e. first half) of arc.calculations_atom_single.StarkMap.diagonalise(),
    which uses the arc.alkali_atom_functions.AlkaliAtom.getDipoleMatrixElement() method to calculate couplings to a
    laser.
    """
    if (int(abs(l2 - l)) == 1) and \
            (int(abs(j2 - j)) <= 1) and \
            (int(abs(mj2 - mj - q)) == 0):
        # logger.info(
        #     f"{n} {l} {j} {mj} <- {q} -> {n2} {l2} {j2} {mj2}"
        # )
        dipole_matrix_element = stark_map.atom.getDipoleMatrixElement(
            n, l, j, mj,
            n2, l2, j2, mj2,
            q,
            s=s
        )
        # print(f"NON ZERO: {dipole_matrix_element}")
        return dipole_matrix_element
    else:
        return 0


def load_hamiltonian(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:
    """
    Loads a Hamiltonian.
    :param name:
        Filename for the npz file to be loaded (excluding the `.npz` suffix).
        E.g. `35_rubidium`.
    :return:
    """
    file = GENERATED_HAMILTONIANS_FOLDER / f"{name}.npz"
    if not file.is_file():
        raise FileNotFoundError(
            f"Could not find {file}. "
            f"The file can be generated by running hamiltonians.py with the appropriate parameters."
        )
    with np.load(file) as data:
        return data['mat_1'], data['mat_1_zeeman'], data['mat_2'], data['mat_2_minus'], data['mat_2_plus']


if __name__ == '__main__':
    GENERATED_HAMILTONIANS_FOLDER.mkdir(exist_ok=True)

    n = 60
    # stark_map = arc.StarkMap(arc.Hydrogen())
    # stark_map = arc.StarkMap(arc.Strontium88())
    # stark_map = arc.StarkMap(arc.Rubidium())
    stark_map = arc.StarkMap(arc.Rubidium87())

    states, matrices = generate_matrices(n, stark_map)


    save_file_name = f"{n}_{stark_map.atom.__class__.__name__.lower()}_relevant.npz"
    np.savez_compressed(
        GENERATED_HAMILTONIANS_FOLDER / save_file_name,
        mat_1=matrices[0],
        mat_1_zeeman=matrices[1],
        mat_2=matrices[2],
        mat_2_minus=matrices[3],
        mat_2_plus=matrices[4],
    )

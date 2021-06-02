import logging
from pathlib import Path

import numpy as np
from arc import CG
from tqdm import trange

from system.states import States, Basis

logger = logging.getLogger("transformation_generator")

GENERATED_TRANSFORMATIONS_FOLDER = Path(__file__).parent / "generated_transformations"


def nljmj_to_nlmlms(n) -> np.ndarray:
    """
    Generates the transformation matrix for the transformation from the
    n, l, j, mj basis to the n, l, ml, ms basis.

    :param n:
    :return:
    """
    source_states = States(n, Basis.N_L_J_MJ)
    target_states = States(n, Basis.N_L_ML_MS)
    # source_states = States(n, Basis.N_L_J_MJ_RELEVANT)
    # target_states = States(n, Basis.N_L_ML_MS_RELEVANT)

    dimension = len(source_states.states)
    identity = np.identity(dimension)
    transform = np.zeros_like(identity)

    for ii in trange(dimension, desc="Convert to nlmlms"):
        _n, _l, _ml, _ms = target_states.states[ii]
        coeff_sum = 0
        for jj in range(dimension):
            __n, __l, __j, __mj = source_states.states[jj]
            if _l != __l:
                continue

            try:
                coeff = CG(
                    j1=_l, j2=0.5, j3=__j,
                    m1=_ml, m2=_ms, m3=__mj,
                )
                coeff_sum += coeff ** 2
                if coeff != 0:
                    logger.info(
                        f"j,mj to ml,ms: {coeff:.2f}.  \tn {_n}, \tl {_l}, \tml {_ml}, \tms {_ms}, \tj {__j}, \tmj {__mj}")
                    transform[ii] += coeff * identity[jj]
            except (ValueError, AttributeError):
                continue
        if abs(coeff_sum - 1) > 1e-3:
            logger.error(f"CG coeff sum discrepancy exceed threshold: {coeff_sum} ({_n}, {_l}, {_ml}, {_ms})")

    return transform


def nlmlms_to_n1n2mlms(n) -> np.ndarray:
    """
    Generates the transformation matrix for the transformation from the
    n, l, ml, ms basis to the n1, n2, ml, ms basis.

    CG coefficients generated according to:
        Park, D. Relation between the parabolic and spherical eigenfunctions of hydrogen. Z. Physik 159, 155–157 (1960).
        https://doi.org/10.1007/BF01338343

    :param n:
    :return:
    """
    source_states = States(n, Basis.N_L_ML_MS)
    target_states = States(n, Basis.N1_N2_ML_MS)

    dimension = len(source_states.states)
    identity = np.identity(dimension)
    transform = np.zeros_like(identity)

    for ii in trange(dimension, desc="Convert to n1n2"):
        n1, n2, _ml, _ms = target_states.states[ii]
        coeff_sum = 0
        for jj in range(dimension):
            __n, __l, __ml, __ms = source_states.states[jj]
            if _ms != __ms:
                continue

            # Satisfies: -K <= k1, k2 <= K, m = k1 + k2, n = 2K + 1 = n1 + n2 + |m| + 1
            k1 = (_ml + n1 - n2) / 2
            k2 = (_ml - n1 + n2) / 2
            K = (n - 1) / 2

            try:
                # See citation in docstring for source
                coeff = CG(
                    j1=K, j2=K, j3=__l,
                    m1=k1, m2=k2, m3=__ml,
                )
                coeff_sum += coeff ** 2
                if coeff != 0:
                    transform[ii] += coeff * identity[jj]
            except (ValueError, AttributeError):
                continue
        if abs(coeff_sum - 1) > 1e-3:
            logger.error(f"CG coeff sum discrepancy exceed threshold: {coeff_sum} ({n1}, {n2}, {_ml}, {_ms})")

    return transform


if __name__ == '__main__':
    GENERATED_TRANSFORMATIONS_FOLDER.mkdir(exist_ok=True)

    n = 51

    # transform = nljmj_to_nlmlms(n)
    # np.savez_compressed(GENERATED_TRANSFORMATIONS_FOLDER / f"{n}_nljmj_to_nlmlms.npz", transform)

    transform = nlmlms_to_n1n2mlms(n)
    np.savez_compressed(GENERATED_TRANSFORMATIONS_FOLDER / f"{n}_nlmlms_to_n1n2mlms.npz", transform)

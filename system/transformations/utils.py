import numpy as np
import qutip

from system.states import Basis
from system.transformations.transformation_generator import GENERATED_TRANSFORMATIONS_FOLDER


def load_transformation(n: int, source_basis: Basis, target_basis: Basis) -> np.ndarray:
    """
    Loads a transformation matrix generated with `transformation_generator`.

    :param n:
    :param source_basis:
    :param target_basis:
    :return:
    """
    if source_basis == Basis.N_L_J_MJ and target_basis == Basis.N_L_ML_MS:
        with np.load(GENERATED_TRANSFORMATIONS_FOLDER / f"{n}_nljmj_to_nlmlms.npz") as data:
            return data[data.files[0]]
    elif source_basis == Basis.N_L_ML_MS and target_basis == Basis.N1_N2_ML_MS:
        with np.load(GENERATED_TRANSFORMATIONS_FOLDER / f"{n}_nlmlms_to_n1n2mlms.npz") as data:
            return data[data.files[0]]
    else:
        raise ValueError("Unhandled basis transformation")


def transform_basis(matrix: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Transforms `matrix` according to the transformation matrix `transform`.
    Uses QuTiP to perform the transformation, but is extremely similar to the NumPy array multiplication:
        `transform @ matrix @ transform.T`
    The differences are likely due to floating-point errors.

    :param matrix:
    :param transform:
    :return:
    """
    # return qutip.Qobj(matrix).transform(transform).full().real.astype(np.float64)
    if len(matrix.shape) != 2:
        raise ValueError(f"Matrix to be transformed has to be 2D. Received shape: {matrix.shape}")
    if matrix.shape[1] > 1:
        assert matrix.shape[0] == matrix.shape[1]
        return transform @ matrix @ transform.T
    else:
        assert matrix.shape[1] == 1
        return transform @ matrix


if __name__ == '__main__':
    x = load_transformation(35, Basis.N_L_ML_MS, Basis.N1_N2_ML_MS)
    print(x)

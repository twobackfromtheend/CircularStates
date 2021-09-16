import functools
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from system.states import States, Basis


def plot_matrices(matrices):
    for i, matrix in enumerate(matrices):
        plt.figure(i)
        # print(matrix.min(), matrix.max())
        plt.imshow(
            np.abs(matrix),
            # matrix,
            interpolation='nearest',
            # norm=LogNorm(),
            # norm=LogNorm(vmin=1e-3),
            norm=LogNorm(vmin=1e-2),
            # norm=SymLogNorm(1),
        )
        plt.colorbar()
    plt.show()


def diagonalise_by_ml(states: States, hamiltonian):
    ml_indices = defaultdict(list)

    # Get list of state indices per ml
    for i, (_n, _l, _ml, _ms) in enumerate(states.states):
        # if _ml < 0 or _ms < 0:
        if _ml < 0:
            continue
        ml_indices[_ml].append(i)

    eigenvalues = {}
    eigenstates = {}
    for _ml, indices in ml_indices.items():
        # Get Hamiltonian for single ml
        hamiltonian_ml = hamiltonian[:, indices][indices, :]
        # Diagonalise Hamiltonian for ml
        # eigvals = np.linalg.eigvals(hamiltonian_ml)
        # eigvals = np.linalg.eigvalsh(hamiltonian_ml)
        # eigvals = np.linalg.eigvalsh(hamiltonian_ml, UPLO="U")

        eigvals, eigvecs = np.linalg.eigh(hamiltonian_ml)

        # # Keep only two lower-energy eigenstates.
        # eigenvalues[_ml] = eigvals[:2]
        # eigenstates[_ml] = eigvecs[:, :2]

        eigenvalues[_ml] = eigvals
        eigenstates[_ml] = eigvecs

    return eigenvalues, eigenstates


@functools.lru_cache(maxsize=None)
def get_ml_indices(states: States):
    """Gets a list of indices for each ml.

    :param states:
    :return:
    """
    if states.basis != Basis.N_L_ML_MS_RELEVANT and states.basis != Basis.N_L_ML_MS:
        raise ValueError(f"Unsupported basis for getting ml indices: {states.basis}")
    ml_indices = defaultdict(list)

    # Get list of state indices per ml
    for i, (_n, _l, _ml, _ms) in enumerate(states.states):
        if _ml < 0 or _ms < 0:
            continue
        ml_indices[_ml].append(i)
    return ml_indices


def diagonalise_for_n1n2(states: States, hamiltonian):
    n = states.n
    ml_indices = get_ml_indices(states)

    eigenvalues = {}
    eigenstates = {}
    for _ml, indices in ml_indices.items():
        # TODO: Remove test.
        # if _ml != 4 and _ml != 3:
        #     continue

        # Get Hamiltonian for single ml
        hamiltonian_ml = hamiltonian[:, indices][indices, :]
        # Diagonalise Hamiltonian for ml
        eigvals, eigvecs = np.linalg.eigh(hamiltonian_ml)

        # # Keep only two lower-energy eigenstates.
        eigenvalues[_ml] = eigvals[:2]
        eigenstates[_ml] = eigvecs[:, :2]

    transformation = np.zeros((sum(len(_v) for _v in eigenvalues.values()), len(hamiltonian)))
    for _ml in sorted(ml_indices):
        # TODO: Remove test.
        # if _ml != 4 and _ml != 3:
        #     continue

        _eigenstates_ml = eigenstates[_ml]
        _eigenvalues_ml = eigenvalues[_ml]

        _ml_indices = ml_indices[_ml]
        if _ml < n - 1:
            # Two eigenstates: n1=0, n1=1
            for i in range(len(_ml_indices)):
                index = _ml_indices[i]
                # TODO: Remove test.
                # transformation[_ml - 3, index] = _eigenstates_ml[i, 0]
                # transformation[_ml - 1, index] = _eigenstates_ml[i, 1]
                transformation[_ml, index] = _eigenstates_ml[i, 0]
                transformation[_ml + n, index] = _eigenstates_ml[i, 1]
        else:
            # Single eigenstate for n1=0; _ml == n
            for i in range(len(_ml_indices)):
                index = _ml_indices[i]
                transformation[_ml, index] = _eigenstates_ml[i, 0]

    return eigenvalues, eigenstates, transformation

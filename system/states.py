import logging
from enum import Enum, auto
from typing import Tuple, List

import numpy as np

logger = logging.getLogger("states")


class Basis(Enum):
    N_L_J_MJ = auto()
    N_L_ML_MS = auto()
    N1_N2_ML_MS = auto()


class States:
    def __init__(self, n: int, basis: Basis):
        """
        Generates the appropriate basis states. These states are exposed within the .states attribute.

        :param n:
        :param basis:
        """
        self.n = n
        self.basis = basis

        self.states: List[Tuple[float, float, float, float]] = self.generate_states(self.n, self.basis)
        logger.info(f"Generated {len(self.states)} states in basis {self.basis}.")

    @staticmethod
    def generate_states(n: int, basis: Basis):
        """
        Generates the appropriate list of states.
        Assumes s = 0.5 for Basis.N_L_J_MJ and Basis.N_L_ML_MS.

        :param n:
        :param basis:
        :return:
        """
        if basis == Basis.N_L_J_MJ:
            s = 0.5
            states = []
            for _l in range(n):
                for _j in np.linspace(_l - s, _l + s, int(2 * s + 1)):
                    for _mj in np.arange(-_j, _j + 1e-5):
                        states.append((n, _l, _j, _mj))
            return states
        elif basis == Basis.N_L_ML_MS:
            s = 0.5
            states = []
            for _l in range(n):
                for _ml in range(-_l, _l + 1):
                    for _ms in (-s, s):
                        states.append((n, _l, _ml, _ms))
            return states
        elif basis == Basis.N1_N2_ML_MS:
            states = {}  # Used as an ordered set
            for _l in range(n):
                for _ml in range(-_l, _l + 1):
                    for _ms in (-0.5, 0.5):
                        for n1 in range(n - abs(_ml) - 1 + 1):
                            n2 = n - n1 - abs(_ml) - 1
                            states[(n1, n2, _ml, _ms)] = None
            return list(states.keys())
        else:
            raise ValueError(f"Unhandled basis: {basis}")


__all__ = ['Basis', 'States']

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    States(35, Basis.N_L_J_MJ)
    States(35, Basis.N_L_ML_MS)
    States(35, Basis.N1_N2_ML_MS)

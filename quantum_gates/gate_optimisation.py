from functools import partial
from typing import List, Callable

import numpy as np
from GPyOpt.methods import BayesianOptimization
from qutip import Qobj, mesolve, Options
from scipy.integrate import simps as simpson

print = partial(print, flush=True)

METHOD = 1  # Set 1 or 2

### Method 1 Setup ###
if METHOD == 1:
    C6 = 0.01
    V6 = C6  # C6 / r^6

    rcrt = np.zeros((2, 1))
    rcrt[0] = 1

    rc1t = np.zeros((2, 1))
    rc1t[1] = 1
    psi_0 = Qobj(rc1t)


    def solve_method_1(input_):
        t = 1e-6

        def Omega(_t: float, *args) -> float:
            if _t <= 0 or _t >= t:
                return 0
            # scaling = 1 / 100
            scaling = input_
            return np.sin(_t / t * np.pi) * scaling

        t_list = np.linspace(0, t, 500)
        Omegas = np.array([Omega(_t) for _t in t_list])

        area = simpson(Omegas, t_list)
        print(f"Area: {area}")

        time_independent_terms = Qobj(np.zeros((2, 2)) + V6 * 1e9 * rcrt @ rcrt.T)
        Omega_coeff_terms = Qobj((rcrt @ rc1t.T + rc1t @ rcrt.T) / 2)

        solver = mesolve(
            [
                time_independent_terms,
                [Omega_coeff_terms, Omega]
            ],
            psi_0,
            t_list,
            options=Options(store_states=True, nsteps=20000),
        )
        c_r1 = np.abs(solver.states[-1].data[1, 0])
        return c_r1


    def method_1_f(inputs: np.ndarray):
        """
        :param inputs: 2-dimensional array
        :return: 2-dimentional array, one-evaluation per row
        """
        outputs = []
        for input_ in inputs:
            output = -solve_method_1(input_)
            outputs.append(output)
            print(f"FoM: {output} for input: {input_}")
        return np.array(outputs)


    f = method_1_f

### Method 2 Setup ###
if METHOD == 2:
    C3 = 2.5451803588748686  # GHz micrometer^3
    Vdd = C3  # C3 / r^3

    rcrt = np.zeros((3, 1))
    rcrt[0] = 1

    rc1t = np.zeros((3, 1))
    rc1t[1] = 1

    acbt = np.zeros((3, 1))
    acbt[2] = 1

    psi_0 = Qobj(rc1t)


    def solve_method_2(input_):
        t = 1e-6

        def Omega(_t: float, *args) -> float:
            if _t <= 0 or _t >= t:
                return 0
            # scaling = 1 / 100
            scaling = input_
            return np.sin(_t / t * np.pi) * scaling

        t_list = np.linspace(0, t, 500)
        Omegas = np.array([Omega(_t) for _t in t_list])

        area = simpson(Omegas, t_list)
        print(f"Area: {area}")

        time_independent_terms = Qobj(np.zeros((3, 3)) + Vdd * 1e9 * rcrt @ rcrt.T)
        Omega_coeff_terms = Qobj((rcrt @ rc1t.T + rc1t @ rcrt.T) / 2)

        solver = mesolve(
            [
                time_independent_terms,
                [Omega_coeff_terms, Omega]
            ],
            psi_0,
            t_list,
            options=Options(store_states=True, nsteps=20000),
        )
        c_r1 = np.abs(solver.states[-1].data[1, 0])
        return c_r1


    def method_2_f(inputs: np.ndarray):
        """
        :param inputs: 2-dimensional array
        :return: 2-dimentional array, one-evaluation per row
        """
        outputs = []
        for input_ in inputs:
            output = -solve_method_2(input_)
            outputs.append(output)
            print(f"FoM: {output} for input: {input_}")
        return np.array(outputs)


    f = method_1_f

domain = [
    {
        'name': f'scaling',
        'type': 'continuous',
        'domain': [0.5 / 100, 3 / 100],
    },
]

max_iter = 300
exploit_iter = 50


def optimise(
        f: Callable, domain: List[dict], max_iter: int, exploit_iter: int,
        initial_design_numdata_factor: int = 4,
        exact_feval: bool = True,
):
    """
    :param f:
        function to optimize.
        It should take 2-dimensional numpy arrays as input and return 2-dimensional outputs (one evaluation per row).
    :param domain:
        list of dictionaries containing the description of the inputs variables
        (See GPyOpt.core.task.space.Design_space class for details).
    :param max_iter:
    :param exploit_iter:
    :param initial_design_numdata_factor:
    :param exact_feval:
    :return:
    """
    bo_kwargs = {
        'domain': domain,  # box-constraints of the problem
        'acquisition_type': 'EI',  # Selects the Expected improvement
        'initial_design_numdata': initial_design_numdata_factor * len(domain),  # Number of initial points
        'exact_feval': exact_feval
    }
    print(f"bo_kwargs: {bo_kwargs}")

    bo = BayesianOptimization(
        f=f,
        **bo_kwargs
    )

    optimisation_kwargs = {
        'max_iter': max_iter,
    }
    print(f"optimisation_kwargs: {optimisation_kwargs}")
    bo.run_optimization(**optimisation_kwargs)

    print(f"Optimised result: {bo.fx_opt}")
    print(f"Optimised controls: {bo.x_opt}")

    # Exploit
    bo.acquisition_type = 'LCB'
    bo.acquisition_weight = 1e-6
    bo.kwargs['acquisition_weight'] = 1e-6

    bo.acquisition = bo._acquisition_chooser()
    bo.evaluator = bo._evaluator_chooser()

    bo.run_optimization(exploit_iter)

    print(f"Optimised result: {bo.fx_opt}")
    print(f"Optimised controls: {bo.x_opt}")
    return bo


bo = optimise(
    f,
    domain,
    max_iter=max_iter,
    exploit_iter=exploit_iter,
)

print("x_opt", bo.x_opt)
print("fx_opt", bo.fx_opt)

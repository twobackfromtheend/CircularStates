from functools import partial
from typing import Tuple, List, Callable

from GPyOpt.methods import BayesianOptimization

print = partial(print, flush=True)


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


__all__ = ['optimise']

from pathlib import Path

import matplotlib.pyplot as plt

PLOTS_FOLDER_PATH = Path(r"plots")


def setup_plot(figsize=(5, 3.5)):
    plt.rc('text', usetex=True)
    plt.rc('font', family="serif", serif="CMU Serif")
    plt.rc('figure', figsize=figsize)
    plt.rc('font', size=16)

def setup_upmu():
    LATEX_PREAMBLE = r"""
    \usepackage{upgreek}
    """
    plt.rc('text.latex', preamble=LATEX_PREAMBLE)


def save_current_fig(name: str):
    PLOTS_FOLDER_PATH.mkdir(exist_ok=True)
    plt.savefig(
        PLOTS_FOLDER_PATH / f"{name}.png",
        dpi=600
    )
    plt.close('all')

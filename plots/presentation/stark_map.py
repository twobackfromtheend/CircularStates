import arc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

from plots.presentation.utils import setup_plot, save_current_fig
from timer import timer

stark_map = arc.StarkMap(arc.Rubidium())

n = 56
l = 0
j = 0.5
mj = 0.5

n_buffer = 3
n_min = n - n_buffer
n_max = n + n_buffer
max_l = n - 1

n = 56
l = n - 1
j = 0.5
mj = j

n_buffer = 3
n_min = n - n_buffer
n_max = n + n_buffer
max_l = n - 1

with timer('defineBasis'):
    stark_map.defineBasis(
        n=n,
        l=l,
        j=j,
        mj=mj,
        nMin=n_min,
        nMax=n_max,
        maxL=max_l,
        progressOutput=True,
    )

# V / m
e_field = np.linspace(0, 3, 1000) * 100

with timer('diagonalise'):
    stark_map.diagonalise(
        eFieldList=e_field,
        progressOutput=True,
    )

setup_plot()

# stark_map.fig, stark_map.ax = plt.subplots(1, 1, figsize=(4, 3.5))
with timer('plotLevelDiagram'):
    stark_map.plotLevelDiagram(
        units=2,  # GHz
        highlighState=False,
        # highlighState=True,
        progressOutput=True,
        addToExistingPlot=True,
    )

stark_map.fig.set_size_inches(5, 3.5)

plt.xlabel(r"$E_{\mathrm{d.c.}}$ [V $\mathrm{cm}^{-1}$]")
plt.ylabel(r"State Energy [GHz]")

# plt.ylim(-1200, -1000)
# plt.ylim(-1200, -1100)
plt.ylim(-1400, -1200)

plt.tight_layout(pad=0.5)
save_current_fig('stark_map_n2')

# cm = LinearSegmentedColormap.from_list('mymap',
#                                                 ['0.9', highlightColour, 'black'])
# plt.colorbar(ScalarMappable())
plt.show()

# stark_map.showPlot(
#     interactive=False
# )

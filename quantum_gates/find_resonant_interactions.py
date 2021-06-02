import numpy as np
from arc import StarkMapResonances, Rubidium

n = 51
n = 70
n = 90
n = 30
s = 0.5
calculation = StarkMapResonances(
    Rubidium(),
    [n, n - 1, n - 1 + s, n - 1 + s],
    Rubidium(),
    [n, n - 1, n - 1 + s, n - 1 + s],
)
n_buffer = 10
calculation.findResonances(
    nMin=n - n_buffer, nMax=n + n_buffer, maxL=5000,
    eFieldList=np.linspace(0, 100, 200),
    # energyRange=[-0.8e9, 4.e9],
    energyRange=[-10e9, 10.e9],
    progressOutput=True,
)
calculation.showPlot()

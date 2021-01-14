import matplotlib.pyplot as plt


def plot_matrices(matrices):
    for i, matrix in enumerate(matrices):
        plt.figure(i)
        # print(matrix.min(), matrix.max())
        plt.imshow(
            matrix,
            interpolation='nearest',
            # norm=LogNorm(),
            # norm=SymLogNorm(1),
        )
        plt.colorbar()
    plt.show()

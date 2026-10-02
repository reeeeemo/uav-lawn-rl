import numpy as np
import matplotlib.pyplot as plt

class Grid:
    """
        n x m grid for terrain generation.
    """
    def __init__(self, polygon: np.ndarray):

        ### create fishnet grid of polygon ###
        self.polygon = polygon
        self.cell_size = 3
        self.xmin, self.ymin = polygon.min(axis=0)
        xmax, ymax = polygon.max(axis=0)
        self.n_cols = int(np.ceil((xmax - self.xmin) / self.cell_size))
        self.n_rows = int(np.ceil((ymax - self.ymin) / self.cell_size))

        # starts from bottom left
        X, Y = np.meshgrid(
            self.xmin + np.arange(self.n_cols) * self.cell_size,
            self.ymin + np.arange(self.n_rows) * self.cell_size, indexing='xy'
        )
        self.cells = np.vstack([Y.ravel(), X.ravel()])  # lower left corner of each cell
        self.uncertainties = np.zeros((self.n_rows, self.n_cols), dtype=np.float32)

    def plot(self):
        xe = self.xmin + np.arange(self.n_cols + 1) * self.cell_size
        ye = self.ymin + np.arange(self.n_rows + 1) * self.cell_size
        plt.scatter(*self.cells[::-1])
        plt.pcolormesh(xe, ye, self.uncertainties, edgecolors='k', alpha=0.5)
        plt.fill(*self.polygon.T, alpha=0.3)
        plt.show(block=True)

if __name__ == "__main__":
    grid = Grid(np.array([[0,0],[30,5],[25,20],[5,15]], dtype=np.float32))
    grid.plot()
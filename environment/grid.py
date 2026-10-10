import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.transforms import Bbox
from utils.filter import CertaintyKalmanFilter

class Grid:
    """
        n x m grid for terrain generation.
    """
    def __init__(self, polygon: np.ndarray):
        self.objects = []
        self._obj_xy = np.empty((0, 2), dtype=np.float32)
        self.cell_size = 3

        ### create fishnet grid of polygon ###
        self.polygon = polygon
        self.xmin, self.ymin = polygon.min(axis=0)
        self.xmax, self.ymax = polygon.max(axis=0)

        self.n_cols = int(np.ceil((self.xmax - self.xmin) / self.cell_size))
        self.n_rows = int(np.ceil((self.ymax - self.ymin) / self.cell_size))

        # starts from bottom left #
        X, Y = np.meshgrid(
            self.xmin + np.arange(self.n_cols) * self.cell_size,
            self.ymin + np.arange(self.n_rows) * self.cell_size, indexing='xy'
        )
        self.cell_cx = X + self.cell_size / 2
        self.cell_cy = Y + self.cell_size / 2

        # generate cells, cells inside the polygon, uncertainty values #
        self.cells = np.vstack([Y.ravel(), X.ravel()])  # lower left corner of each cell
        self.filter = CertaintyKalmanFilter((self.n_rows, self.n_cols), R=0.5)

        self.polygon_path = Path(self.polygon)
        self.inside_cells = np.array([
            self.polygon_path.intersects_bbox(Bbox.from_bounds(x, y, self.cell_size, self.cell_size))
            for x, y in zip(X.ravel(), Y.ravel())
        ]).reshape(self.n_rows, self.n_cols)

    def reset(self):
        self.filter.reset()
        self.objects = []

    def generate_objects(self, obj_type: str, num_objs: int):
        """Generates object type within the polygon boundaries."""
        for _ in range(num_objs):
            while True:
                rand_x = np.random.uniform(self.xmin, self.xmax)
                rand_y = np.random.uniform(self.ymin, self.ymax)

                if self.polygon_path.contains_point((rand_x, rand_y)):
                    break
            self.objects.append([obj_type, (rand_x, rand_y)])
        self._obj_xy = np.array([xy for _, xy in self.objects], dtype=np.float32)
            
    def update_certainties(self, rows, cols, detected: int, noise=None):
        return self.filter.step(rows, cols, detected, R=noise)
    
    def get_cur_cell(self, x, y):
        x, y = np.atleast_1d(x), np.atleast_1d(y)
        cols = ((x - self.xmin) // self.cell_size).astype(int)
        rows = ((y - self.ymin) // self.cell_size).astype(int)
        valid = (rows >= 0) & (rows < self.n_rows) & (cols >= 0) & (cols < self.n_cols)
        return rows, cols, valid

    def plot(self):
        """Showcase a plotted figure of grid, polygon, and placed objects w/certainty principle."""
        ### plot fishnet grid ###
        xe = self.xmin + np.arange(self.n_cols + 1) * self.cell_size
        ye = self.ymin + np.arange(self.n_rows + 1) * self.cell_size
        mask = self.inside_cells.ravel()
        plt.scatter(*(self.cells[:, mask])[::-1], linewidths=0.01)
        plt.pcolormesh(xe, ye, self.certainties, edgecolors='k', alpha=0.5)
        plt.fill(*self.polygon.T, alpha=0.3)

        ### scatter objects ###
        for obj_type, xy in self.objects:
            obj_color = 'green' if obj_type == 'tree' else 'red'
            plt.plot(*xy, marker='o', color=obj_color, linestyle='none')
        plt.show(block=True)

    @property
    def certainties(self):
        return self.filter.x
    
    @property
    def confidence(self):
        return 1 - self.filter.P  # 0: no evidence, 1 otherwise
    
    @property
    def obj_xy(self):
        return self._obj_xy

if __name__ == "__main__":
    grid = Grid(np.array([[0,0],[30,5],[25,20],[5,15]], dtype=np.float32))
    grid.plot()
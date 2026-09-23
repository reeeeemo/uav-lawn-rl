from matplotlib.collections import LineCollection
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

class DroneLogger:
    def __init__(self, sim_dt: float, save_path: str):
        self.data = {}
        self.sim_dt = sim_dt
        self.save_path = save_path

    def _compute_speeds(self, trajs: list):
        all_speeds = [
            np.linalg.norm(np.diff(t, axis=0), axis=1) / self.sim_dt 
            for t in trajs
        ]
        vmax = max((s.max() for s in all_speeds if len(s)), default=1.0)
        return all_speeds, Normalize(0, vmax)


    def load_data(self, npz: list[str], keys: list[str]):
        """Load all data files for plotting."""
        if len(npz) <= 0:
            return

        self.data = {k: [] for k in keys}
        for n in npz:
            n_data = np.load(n)
            for k in self.data:
                self.data[k].append(n_data[k])

    def _plot_xy(self, traj: list, speed: list, norm: Normalize):
        if not traj:
            return
        fig, ax = plt.subplots(figsize=(10,8))

        for i, (t, spd) in enumerate(zip(traj, speed)):
            pts = t[:, [0, 1]].reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap='viridis', norm=norm, array=spd, linewidth=2)
            ax.add_collection(lc)

            ax.scatter(
                t[0, 0], t[0, 1], c="green", s=60,
                marker='o', label="start" if i == 0 else None, zorder=60
            )
            ax.scatter(
                t[-1, 0], t[-1, 1], c="red", s=60,
                marker='o', label="end" if i == 0 else None, zorder=60
            )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        fig.colorbar(lc, ax=ax, label="speed (m/s)")
        ax.autoscale()

        plt.tight_layout()
        plt.savefig(f"{self.save_path}/xy_traj.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_xz(self, traj: list, speed: list, norm: Normalize):
        if not traj:
            return
        fig, ax = plt.subplots(figsize=(10,8))

        for i, (t, spd) in enumerate(zip(traj, speed)):
            pts = t[:, [0, 2]].reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap='viridis', norm=norm, array=spd, linewidth=2)
            ax.add_collection(lc)

            ax.scatter(
                t[0, 0], t[0, 2], c="green", s=60,
                marker='o', label="start" if i == 0 else None, zorder=60
            )
            ax.scatter(
                t[-1, 0], t[-1, 2], c="red", s=60,
                marker='o', label="end" if i == 0 else None, zorder=60
            )
        ax.set_xlabel("X north (m)")
        ax.set_ylabel("Z down (m)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        ax.invert_yaxis()

        fig.colorbar(lc, ax=ax, label="speed (m/s)")
        ax.autoscale()

        plt.tight_layout()
        plt.savefig(f"{self.save_path}/xz_traj.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def log_trajectories(self):
        traj = self.data.get("traj", None)
        if traj is None:
            return

        all_speeds, norm = self._compute_speeds(traj)

        self._plot_xy(traj, all_speeds, norm)
        self._plot_xz(traj, all_speeds, norm)
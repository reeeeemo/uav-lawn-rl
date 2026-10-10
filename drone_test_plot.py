from drone.drone import Drone
from drone.camera import Camera
from environment.grid import Grid

import numpy as np
import matplotlib.pyplot as plt

def yaw(psi):
    c, s = np.cos(psi), np.sin(psi)
    
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ], dtype=np.float32)

def main():
    action = np.array([0, 0.5, 0], dtype=np.float32)
    n_timesteps = 16
    grid = Grid(np.array([[0,0],[58,8],[52,42],[6,34]], dtype=np.float32))
    cam = Camera()
    
    grid.generate_objects("tree", 60)
    
    ranges, gains, visits = [], [], []
    seen_count = np.zeros((grid.n_rows, grid.n_cols), dtype=int)
    # per-step path:
    # do an action, check all visible objects
    # index them, measure their positions, get the current
    # cell associated, then update certainty
    #xyz = np.array([0, 0, -5], dtype=np.float32)
    xyz = np.array([29, 21, -5], dtype=np.float32)
    R = np.eye(3, dtype=np.float32)
    for k in range(n_timesteps):
        R = yaw(2 * np.pi * k / n_timesteps)
        
        # get all visible objects
        vis = cam.visible(xyz, R, grid.obj_xy)
        est, rng_m = cam.measure(xyz, R, grid.obj_xy[vis])
        rows, cols, valid = grid.get_cur_cell(*est.T)
        
        max_r = -(xyz[2] - cam.offset[2]) / np.tan(np.arcsin(cam.min_sin))
        R_meas = 1 + 3 * (rng_m / max_r) ** 2
        
        r, c = rows[valid], cols[valid]
        before = grid.certainties[r, c].copy()
        grid.update_certainties(r, c, detected=1, noise=R_meas[valid])
        after = grid.certainties[r, c]
        
        ranges.extend(rng_m[valid].tolist())
        gains.extend((after - before).tolist())
        visits.extend(seen_count[r, c].tolist())
        seen_count[r, c] += 1

    grid.plot()
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    s = ax.scatter(ranges, gains, c=visits, cmap="Blues",vmin=0, s=60, edgecolor="white", linewidth=0.8)
    ax.set_xlabel("range to object (m)")
    ax.set_ylabel("certainty gained this look")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    fig.colorbar(s, ax=ax, label="prior looks at cell")
    plt.show(block=True)

if __name__ == "__main__":
    main()
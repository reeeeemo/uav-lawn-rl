import numpy as np
import matplotlib.pyplot as plt


class CertaintyKalmanFilter:
    def __init__(self, shape: tuple, R: float = 0.5):
        self.shape = shape
        self.R = R
        self.reset()
    
    def step(self, rows, cols, z: float, R=None):
        R = self.R if R is None else R
        P = self.P[rows, cols]
        K = P / (P + R)
        self.x[rows, cols] += K * (z - self.x[rows, cols])
        self.P[rows, cols] = (1 - K) * P
        return self.x[rows, cols]
    
    def reset(self):
        self.P = np.ones(self.shape, dtype=np.float32)
        self.x = np.full(self.shape, 0.5, dtype=np.float32)
        
if __name__ == "__main__":
    f = CertaintyKalmanFilter((1,1))
    looks = 6
    n = np.arange(1, looks+1)
    w = 0.38
    
    def run(R, z):
        f = CertaintyKalmanFilter((1,1), R=R)
        conf, belief = [], []
        for _ in range(looks):
            f.step(0, 0, z=z)
            conf.append(1.0 - float(f.P[0, 0]))
            belief.append(float(f.x[0, 0]))
        return conf, belief
    
    near_conf, hit = run(R=1.2, z=1.0)
    far_conf, _ = run(R=4.0, z=1.0)
    _, miss = run(R=1.2, z=0.0)
    
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    a.bar(n - w/2, near_conf, w, label="near (5 m)")
    a.bar(n + w/2, far_conf, w, label="far (20 m)")
    a.set_title("Range of confidence in cell coverage (1-P)")    

    b.bar(n-w/2, hit, w, label="detected (z=1)")
    b.bar(n+w/2, miss, w, label="not detected (z=0)")
    b.axhline(0.5, color="0.5", lw=1, ls="--")
    b.set_title("Certainty of existence of detection x")
    
    for ax in (a, b):
        ax.set_xticks(n)
        ax.set_xlabel("look")
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
    a.set_ylabel("value")
    plt.tight_layout()
    plt.show(block=True)
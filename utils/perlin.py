import numpy as np
import matplotlib.pyplot as plt


class PerlinNoise:
    """Implementation of Ken Perlin's Improved Noise.
    
    Source: https://mrl.cs.nyu.edu/~perlin/noise/

    Args:
        seed: random seed to generate permutations of a gradient
    """
    def __init__(self, seed: int):
        self.G = np.array([
            [1, 1], [-1, 1], [1, -1],
            [-1, -1], [1, 0], [-1, 0],
            [0, 1], [0, -1]
        ])
        self.reset(seed=seed)

    def reset(self, seed: int):
        """Repermute gradients given an RNG seed."""
        p = np.random.default_rng(seed).permutation(256)
        self.p = np.concatenate([p, p])  # p[p[X] + Y] never overflows

    def noise(self, xy, persistence=0.5, octaves=4, lacunarity=2.0, amplitude=10.0, frequency=0.1):
        """Generate 2D height map of perlin noise.
        
        Args:
            persistence: height each layer keeps relative to one before.
            octaves: number of layers
            lacunarity: how much freqency per octave
            amplitude: overall height scale
            frequency: how wide each layer's area is
        """
        h, amp, total, freq = 0.0, 1.0, 0.0, frequency

        ### adds diff iterations of noise (octaves) to increase frequencies ###
        ### also called fractal borwnian motion! https://thebookofshaders.com/13/ ###
        for k in range(octaves):
            h += amp * self._noise(xy * freq + k * 17.3)
            total += amp
            amp *= persistence
            freq *= lacunarity
        return amplitude * h / total  # scale noise back down to [-1, 1]

    def _noise(self, xy):
        """Generate single layer of 2D Perlin noise."""
        ### generate unit cube and relative xyz of point in cube ###
        cell = np.floor(xy).astype(int)
        idx = cell & 255
        frac = xy - cell
        uv = self._fade(frac)  # fade curves

        ### get cube corners ###
        a = self.p[idx[:, 0]] + idx[:, 1]
        b = self.p[idx[:, 0] + 1] + idx[:, 1]
        aa, ab = self.p[a], self.p[a+1]
        ba, bb = self.p[b], self.p[b+1]

        ### gradient interpolated with offset from the cube's corner ###
        u, v = uv[:, 0], uv[:, 1]
        x1 = self._lerp(u, self._grad(aa, frac - [0, 0]), self._grad(ba, frac - [1, 0]))
        x2 = self._lerp(u, self._grad(ab, frac - [0, 1]), self._grad(bb, frac - [1, 1]))
        return self._lerp(v, x1, x2)

    def _lerp(self, t: np.ndarray, a: float, b: float) -> float:
        """Linearly interpolation floating-point value t."""
        return a + t * (b - a)

    def _fade(self, t: np.ndarray) -> float:
        """Smooth interpolation weights."""
        return t**3 * (t * (t * 6 - 15) + 10)

    def _grad(self, hash: int, offset: np.ndarray):
        """Compute 8 gradient directions from coordinates."""
        g = self.G[hash & 7]
        return (g * offset).sum(axis=1)  # gradient * offset between axes dists

if __name__ == "__main__":
    xs, ys = np.meshgrid(np.linspace(0, 30, 300), np.linspace(0, 20, 200))
    pts = np.column_stack([xs.ravel(), ys.ravel()])
    polygon = np.array([[0,0],[30,5],[25,20],[5,15]], dtype=np.float32)

    noiser = PerlinNoise(4)
    h = noiser.noise(pts).reshape(xs.shape)

    fig, ax = plt.subplots(figsize=(6, 4))
    mesh = ax.pcolormesh(xs, ys, h, cmap="terrain")
    ax.plot(*np.vstack([polygon, polygon[0]]).T, "k-")
    fig.colorbar(mesh, ax=ax, label="noise")

    plt.show(block=True)
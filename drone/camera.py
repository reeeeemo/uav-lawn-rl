import numpy as np

class Camera:
    """Class that mimics a camera.
    
    Args:
        offset: position from local body frame to the camera
        tx, ty: half_w / half_h of image plane at unit depth
        min_sin: sin of far-range visibility cutoff (rejects shallow views)
        px_sigma: std dev of detectior's pixel localization error
        M: camera rotation, rows are [right, down, forward]
    """
    def __init__(self, 
                 fx=87.0,
                 fy=58.0,
                 tilt_deg=45.0,
                 offset=(0.0, 0.0, 0.15),
                 min_grazing_deg=10.0,
                 px_sigma=1.0,
                 rows_px=480,
                 cols_px=848
                 ):
        self.offset = np.array(offset, dtype=np.float32)
        self.tx, self.ty = np.tan(np.radians(fx/2)), np.tan(np.radians(fy/2))
        self.min_sin = np.sin(np.radians(min_grazing_deg))
        self.px_sigma = px_sigma
        
        # camera intrinsics mat
        self.K = np.array([
            [(cols_px/2)/self.tx, 0, cols_px/2],
            [0, (rows_px/2)/self.ty, rows_px/2],
            [0, 0, 1]
        ])
        
        # convert cam mat to [right, down, forward]
        t = np.radians(tilt_deg)
        f = np.array([np.cos(t), 0.0, np.sin(t)])
        r = np.array([0.0, 1.0, 0.0])
        self.M = np.stack([r, np.cross(f, r), f])
        
    def visible(self, pos, R, pts):
        """Point to camera projection via pinhole model, then return if in FOV range."""
        p_cam = pos + R @ self.offset
        d = np.column_stack([pts, np.zeros(len(pts))]) - p_cam
        v = (self.M @ R.T @ d.T).T
        in_frustum = (
            (v[:, 2] > 0)
            & (np.abs(v[:, 0]) <= self.tx*v[:, 2])
            & (np.abs(v[:, 1]) <= self.ty*v[:, 2])
        )
        
        # angle between line of sight and ground, too close increases error dramatically.
        grazing = d[:, 2] / np.linalg.norm(d, axis=1)
        return in_frustum & (grazing >= self.min_sin)
    
    def measure(self, pos, R, pts):
        """Ground truth -> observed position. Range-dependent PnP error."""
        # assuming model plane z=0, in world frame (ground)
        # https://opi-lab.github.io/topics-computer-vision/pdfs/PAMI_2000_Zhang.pdf
        p_cam = pos + R @ self.offset
        R_cw = self.M @ R.T
        H = self.K @ np.column_stack([R_cw[:, 0], R_cw[:, 1], -R_cw @ p_cam])
        uv = (H @ np.column_stack([pts, np.ones(len(pts))]).T)
        uv = (uv[:2] / uv[2]).T + np.random.normal(0, self.px_sigma, (len(pts), 2))
        w = np.linalg.inv(H) @ np.column_stack([uv, np.ones(len(pts))]).T
        
        est = (w[:2] / w[2]).T
        rng_m = np.linalg.norm(est - p_cam[:2], axis=1)
        return est, rng_m
import torch
import numpy as np

class Drone():
    """Quadrotor physics rigidbody that uses a P rate controller to fly.

    Uses NED conversion for world frame, and FRD for body frame. Rates are tracked
    by a P-controller so first-order lag is inherent.

    NED (world): [+x = north, +y = east, +z = down]
    FRD (body): [+x = forward, +y = right, +z = down]
    TODO: mixer / control allocation (arm length of rotors (L), yaw torque per N of thrust (k), motor ceiling (f_max))
    TODO: motor dynamics (motor lag (T_m), air resistance (c_drag))
    TODO: add integral and derivative to p controller :P
    """
    def __init__(self):
        # drone internal variables
        self.mass = 2.0
        self.J = torch.diag(torch.tensor([0.0217, 0.0217, 0.04], dtype=torch.float32))  # moments of inertia
        self.J_inv = torch.inverse(self.J)
        self.gravity = torch.tensor([0, 0, 9.81], dtype=torch.float32)

        # sim / controller params
        self.dt = 0.02  # 50 Hz deltatime
        self.kp = 0.2
        self.f_max = 10  # max force on each motor allocated
        self.max_rads = 6  # to scale body rates on
        self.kappa = 0.015  # yaw torque per N of motor
        self.L = 0.25  # arm length
        self.thrusts = torch.zeros(4, dtype=torch.float32)

        # drone state vars
        self.xyz = torch.zeros(3, dtype=torch.float32)
        self.lin_vel = torch.zeros(3, dtype=torch.float32)  # vx, vy, vz
        self.ang_vel = torch.zeros(3, dtype=torch.float32)  # w_r, w_p, w_y
        self.R = torch.eye(3, dtype=torch.float32)  # rot_mat

        # https://cookierobotics.com/066/ explains well
        # t, mx, my, mz where:
        # m1 (+l, +l) = CCW || m2 (-l, -l) = CCW || m3 (+l, -l) = CW || m4 (-l, +l) = CW
        a = self.L / 2 ** 0.5
        self.M = torch.tensor([
            [1, 1, 1, 1],
            [-a, a, a, -a],
            [a, -a, a, -a],
            [self.kappa, self.kappa, -self.kappa, -self.kappa]
        ], dtype=torch.float32)  # motor mixing table. Uses PX4
        self.M_inv = torch.inverse(self.M)

    def __str__(self):
        return f"v: {self.lin_vel}\nw: {self.ang_vel}\nxyz: {self.xyz}\nrpy: {self.rpy_euler}\nthrusts: {self.thrusts}"

    def step(self, action: torch.tensor):
        """Step once through drone dynamics model.

        [thrust in [0, 1], rates in [-1, 1]].
        Args:
            action: tensor of [collective_thrust, roll, pitch, yaw] body rates
        Returns:
            wawa
        """
        T = self._normalize_thrust(action[0])

        # translate dynamics to 4 motor thrusts with a mixer capped at f_max
        tau = self.kp * (action[1:] * self.max_rads - self.ang_vel)  # kp * (setpoint - measured). torques from p loop
        cmd = torch.cat([T.reshape(1), tau])
        self.thrusts = torch.clamp(self.M_inv @ cmd, 0, self.f_max)
        cmd = self.M @ self.thrusts
        T, tau = cmd[0], cmd[1:]

        # 2: J^-1 (tau - w x Jw)
        ang_accel = self.J_inv @ (tau - torch.linalg.cross(self.ang_vel, self.J @ self.ang_vel))
        # v_dot = g+1/m * RF^B (force in body frame)
        F_body = torch.tensor([0, 0, -T])
        lin_accel = self.R @ F_body / self.mass + self.gravity

        # dynamics integration equations
        self.lin_vel += lin_accel * self.dt
        self.ang_vel += ang_accel * self.dt
        self.xyz += self.lin_vel * self.dt
        # R(t)[w(t)]_x where [w]_x is the skew symmetric matrix representation of ang vel
        self.R = self.R @ torch.linalg.matrix_exp(self._skew(self.ang_vel * self.dt))

        # gram-schmidt to mitigate floating point errors
        self.R = self._orthonormalize(self.R.clone())

    def reset(self):
        self.R = torch.eye(3, dtype=torch.float32)
        self.xyz = torch.zeros_like(self.xyz)
        self.lin_vel = torch.zeros_like(self.lin_vel)
        self.ang_vel = torch.zeros_like(self.ang_vel)
        self.thrusts = torch.zeros_like(self.thrusts)

    def _orthonormalize(self, R):
        """Gram-Schmidt on a rotation matrix to ensure axes are orthogonal."""
        # https://www.geeksforgeeks.org/machine-learning/gram-schmidt-process-for-ml/
        x = R[:, 0] / torch.linalg.norm(R[:, 0])
        y = R[:, 1]
        y -= torch.dot(x, y) * x
        y /= torch.linalg.norm(y)
        z = torch.linalg.cross(x, y)
        return torch.stack([x, y, z], dim=1)

    def _skew(self, v):
        """Returns a skew-symmetric matrix given (x,y,z) vals."""
        # https://www.geeksforgeeks.org/maths/what-is-symmetric-matrix-and-skew-symmetric-matrix/
        x, y, z = v
        return torch.tensor(
            [[0, -z, y],
            [z, 0, -x],
            [-y, x, 0]],
            dtype=torch.float32
        )

    def _normalize_thrust(self, t: float):
        """Normalize thrust of 4 motors"""
        return t * (4 * self.f_max)

    @property
    def rpy_euler(self):
        r = torch.atan2(self.R[2, 1], self.R[2, 2])
        p = torch.atan2(-self.R[2, 0], torch.sqrt(self.R[2,1]**2 + self.R[2,2]**2))
        y = torch.atan2(self.R[1,0], self.R[0,0])
        return r, p, y

if __name__ == "__main__":
    drone = Drone()

    # see init drone
    drone.step(torch.zeros(4, dtype=torch.float32))
    print(drone)

    # pitch forward and see how drone moves
    #drone.step(torch.tensor([0.1, 0, 0.1, 0], dtype=torch.float32))
    for i in range(10):
        print(f"\n-----------\nstep: {i+1}\n-----------\n")
        drone.step(torch.tensor([1.0, 1, 0, 0], dtype=torch.float32))
        print(drone)
import numpy as np


class Flame3D:
    """
    Reduced-order 3D reaction-diffusion flame model.

    This is NOT full CFD.
    It is a computational experiment for studying how
    gravity changes a simplified reacting thermal field.
    """

    def __init__(
        self,
        N=64,
        L=0.04,
        dt=1e-5,
        gravity=9.81,
    ):
        self.N = N
        self.L = L
        self.dx = L / (N - 1)
        self.dt = dt
        self.gravity = gravity

        # Physical constants / reduced-order parameters
        self.alpha = 2.0e-5       # thermal diffusivity
        self.Df = 1.0e-5          # fuel diffusivity
        self.Do = 1.0e-5          # oxygen diffusivity

        self.rho = 1.0
        self.cp = 1000.0
        self.Q = 2.0e6

        self.A = 5.0e4
        self.Ea = 8.0e3
        self.R = 8.314

        # Ambient/reference temperature
        self.T0 = 300.0

        # Fields
        shape = (N, N, N)

        self.T = np.full(shape, self.T0, dtype=np.float32)
        self.Yf = np.zeros(shape, dtype=np.float32)
        self.Yo = np.zeros(shape, dtype=np.float32)

        # Coordinate system
        x = np.linspace(-L / 2, L / 2, N)
        y = np.linspace(-L / 2, L / 2, N)
        z = np.linspace(0, L, N)

        self.x, self.y, self.z = np.meshgrid(
            x, y, z, indexing="ij"
        )

        self.initialize_flame()

    def initialize_flame(self):
        """
        Create a small hot fuel-rich region near the bottom.
        """

        x0 = 0.0
        y0 = 0.0
        z0 = self.L * 0.18

        sigma_xy = self.L * 0.08
        sigma_z = self.L * 0.10

        r2 = (
            ((self.x - x0) ** 2 + (self.y - y0) ** 2)
            / sigma_xy**2
            + ((self.z - z0) ** 2)
            / sigma_z**2
        )

        flame_seed = np.exp(-r2).astype(np.float32)

        # Initial hot region
        self.T += 1200.0 * flame_seed

        # Fuel concentrated around the source
        self.Yf = (0.8 * flame_seed).astype(np.float32)

        # Oxygen initially surrounding the flame
        self.Yo = np.ones_like(self.T, dtype=np.float32)

    def laplacian(self, field):
        """
        3D finite-difference Laplacian:

        ∇²f =
            f_xx + f_yy + f_zz
        """

        out = np.zeros_like(field)

        dx2 = self.dx * self.dx

        out[1:-1, 1:-1, 1:-1] = (
            field[2:, 1:-1, 1:-1]
            + field[:-2, 1:-1, 1:-1]
            + field[1:-1, 2:, 1:-1]
            + field[1:-1, :-2, 1:-1]
            + field[1:-1, 1:-1, 2:]
            + field[1:-1, 1:-1, :-2]
            - 6.0 * field[1:-1, 1:-1, 1:-1]
        ) / dx2

        return out

    def reaction_rate(self):
        """
        Arrhenius reaction:

        ω = A Yf Yo exp(-Ea / RT)
        """

        temperature = np.maximum(self.T, 300.0)

        exponent = -self.Ea / (self.R * temperature)

        rate = (
            self.A
            * self.Yf
            * self.Yo
            * np.exp(exponent)
        )

        return rate.astype(np.float32)

    def buoyancy(self):
        """
        Very simplified vertical buoyancy velocity.

        Positive temperature differences produce
        upward motion when gravity is present.
        """

        beta = 2.0e-3

        velocity = (
            beta
            * self.gravity
            * (self.T - self.T0)
        )

        # Limit velocity for numerical stability
        velocity = np.clip(
            velocity,
            -0.5,
            0.5
        )

        return velocity.astype(np.float32)

    def advection_z(self, field, velocity):
        """
        First-order upwind vertical advection.
        """

        out = np.zeros_like(field)

        dz = self.dx

        positive = velocity >= 0
        negative = ~positive

        interior = slice(1, -1)

        # Upward velocity
        v = velocity[1:-1, 1:-1, 1:-1]
        f = field

        backward = (
            f[1:-1, 1:-1, 1:-1]
            - f[1:-1, 1:-1, :-2]
        ) / dz

        forward = (
            f[1:-1, 1:-1, 2:]
            - f[1:-1, 1:-1, 1:-1]
        ) / dz

        mask = v >= 0

        derivative = np.where(
            mask,
            backward,
            forward
        )

        out[1:-1, 1:-1, 1:-1] = v * derivative

        return out

    def step(self):
        """
        Advance the system by one timestep.
        """

        omega = self.reaction_rate()

        # Diffusion
        lap_T = self.laplacian(self.T)
        lap_Yf = self.laplacian(self.Yf)
        lap_Yo = self.laplacian(self.Yo)

        # Simplified buoyancy
        velocity = self.buoyancy()

        adv_T = self.advection_z(self.T, velocity)
        adv_Yf = self.advection_z(self.Yf, velocity)
        adv_Yo = self.advection_z(self.Yo, velocity)

        # Temperature equation
        dT = (
            self.alpha * lap_T
            - adv_T
            + (self.Q / (self.rho * self.cp)) * omega
        )

        # Fuel equation
        dYf = (
            self.Df * lap_Yf
            - adv_Yf
            - omega
        )

        # Oxygen equation
        dYo = (
            self.Do * lap_Yo
            - adv_Yo
            - omega
        )

        # Update
        self.T += self.dt * dT
        self.Yf += self.dt * dYf
        self.Yo += self.dt * dYo

        # Physical bounds
        self.Yf = np.clip(self.Yf, 0.0, 1.0)
        self.Yo = np.clip(self.Yo, 0.0, 1.0)

        self.T = np.maximum(self.T, self.T0)

    def run(self, steps):
        for _ in range(steps):
            self.step()

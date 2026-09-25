import numpy as np
import plotly.graph_objects as go

from flame3d import Flame3D


# --------------------------------------------------
# Simulation
# --------------------------------------------------

N = 64
STEPS = 500

sim = Flame3D(
    N=N,
    gravity=9.81
)

print("Running simulation...")

for i in range(STEPS):
    sim.step()

    if i % 100 == 0:
        print(
            f"step={i:4d}  "
            f"Tmax={sim.T.max():.1f} K"
        )


# --------------------------------------------------
# Extract data
# --------------------------------------------------

T = sim.T.astype(np.float32)
omega = sim.reaction_rate()

x = sim.x[:, 0, 0]
y = sim.y[0, :, 0]
z = sim.z[0, 0, :]


# --------------------------------------------------
# Downsample for visualization
# --------------------------------------------------

skip = 2

X = sim.x[::skip, ::skip, ::skip].ravel()
Y = sim.y[::skip, ::skip, ::skip].ravel()
Z = sim.z[::skip, ::skip, ::skip].ravel()

Tv = T[::skip, ::skip, ::skip].ravel()
Ov = omega[::skip, ::skip, ::skip].ravel()


# --------------------------------------------------
# Temperature isosurface
# --------------------------------------------------

fig = go.Figure()

fig.add_trace(
    go.Isosurface(
        x=X,
        y=Y,
        z=Z,
        value=Tv,

        isomin=500,
        isomax=float(Tv.max()),

        surface_count=4,

        opacity=0.35,

        caps=dict(
            x_show=False,
            y_show=False,
            z_show=False
        ),

        colorbar=dict(
            title="Temperature (K)"
        )
    )
)


# --------------------------------------------------
# Reaction-zone isosurface
# --------------------------------------------------

if Ov.max() > 0:

    fig.add_trace(
        go.Isosurface(
            x=X,
            y=Y,
            z=Z,
            value=Ov,

            isomin=0.2 * float(Ov.max()),
            isomax=float(Ov.max()),

            surface_count=2,

            opacity=0.5,

            caps=dict(
                x_show=False,
                y_show=False,
                z_show=False
            ),

            showscale=False
        )
    )


# --------------------------------------------------
# Layout
# --------------------------------------------------

fig.update_layout(
    title="3D Flame — Reduced-Order Combustion Model",

    scene=dict(
        xaxis_title="x (m)",
        yaxis_title="y (m)",
        zaxis_title="z (m)",

        aspectmode="cube"
    ),

    width=1000,
    height=800
)

fig.show()

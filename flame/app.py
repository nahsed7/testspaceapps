import numpy as np
import streamlit as st
import plotly.graph_objects as go

from flame3d import Flame3D


st.set_page_config(
    page_title="Flame in Freefall",
    layout="wide"
)

st.title("🔥 Flame in Freefall — 3D Combustion Experiment")

st.write(
    "The controls directly define the local thermochemical conditions "
    "and gravity of the 3D simulation."
)


# ============================================================
# CONTROLS
# ============================================================

st.sidebar.header("🔥 Flame Conditions")

temperature = st.sidebar.slider(
    "Temperature at point (K)",
    300, 2500, 1200, 10
)

oxygen = st.sidebar.slider(
    "Oxygen concentration (%)",
    0, 100, 21, 1
)

fuel = st.sidebar.slider(
    "Fuel concentration (%)",
    0, 100, 80, 1
)

gravity_g = st.sidebar.slider(
    "Gravity (g)",
    0.0, 1.0, 1.0, 0.01
)

gravity = gravity_g * 9.81


# ============================================================
# CONSTANTS
# ============================================================

A = 5.0e4
Ea = 8.0e3
R = 8.314
Q = 2.0e6

T0 = 300.0

Yf = fuel / 100.0
Yo = oxygen / 100.0


# ============================================================
# ARRHENIUS CHEMISTRY
# ============================================================

arrhenius = np.exp(
    -Ea / (R * max(temperature, 300.0))
)

local_reaction_rate = (
    A
    * Yf
    * Yo
    * arrhenius
)

local_heat_release = (
    Q * local_reaction_rate
)


# ============================================================
# CREATE A FRESH SIMULATION EVERY TIME
# ============================================================

sim = Flame3D(
    N=64,
    gravity=gravity
)


# ============================================================
# REBUILD INITIAL 3D FLAME FROM THE SLIDERS
# ============================================================

N = sim.N

cx = N // 2
cy = N // 2
cz = int(N * 0.20)

xx, yy, zz = np.indices((N, N, N))

distance2 = (
    (xx - cx) ** 2
    + (yy - cy) ** 2
    + ((zz - cz) * 0.8) ** 2
)

sigma = 7.0

flame_shape = np.exp(
    -distance2 / (2.0 * sigma**2)
).astype(np.float32)


# ------------------------------------------------------------
# Temperature field
# ------------------------------------------------------------

# Temperature at the selected point determines the
# temperature peak of the initial flame.

sim.T[:] = (
    T0
    + (temperature - T0) * flame_shape
).astype(np.float32)


# ------------------------------------------------------------
# Fuel field
# ------------------------------------------------------------

sim.Yf[:] = (
    Yf * flame_shape
).astype(np.float32)


# ------------------------------------------------------------
# Oxygen field
# ------------------------------------------------------------

# Oxygen exists around the flame.
# The selected concentration controls the local oxygen field.

sim.Yo[:] = np.clip(
    Yo * (0.35 + 0.65 * (1.0 - flame_shape)),
    0.0,
    1.0
).astype(np.float32)


# ============================================================
# RUN SIMULATION
# ============================================================

steps = 25

for _ in range(steps):
    sim.step()


# ============================================================
# MEASURE RESULTING FIELD
# ============================================================

Tfield = sim.T

reaction_field = sim.reaction_rate()

Tmax = float(Tfield.max())
reaction_total = float(reaction_field.sum())


# ============================================================
# NUMERICAL READOUT
# ============================================================

st.subheader("Current conditions")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Temperature",
    f"{temperature} K"
)

c2.metric(
    "Oxygen",
    f"{oxygen}%"
)

c3.metric(
    "Fuel",
    f"{fuel}%"
)

c4.metric(
    "Gravity",
    f"{gravity_g:.2f} g"
)


c5, c6, c7 = st.columns(3)

c5.metric(
    "Resulting Tmax",
    f"{Tmax:.1f} K"
)

c6.metric(
    "Local reaction rate",
    f"{local_reaction_rate:.3e}"
)

c7.metric(
    "Total reaction",
    f"{reaction_total:.3e}"
)


# ============================================================
# 3D DATA
# ============================================================

skip = 2

X = sim.x[::skip, ::skip, ::skip].ravel()
Y = sim.y[::skip, ::skip, ::skip].ravel()
Z = sim.z[::skip, ::skip, ::skip].ravel()

Tv = Tfield[::skip, ::skip, ::skip].ravel()

Rv = reaction_field[::skip, ::skip, ::skip].ravel()


# ============================================================
# 3D FLAME
# ============================================================

st.subheader("3D Flame")

fig = go.Figure()


# ------------------------------------------------------------
# Temperature field
# ------------------------------------------------------------

T_low = max(350.0, float(Tv.min()))
T_high = max(T_low + 1.0, float(Tv.max()))

fig.add_trace(
    go.Isosurface(
        x=X,
        y=Y,
        z=Z,
        value=Tv,

        isomin=T_low,
        isomax=T_high,

        surface_count=6,

        opacity=0.50,

        colorscale="Inferno",

        caps=dict(
            x_show=False,
            y_show=False,
            z_show=False
        ),

        colorbar=dict(
            title="Temperature (K)"
        ),

        name="Temperature"
    )
)


# ------------------------------------------------------------
# Reaction zone
# ------------------------------------------------------------

if Rv.max() > 0:

    fig.add_trace(
        go.Isosurface(
            x=X,
            y=Y,
            z=Z,
            value=Rv,

            isomin=0.20 * float(Rv.max()),
            isomax=float(Rv.max()),

            surface_count=3,

            opacity=0.35,

            colorscale="Hot",

            showscale=False,

            caps=dict(
                x_show=False,
                y_show=False,
                z_show=False
            ),

            name="Reaction zone"
        )
    )


fig.update_layout(
    height=750,

    scene=dict(
        xaxis_title="x (m)",
        yaxis_title="y (m)",
        zaxis_title="z (m)",

        aspectmode="cube"
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# PHYSICS
# ============================================================

st.subheader("Reaction model")

st.latex(
    r"""
    \dot{\omega}
    =
    A\,Y_FY_O
    \exp\left(-\frac{E_a}{RT}\right)
    """
)

st.write(
    "Temperature enters exponentially through the Arrhenius term. "
    "Fuel and oxygen enter multiplicatively. Gravity changes the "
    "simplified buoyancy-driven transport."
)


# ============================================================
# EXPERIMENT INTERPRETATION
# ============================================================

if oxygen == 0 or fuel == 0:

    st.warning(
        "No fuel or no oxygen: the reaction source is zero."
    )

elif temperature < 450:

    st.info(
        "The selected temperature is relatively low for this "
        "reduced-order reaction model."
    )

elif gravity_g < 0.05:

    st.info(
        "Near-freefall condition: buoyancy is strongly suppressed."
    )

else:

    st.success(
        "Combustion conditions are active."
    )

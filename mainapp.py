"""
Streamlit app — Trust & Innovation dynamics visualisation.

Reads calibrated values from calibrate.py.
    - Chart 1a / 1b : historical T(t)  vs  equation-based T(t)
    - Chart 2a / 2b : historical I(t)  vs  equation-based I(t)

Run with:
    streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import calibrate


# -----------------------------------------------------------------------------
#  Get the calibrated values from the file 'calibrate.py'
# -----------------------------------------------------------------------------
@st.cache_data
def get_calibrated_trust():
    return calibrate.calibrate_trust()

@st.cache_data
def get_calibrated_innovation():
    return calibrate.calibrate_innovation()

@st.cache_data
def get_calibrated_attractiveness():
    return calibrate.calibrate_attractiveness()


@st.cache_data
def get_calibrated_fdi():
    return calibrate.calibrate_fdi()

ALPHA_T, DELTA_T, R2_T, RMSE_T            = get_calibrated_trust()
ALPHA_I, DELTA_I, K_I_CAL, R2_I, RMSE_I   = get_calibrated_innovation()
A0_CAL, W_T_CAL, W_I_CAL, W_P_CAL, W_C_CAL  = get_calibrated_attractiveness()
BETA_CAL, GAMMA_CAL, K_F_CAL, R2_F, RMSE_F  = get_calibrated_fdi()

# -----------------------------------------------------------------------------
#  Page setup
# -----------------------------------------------------------------------------
st.title("Trust, Innovation and FDI - Historical vs Equation-based")
st.caption("Master's thesis, System Dynamics")


# -----------------------------------------------------------------------------
#  Load historical data
# -----------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent / "data" / "historical_data.csv"
df_full = pd.read_csv(DATA_PATH)

# Force numeric columns to be numbers (strips commas, handles empty cells)
for col in ["T", "R", "I", "C", "FDI", "E", "P", "A"]:
    if col in df_full.columns:
        df_full[col] = pd.to_numeric(
            df_full[col].astype(str).str.replace(",", ""),
            errors="coerce"
        )

df_T = df_full.dropna(subset=["T", "R"]).reset_index(drop=True)
df_I = df_full.dropna(subset=["I", "C", "FDI"]).reset_index(drop=True)
df_E = df_full.dropna(subset=["E"]).reset_index(drop=True)

years_hist = df_T["Year"].to_numpy()
T_hist     = df_T["T"].to_numpy()
R_hist     = df_T["R"].to_numpy()

years_I    = df_I["Year"].to_numpy()
I_hist     = df_I["I"].to_numpy()
C_hist     = df_I["C"].to_numpy()
F_hist     = df_I["FDI"].to_numpy()

years_E    = df_E["Year"].to_numpy()
E_hist     = df_E["E"].to_numpy()

# =============================================================================
#  SECTION 0 — GDPR Enforcement Intensity E(t)
# =============================================================================
st.subheader("GDPR Enforcement Intensity E(t) - the central policy variable")
st.markdown(
    "This is the variable the thesis is actually about: how vigorously the "
    "Irish DPC enforces GDPR each year. E(t) is constructed from four "
    "indicators - formal decisions issued, statutory inquiries, "
    "cross-border inquiries, and fines - each normalised to 0-1 and averaged.  \n\n"
    "**E(t) enters the model through two channels:**  \n\n"
    "**Channel 1 (positive - through Trust).** E(t) is one of the three "
    "components of Regulatory Credibility R(t). Higher enforcement raises "
    "regulatory credibility, which raises trust through the Trust equation, "
    "which raises Attractiveness, which attracts FDI.  \n\n"
    "**Channel 2 (negative - through Compliance Cost).** The construction "
    "C(t) = ½·E(t) + ½·(normalised DPO cost) means compliance cost responds "
    "directly to enforcement intensity. Higher C suppresses Innovation "
    "(through the -δᵢ·C term) and accelerates FDI outflow (through the -γ·C·F term).  \n\n"
)
 
# Historical E chart
figE = go.Figure()
figE.add_trace(go.Scatter(x=years_E, y=E_hist,
                          mode="lines+markers",
                          name="E(t) historical",
                          line=dict(color="#d62728", width=2.5),
                          marker=dict(size=10)))
figE.update_layout(
    title="Historical GDPR Enforcement Intensity in Ireland (2018-2024)",
    xaxis_title="Year",
    yaxis_title="E(t), normalised 0-1",
    yaxis=dict(range=[0, 1.0]),
    height=380,
    margin=dict(l=40, r=20, t=50, b=40),
)
st.plotly_chart(figE, use_container_width=True)
st.markdown("---")
 

# =============================================================================
#  SECTION 1 — TRUST
# =============================================================================
st.subheader("The Trust equation")
st.latex(r"\frac{dT}{dt} \;=\; \alpha_T \cdot R(t) \cdot (1 - T) \;-\; \delta_T \cdot T")

st.subheader("Calibrated values (loaded from the file 'calibrate.py')")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("αₜ", f"{ALPHA_T:.4f}")
col_b.metric("δₜ", f"{DELTA_T:.4f}")
col_c.metric("R²",   f"{R2_T:.3f}")
col_d.metric("RMSE", f"{RMSE_T:.3f}")

# --- Sidebar sliders for Trust ---
st.sidebar.header("Trust - equation parameters")
st.sidebar.markdown(
    "These sliders control **Chart 2** only. "
    "Defaults are the calibrated values from the file `calibrate.py`."
)
alpha_T = st.sidebar.slider("αₜ - trust building", 0.0, 3.0,
                            float(ALPHA_T), 0.01)
delta_T = st.sidebar.slider("δₜ - trust decay", 0.0, 1.5,
                            float(DELTA_T), 0.01)
T0      = st.sidebar.slider("T(0) - initial trust", 0.0, 1.0,
                            float(T_hist[0]), 0.01)

st.sidebar.header("R(t) input")
r_mode = st.sidebar.radio("How to feed R(t) into the equation?",
                          ["Use historical R(t) (2018-2024)",
                           "Use a constant R value"])
if r_mode == "Use a constant R value":
    R_const = st.sidebar.slider("Constant R", 0.0, 1.0, 0.53, 0.01)


# --- Solve the trust differential equation (simple Euler method) ---
def R_at(year):
    """Return R(t) for a given calendar year, depending on the chosen mode."""
    if r_mode == "Use a constant R value":
        return R_const
    return float(np.interp(year, years_hist, R_hist))


year_start = float(years_hist[0])
year_end   = float(years_hist[-1])
dt = 0.05
n_steps = int((year_end - year_start) / dt) + 1
t_grid = np.linspace(year_start, year_end, n_steps)

T_sim = np.zeros(n_steps)
T_sim[0] = T0
for i in range(n_steps - 1):
    R_curr = R_at(t_grid[i])
    slope = alpha_T * R_curr * (1 - T_sim[i]) - delta_T * T_sim[i]
    T_sim[i + 1] = T_sim[i] + dt * slope


# --- Trust charts ---
left, right = st.columns(2)
with left:
    st.subheader("Chart 1 - Historical Trust")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=years_hist, y=T_hist,
                              mode="lines+markers",
                              name="T historical",
                              line=dict(color="#1f77b4", width=2.5),
                              marker=dict(size=10)))
    fig1.update_layout(xaxis_title="Year", yaxis_title="Trust T(t)",
                       yaxis=dict(range=[-0.05, 1.1]),
                       height=420,
                       margin=dict(l=40, r=20, t=30, b=40))
    st.plotly_chart(fig1, use_container_width=True)

with right:
    st.subheader("Chart 2 - Equation-based Trust")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=t_grid, y=T_sim,
                              mode="lines",
                              name="T from equation",
                              line=dict(color="#d62728", width=2.5)))
    fig2.update_layout(xaxis_title="Year", yaxis_title="Trust T(t)",
                       yaxis=dict(range=[-0.05, 1.1]),
                       height=420,
                       margin=dict(l=40, r=20, t=30, b=40))
    st.plotly_chart(fig2, use_container_width=True)


# --- Trust end-of-period numbers ---
st.markdown("---")
m1, m2, m3 = st.columns(3)
m1.metric("Historical T(2024)", f"{T_hist[-1]:.3f}")
m2.metric("Equation T(2024)",   f"{T_sim[-1]:.3f}")
m3.metric("Difference",         f"{T_sim[-1] - T_hist[-1]:+.3f}")


# =============================================================================
#  SECTION 2 - INNOVATION
# =============================================================================
st.markdown("---")
st.subheader("The Innovation equation")
st.latex(r"\frac{dI}{dt} \;=\; \alpha_I \cdot \frac{F}{K_I + F} \cdot (1 - I) \;-\; \delta_I \cdot C(t)")

st.subheader("Calibrated values (loaded from the file 'calibrate.py')")
cI1, cI2, cI3, cI4, cI5 = st.columns(5)
cI1.metric("αᵢ", f"{ALPHA_I:.4f}")
cI2.metric("δᵢ", f"{DELTA_I:.4f}")
cI3.metric("Kᵢ (€M)", f"{K_I_CAL:,.0f}")
cI4.metric("R²",   f"{R2_I:.3f}")
cI5.metric("RMSE", f"{RMSE_I:.4f}")

st.info(
    f"**Calibration note.** K_I is fixed at the historical mean of Ireland's "
    f"ICT FDI stock ({K_I_CAL:,.0f} €M). With K_I fixed, α_I and δ_I are "
    f"fitted by OLS as Trust. "
    f"R² remains low ({R2_I:.3f}) because Ireland's innovation index varied "
    f"very little over 2018-2024 (range 0.598-0.634)."
)

# --- Sidebar sliders for Innovation ---
st.sidebar.header("Innovation - equation parameters")
alpha_I = st.sidebar.slider("αᵢ - innovation gain", 0.0, 1.0,
                            float(ALPHA_I), 0.001, format="%.4f")
delta_I = st.sidebar.slider("δᵢ - compliance drag", 0.0, 0.5,
                            float(DELTA_I), 0.001, format="%.4f")
K_I     = st.sidebar.slider("Kᵢ - half-saturation FDI (€M)",
                            10_000.0, 500_000.0, float(K_I_CAL), 5_000.0)
I0      = st.sidebar.slider("I(0) - initial innovation", 0.0, 1.0,
                            float(I_hist[0]), 0.01)

st.sidebar.header("F(t) and C(t) input")
fc_mode = st.sidebar.radio("How to feed F(t) and C(t)?",
                           ["Use historical (2018-2024)",
                            "Use constant values"])
if fc_mode == "Use constant values":
    F_const = st.sidebar.slider("Constant F (€M)",
                                10_000.0, 500_000.0, float(np.mean(F_hist)), 5_000.0)
    C_const = st.sidebar.slider("Constant C", 0.0, 1.0, float(np.mean(C_hist)), 0.01)


# --- Solve the innovation differential equation (simple Euler method) ---
def F_at(year):
    if fc_mode == "Use constant values":
        return F_const
    return float(np.interp(year, years_I, F_hist))

def C_at(year):
    if fc_mode == "Use constant values":
        return C_const
    return float(np.interp(year, years_I, C_hist))


year_start_I = float(years_I[0])
year_end_I   = float(years_I[-1])
n_steps_I = int((year_end_I - year_start_I) / dt) + 1
t_grid_I = np.linspace(year_start_I, year_end_I, n_steps_I)

I_sim = np.zeros(n_steps_I)
I_sim[0] = I0
for i in range(n_steps_I - 1):
    F_curr = F_at(t_grid_I[i])
    C_curr = C_at(t_grid_I[i])
    slope = (alpha_I * F_curr / (K_I + F_curr) * (1 - I_sim[i])
             - delta_I * C_curr)
    I_sim[i + 1] = I_sim[i] + dt * slope


# --- Innovation charts ---
left_I, right_I = st.columns(2)
with left_I:
    st.subheader("Chart 3 - Historical Innovation")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=years_I, y=I_hist,
                              mode="lines+markers",
                              name="I historical",
                              line=dict(color="#2ca02c", width=2.5),
                              marker=dict(size=10)))
    fig3.update_layout(xaxis_title="Year", yaxis_title="Innovation I(t)",
                       yaxis=dict(range=[0.5, 0.7]),
                       height=420,
                       margin=dict(l=40, r=20, t=30, b=40))
    st.plotly_chart(fig3, use_container_width=True)

with right_I:
    st.subheader("Chart 4 - Equation-based Innovation")
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=t_grid_I, y=I_sim,
                              mode="lines",
                              name="I from equation",
                              line=dict(color="#ff7f0e", width=2.5)))
    fig4.update_layout(xaxis_title="Year", yaxis_title="Innovation I(t)",
                       yaxis=dict(range=[0.5, 0.7]),
                       height=420,
                       margin=dict(l=40, r=20, t=30, b=40))
    st.plotly_chart(fig4, use_container_width=True)


# --- Innovation end-of-period numbers ---
st.markdown("---")
mI1, mI2, mI3 = st.columns(3)
mI1.metric("Historical I(2024)", f"{I_hist[-1]:.3f}")
mI2.metric("Equation I(2024)",   f"{I_sim[-1]:.3f}")
mI3.metric("Difference",         f"{I_sim[-1] - I_hist[-1]:+.3f}")



# =============================================================================
#  SECTION 3 — ATTRACTIVENESS
# =============================================================================
st.markdown("---")
st.subheader("The Investment Attractiveness Index")
st.latex(r"A(t) \;=\; A_0 \;+\; w_T \cdot T \;+\; w_I \cdot I \;+\; w_P \cdot P \;-\; w_C \cdot C")
 
st.subheader("Calibrated values (loaded from the file 'calibrate.py')")
cA1, cA2, cA3, cA4, cA5 = st.columns(5)
cA1.metric("A₀ (from 2017)", f"{A0_CAL:.4f}")
cA2.metric("w_T", f"{W_T_CAL:.2f}")
cA3.metric("w_I", f"{W_I_CAL:.2f}")
cA4.metric("w_P", f"{W_P_CAL:.2f}")
cA5.metric("w_C", f"{W_C_CAL:.2f}")
 
st.info(
    f"**Calibration note.** A₀ = {A0_CAL:.4f} is computed from the 2017 row "
    f"of the dataset (pre-GDPR, so T = 0 and C = 0). Equal weights "
    f"(0.25 each) are assumed for this research."
)
 
# --- Sidebar sliders for Attractiveness ---
st.sidebar.header("Attractiveness - equation parameters")
A_0 = st.sidebar.slider("A₀ - baseline attractiveness",
                        0.0, 1.0, float(A0_CAL), 0.001, format="%.4f")
w_T = st.sidebar.slider("w_T - weight on Trust",       0.0, 1.0, float(W_T_CAL), 0.01)
w_I = st.sidebar.slider("w_I - weight on Innovation",  0.0, 1.0, float(W_I_CAL), 0.01)
w_P = st.sidebar.slider("w_P - weight on Infra/Talent",0.0, 1.0, float(W_P_CAL), 0.01)
w_C = st.sidebar.slider("w_C - weight on Compliance Cost (subtracted)",
                        0.0, 1.0, float(W_C_CAL), 0.01)
 
# --- Compute A(t) two ways: from SIMULATED T,I and from HISTORICAL T,I ---
df_A = df_full.dropna(subset=["T", "I", "C", "P", "A"]).reset_index(drop=True)
years_A     = df_A["Year"].to_numpy()
T_A_hist    = df_A["T"].to_numpy()
I_A_hist    = df_A["I"].to_numpy()
P_A_hist    = df_A["P"].to_numpy()
C_A_hist    = df_A["C"].to_numpy()
A_hist      = df_A["A"].to_numpy()           # EY-based empirical A
 
# Version 1: A from HISTORICAL T, I, P, C (uncoupled — pure formula check)
A_from_hist_TI = A_0 + w_T*T_A_hist + w_I*I_A_hist + w_P*P_A_hist - w_C*C_A_hist
A_from_hist_TI = np.clip(A_from_hist_TI, 0.0, 1.0)
 
# Version 2: A from SIMULATED T, I (using the values from the Trust and
# Innovation sections above). T_sim was solved on t_grid; I_sim on t_grid_I.
# Evaluate them at each historical year of the Attractiveness dataset.
T_sim_at_yearsA = np.interp(years_A, t_grid,   T_sim)
I_sim_at_yearsA = np.interp(years_A, t_grid_I, I_sim)
A_from_sim_TI = (A_0
                 + w_T*T_sim_at_yearsA
                 + w_I*I_sim_at_yearsA
                 + w_P*P_A_hist
                 - w_C*C_A_hist)
A_from_sim_TI = np.clip(A_from_sim_TI, 0.0, 1.0)
 
 
# --- Attractiveness charts (3 across) ---
colA1, colA2, colA3 = st.columns(3)
 
with colA1:
    st.subheader("Chart 5 - Historical A")
    figA1 = go.Figure()
    figA1.add_trace(go.Scatter(x=years_A, y=A_hist,
                               mode="lines+markers", name="A historical",
                               line=dict(color="#9467bd", width=2.5),
                               marker=dict(size=10)))
    figA1.update_layout(xaxis_title="Year", yaxis_title="Attractiveness A(t)",
                        yaxis=dict(range=[0, 1.0]), height=420,
                        margin=dict(l=40, r=20, t=30, b=40))
    st.plotly_chart(figA1, use_container_width=True)
 
with colA2:
    st.subheader("Chart 6 - A from simulated T, I")
    figA2 = go.Figure()
    figA2.add_trace(go.Scatter(x=years_A, y=A_from_sim_TI,
                               mode="lines+markers",
                               name="A (sim T,I → formula)",
                               line=dict(color="#8c564b", width=2.5),
                               marker=dict(size=10)))
    figA2.update_layout(xaxis_title="Year", yaxis_title="Attractiveness A(t)",
                        yaxis=dict(range=[0, 1.0]), height=420,
                        margin=dict(l=40, r=20, t=30, b=40))
    st.plotly_chart(figA2, use_container_width=True)
 
with colA3:
    st.subheader("Chart 7 - A from historical T, I")
    figA3 = go.Figure()
    figA3.add_trace(go.Scatter(x=years_A, y=A_from_hist_TI,
                               mode="lines+markers",
                               name="A (hist T,I → formula)",
                               line=dict(color="#e377c2", width=2.5),
                               marker=dict(size=10)))
    figA3.update_layout(xaxis_title="Year", yaxis_title="Attractiveness A(t)",
                        yaxis=dict(range=[0, 1.0]), height=420,
                        margin=dict(l=40, r=20, t=30, b=40))
    st.plotly_chart(figA3, use_container_width=True)
 
 
# --- Attractiveness end-of-period numbers ---
st.markdown("---")
mA1, mA2, mA3, mA4 = st.columns(4)
mA1.metric("Historical A(2024)",       f"{A_hist[-1]:.3f}")
mA2.metric("A from sim T,I (2024)",    f"{A_from_sim_TI[-1]:.3f}")
mA3.metric("A from hist T,I (2024)",   f"{A_from_hist_TI[-1]:.3f}")
mA4.metric("Δ(sim - hist) at 2024",    f"{A_from_sim_TI[-1] - A_from_hist_TI[-1]:+.3f}")
 
 
# =============================================================================
#  SECTION 4 — FDI
# =============================================================================
st.markdown("---")
st.subheader("The ICT FDI equation")
st.latex(r"\frac{dF}{dt} \;=\; \beta \cdot A(t) \cdot \left(1 - \frac{F}{K_F}\right) \;-\; \gamma \cdot C(t) \cdot F")
 
st.subheader("Calibrated values (loaded from the file 'calibrate.py')")
cF1, cF2, cF3, cF4, cF5 = st.columns(5)
cF1.metric("β", f"{BETA_CAL:,.0f}")
cF2.metric("γ", f"{GAMMA_CAL:.4f}")
cF3.metric("K_F (€M)", f"{K_F_CAL:,.0f}")
cF4.metric("R²",   f"{R2_F:.3f}")
cF5.metric("RMSE", f"{RMSE_F:,.0f}")
 
st.info(
    f"**Calibration note.** K_F = {K_F_CAL:,.0f} €M is calibrated from grid infrastructure"
    f"forecasts. The International Energy Agency projects that data centers will grow from"
    f"22% to at least 30% of Ireland's electricity consumption between 2024 and 2030,"
    f"with total electricity demand rising from approximately 32 TWh to 42 TWh over the same period."
    f"The implied 2.04% growth in absolute data center electricity capacity is applied to the"
    f"historical peak ICT FDI stock (213,000 €M, observed in 2021) to derive K_F. \n\n"
    f"**Chart 9 uses A from the simulated T and I in Sections 1-2**, so "
    f"changing the Trust or Innovation sliders propagates through "
    f"Attractiveness and into FDI - the full coupled chain."
)


 
# --- Sidebar sliders for FDI ---
st.sidebar.header("FDI - equation parameters")
beta_s  = st.sidebar.slider("β - FDI inflow rate", 0.0, 500_000.0,
                            float(BETA_CAL), 1_000.0)
gamma_s = st.sidebar.slider("γ - FDI outflow per unit C", 0.0, 0.5,
                            float(GAMMA_CAL), 0.001, format="%.4f")
K_F_s   = st.sidebar.slider("K_F - carrying capacity (€M)",
                            100_000.0, 500_000.0, float(K_F_CAL), 5_000.0)
F0      = st.sidebar.slider("F(0) - initial ICT FDI (€M)",
                            0.0, 300_000.0, float(F_hist[0]), 1_000.0)
 
# --- Solve the FDI differential equation using A from SIMULATED T, I ---
# This is the chain: simulated T, I → formula A → drives FDI.
df_F = df_full.dropna(subset=["FDI", "C", "P"]).reset_index(drop=True)
years_F = df_F["Year"].to_numpy()
F_F_hist = df_F["FDI"].to_numpy()
C_F_hist = df_F["C"].to_numpy()
P_F_hist = df_F["P"].to_numpy()
 
 
def A_at(year):
    """Attractiveness from SIMULATED T, I (the chain).
    P and C are historical (exogenous). T and I come from the simulated
    trajectories in Sections 1 and 2."""
    T_t = float(np.interp(year, t_grid,   T_sim))
    I_t = float(np.interp(year, t_grid_I, I_sim))
    P_t = float(np.interp(year, years_F,  P_F_hist))
    C_t = float(np.interp(year, years_F,  C_F_hist))
    A_t = A_0 + w_T*T_t + w_I*I_t + w_P*P_t - w_C*C_t
    return max(0.0, min(1.0, A_t))
 
 
def C_at_F(year):
    return float(np.interp(year, years_F, C_F_hist))
 
 
year_start_F = float(years_F[0])
year_end_F   = float(years_F[-1])
n_steps_F = int((year_end_F - year_start_F) / dt) + 1
t_grid_F = np.linspace(year_start_F, year_end_F, n_steps_F)
 
F_sim = np.zeros(n_steps_F)
F_sim[0] = F0
for i in range(n_steps_F - 1):
    A_curr = A_at(t_grid_F[i])
    C_curr = C_at_F(t_grid_F[i])
    slope = (beta_s * A_curr * (1 - F_sim[i] / K_F_s)
             - gamma_s * C_curr * F_sim[i])
    F_sim[i + 1] = F_sim[i] + dt * slope
 
 
# --- FDI charts ---
left_F, right_F = st.columns(2)
with left_F:
    st.subheader("Chart 8 - Historical ICT FDI Stock")
    figF1 = go.Figure()
    figF1.add_trace(go.Scatter(x=years_F, y=F_F_hist,
                               mode="lines+markers",
                               name="F historical",
                               line=dict(color="#17becf", width=2.5),
                               marker=dict(size=10)))
    figF1.update_layout(xaxis_title="Year", yaxis_title="ICT FDI Stock (€M)",
                        height=420,
                        margin=dict(l=40, r=20, t=30, b=40))
    st.plotly_chart(figF1, use_container_width=True)
 
with right_F:
    st.subheader("Chart 9 - FDI from equation (using simulated A)")
    figF2 = go.Figure()
    figF2.add_trace(go.Scatter(x=t_grid_F, y=F_sim,
                               mode="lines",
                               name="F from equation",
                               line=dict(color="#bcbd22", width=2.5)))
    figF2.update_layout(xaxis_title="Year", yaxis_title="ICT FDI Stock (€M)",
                        height=420,
                        margin=dict(l=40, r=20, t=30, b=40))
    st.plotly_chart(figF2, use_container_width=True)
 
 
# --- FDI end-of-period numbers ---
st.markdown("---")
mF1, mF2, mF3 = st.columns(3)
mF1.metric("Historical F(2024) (€M)", f"{F_F_hist[-1]:,.0f}")
mF2.metric("Equation F(2024) (€M)",   f"{F_sim[-1]:,.0f}")
mF3.metric("Difference (€M)",         f"{F_sim[-1] - F_F_hist[-1]:+,.0f}")
 
 
# =============================================================================
#  SECTION 5 — COUPLED FULL MODEL (future projection)
# =============================================================================
st.markdown("---")
st.header("5. Future projection")
st.markdown(
    "This section runs **all four equations together** over a chosen horizon. "
    "Trust, Innovation and FDI feed each other through Attractiveness - the "
    "stocks evolve as a coupled system.  \n\n"
    "**E(t) is the single policy lever.** When E_future is dragged, two things "
    "happen automatically:  \n"
    "- *R(t) responds* because R is partly built from E, so the trust channel "
    "is activated.  \n"
    "- *C(t) responds* because C is built as ½·E + ½·DPO_cost in the dataset, "
    "so the compliance-cost channel is activated.  \n\n"
    "The net effect on F(t) is the resolution of these two channels."
)
 
# --- Sidebar: future-scenario controls ---
st.sidebar.header("Future scenario")
end_year = st.sidebar.slider("Projection end year", 2025, 2040, 2030, 1)
 
# E is the single policy lever
E_future = st.sidebar.slider(
    "E future - enforcement intensity (after 2024)",
    0.0, 1.0, float(E_hist[-1]), 0.01,
    help="The policy lever. Drag right for stricter enforcement, left for lighter touch.",
)
 
# Derive R_future from E_future, holding the resolution-rate component constant
# Historical: R = (E + complaint_resolution + breach_resolution) / 3
# So:   R_2024 = E_2024/3 + (complaint_2024 + breach_2024)/3
# Define residual = R_2024 − E_2024/3, then R_future = E_future/3 + residual
R_residual = float(R_hist[-1] - E_hist[-1] / 3.0)
R_future   = float(np.clip(E_future / 3.0 + R_residual, 0.0, 1.0))
 
# Derive C_future from E_future, holding the DPO half constant at its 2024 level
# C = 0.5*E + 0.5*DPO_norm  =>  DPO_norm_2024 = 2*C_2024 − E_2024
DPO_NORM_2024  = float(2.0 * C_hist[-1] - E_hist[-1])
C_future       = float(np.clip(0.5 * E_future + 0.5 * DPO_NORM_2024, 0.0, 1.0))
 
st.sidebar.caption(
    f"R future (derived) = **{R_future:.3f}**  \n"
    f"C future (derived) = **{C_future:.3f}**  \n"
    f"R and C respond automatically to E."
)
 
# P historical — pull from the dataset if available, else default.
# P_future is NOT user-adjustable; it is held automatically at the
# 2024 historical level (infrastructure & talent change slowly).
if "P" in df_full.columns:
    P_hist_full  = df_full.dropna(subset=["P"])["P"].to_numpy()
    P_years_full = df_full.dropna(subset=["P"])["Year"].to_numpy()
    P_2024 = float(P_hist_full[-1])
else:
    P_years_full = years_hist
    P_hist_full  = np.full_like(years_hist, 0.75, dtype=float)
    P_2024 = 0.75

P_future = P_2024
 
 
# --- Helper: build exogenous trajectories (historical interp + future const) ---
def exog_at(year, hist_years, hist_values, future_value, cutoff=2024):
    if year <= cutoff:
        return float(np.interp(year, hist_years, hist_values))
    return float(future_value)
 
 
# --- Run the coupled Euler simulation ---
start_year_c = 2018.0
end_year_c   = float(end_year)
dt_c = 0.05
n_steps_c = int((end_year_c - start_year_c) / dt_c) + 1
years_c   = np.linspace(start_year_c, end_year_c, n_steps_c)
 
T_traj = np.zeros(n_steps_c)
I_traj = np.zeros(n_steps_c)
F_traj = np.zeros(n_steps_c)
A_traj = np.zeros(n_steps_c)
E_traj = np.zeros(n_steps_c)
R_traj = np.zeros(n_steps_c)
C_traj = np.zeros(n_steps_c)
 
T_traj[0] = T0
I_traj[0] = I0
F_traj[0] = F0
 
for k in range(n_steps_c - 1):
    year_now = years_c[k]
    T_now, I_now, F_now = T_traj[k], I_traj[k], F_traj[k]
 
    # E(t): historical interpolation up to 2024, then E_future
    E_t = exog_at(year_now, years_E, E_hist, E_future)
 
    # R(t): derived from E(t) using the historical residual
    R_t = float(np.clip(E_t / 3.0 + R_residual, 0.0, 1.0))
 
    # C(t): derived from E(t) using the dataset construction C = 0.5*E + 0.5*DPO
    C_t = float(np.clip(0.5 * E_t + 0.5 * DPO_NORM_2024, 0.0, 1.0))
 
    # P(t): historical up to 2024, then P_future
    P_t = exog_at(year_now, P_years_full, P_hist_full, P_future)
 
    # Save the auxiliary trajectories for plotting
    E_traj[k] = E_t
    R_traj[k] = R_t
    C_traj[k] = C_t
 
    # Attractiveness from current state of T, I and current P, C
    A_t = A_0 + w_T*T_now + w_I*I_now + w_P*P_t - w_C*C_t
    A_t = max(0.0, min(1.0, A_t))
    A_traj[k] = A_t
 
    # Slopes — each equation uses the CURRENT state and current exogenous values
    dT = alpha_T * R_t * (1 - T_now) - delta_T * T_now
    dI = (alpha_I * F_now/(K_I + F_now) * (1 - I_now)
          - delta_I * C_t)
    dF = (beta_s * A_t * (1 - F_now/K_F_s)
          - gamma_s * C_t * F_now)
 
    # Euler update — all stocks advance together
    T_traj[k+1] = T_now + dt_c * dT
    I_traj[k+1] = I_now + dt_c * dI
    F_traj[k+1] = F_now + dt_c * dF
 
# Fill in the final-point auxiliaries
E_traj[-1] = exog_at(years_c[-1], years_E, E_hist, E_future)
R_traj[-1] = float(np.clip(E_traj[-1] / 3.0 + R_residual, 0.0, 1.0))
C_traj[-1] = float(np.clip(0.5 * E_traj[-1] + 0.5 * DPO_NORM_2024, 0.0, 1.0))
A_traj[-1] = max(0.0, min(1.0,
    A_0 + w_T*T_traj[-1] + w_I*I_traj[-1]
    + w_P*exog_at(years_c[-1], P_years_full, P_hist_full, P_future)
    - w_C*C_traj[-1]
))
 
 
# --- Plot a 2-by-2 panel of stocks + Attractiveness ---
fig_full = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Trust  T(t)", "Innovation  I(t)",
                    "Attractiveness  A(t)", "ICT FDI Stock  F(t) (€M)"),
)
fig_full.add_trace(go.Scatter(x=years_c, y=T_traj, name="T",
                              line=dict(color="#1f77b4", width=2.5)), row=1, col=1)
fig_full.add_trace(go.Scatter(x=years_c, y=I_traj, name="I",
                              line=dict(color="#2ca02c", width=2.5)), row=1, col=2)
fig_full.add_trace(go.Scatter(x=years_c, y=A_traj, name="A",
                              line=dict(color="#9467bd", width=2.5)), row=2, col=1)
fig_full.add_trace(go.Scatter(x=years_c, y=F_traj, name="F",
                              line=dict(color="#17becf", width=2.5)), row=2, col=2)
 
# Overlay historical points
fig_full.add_trace(go.Scatter(x=years_hist, y=T_hist, name="T hist",
                              mode="markers",
                              marker=dict(color="#1f77b4", size=9, symbol="circle-open")),
                   row=1, col=1)
fig_full.add_trace(go.Scatter(x=years_I, y=I_hist, name="I hist",
                              mode="markers",
                              marker=dict(color="#2ca02c", size=9, symbol="circle-open")),
                   row=1, col=2)
fig_full.add_trace(go.Scatter(x=years_A, y=A_hist, name="A hist",
                              mode="markers",
                              marker=dict(color="#9467bd", size=9, symbol="circle-open")),
                   row=2, col=1)
fig_full.add_trace(go.Scatter(x=years_F, y=F_F_hist, name="F hist",
                              mode="markers",
                              marker=dict(color="#17becf", size=9, symbol="circle-open")),
                   row=2, col=2)
 
# Mark the 2024 historical cutoff
for r in (1, 2):
    for c in (1, 2):
        fig_full.add_vline(x=2024, line_dash="dot", line_color="grey", row=r, col=c)
 
fig_full.update_layout(height=720, showlegend=False,
                       margin=dict(l=40, r=20, t=60, b=20))
fig_full.update_xaxes(title_text="Year")
st.plotly_chart(fig_full, use_container_width=True)
 
 
# # --- Plot the auxiliary inputs E, R, C side-by-side ---
# st.subheader("Auxiliary inputs driving the projection")
# fig_aux = go.Figure()
# fig_aux.add_trace(go.Scatter(x=years_c, y=E_traj, name="E(t) - enforcement",
#                              line=dict(color="#d62728", width=2.5)))
# fig_aux.add_trace(go.Scatter(x=years_c, y=R_traj, name="R(t) - credibility (derived)",
#                              line=dict(color="#1f77b4", width=2, dash="dash")))
# fig_aux.add_trace(go.Scatter(x=years_c, y=C_traj, name="C(t) - compliance cost (derived)",
#                              line=dict(color="#ff7f0e", width=2, dash="dash")))
# fig_aux.add_vline(x=2024, line_dash="dot", line_color="grey")
# fig_aux.update_layout(xaxis_title="Year", yaxis_title="Index (0-1)",
#                       yaxis=dict(range=[0, 1.1]),
#                       height=360,
#                       margin=dict(l=40, r=20, t=30, b=40),
#                       legend=dict(orientation="h", y=-0.2))
# st.plotly_chart(fig_aux, use_container_width=True)
 
 
# --- End-of-projection summary ---
st.subheader(f"End of projection ({int(end_year)})")
e1, e2, e3, e4 = st.columns(4)
e1.metric(f"T({int(end_year)})",        f"{T_traj[-1]:.3f}")
e2.metric(f"I({int(end_year)})",        f"{I_traj[-1]:.3f}")
e3.metric(f"A({int(end_year)})",        f"{A_traj[-1]:.3f}")
e4.metric(f"F({int(end_year)}) (€M)",   f"{F_traj[-1]:,.0f}")

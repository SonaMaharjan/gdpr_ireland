# Trust, Fines and Foreign Investment

### A System Dynamics Model of GDPR Enforcement in Ireland

Python code and an interactive Streamlit dashboard accompanying my master's thesis, "**Trust, Fines and Foreign Investment: A System Dynamics of GDPR Enforcement in Ireland**".

This project explores how GDPR enforcement intensity could influence Ireland's information and communication technology (ICT) foreign direct investment (FDI) stock through interconnected changes in trust, innovation capacity, compliance costs and investment attractiveness.

Developed as part of the Erasmus Mundus MSc in Transition, Innovation and Sustainability Environments (TISE).

## Research question

How does GDPR Enforcement Intensity affect the long-term ICT FDI Stock
trajectory of Ireland?

Enforcement creates competing pressures in the model: it can strengthen regulatory credibility and trust, while increasing compliance costs that constrain innovation and investment. System dynamics provides a framework for exploring how these mechanisms interact over time.

## What the application does

- Calibrates trust, innovation and FDI equation parameters using historical observations.
- Displays calibration diagnostics, including R² and root mean squared error (RMSE).
- Compares historical series with equation-based trajectories.
- Provides controls for model parameters, initial conditions and selected inputs.
- Simulates trust, innovation capacity and ICT FDI stock as a coupled system.
- Lets users vary future enforcement intensity and explore projections through 2040; the dashboard defaults to 2030.

The thesis examines lenient, baseline and strict enforcement scenarios. The dashboard provides a continuous enforcement slider for exploring alternative settings.

## Repository structure

```text
Thesis Code/
├── README.md
├── calibrate.py
├── mainapp.py
├── requirements.txt
└── data/
    └── historical_data.csv
```

| File | Purpose |
| --- | --- |
| `calibrate.py` | Estimates equation parameters and calibration diagnostics; calculates baseline attractiveness. |
| `mainapp.py` | Runs the Streamlit dashboard, numerical simulations and Plotly visualisations. |
| `requirements.txt` | Lists Python dependencies. |
| `data/historical_data.csv` | Supplies the historical observations required by calibration and the dashboard. |

## Model overview

The model has three stocks: trust, innovation capacity and ICT FDI stock. Investment attractiveness is calculated from trust, innovation, infrastructure and talent, and compliance costs.

$$\frac{dT}{dt}=\alpha_T R(t)(1-T)-\delta_T T$$

$$\frac{dI}{dt}=\alpha_I\frac{F}{K_I+F}(1-I)-\delta_I C(t)$$

$$A(t)=A_0+w_TT+w_II+w_PP-w_CC$$

$$\frac{dF}{dt}=\beta A(t)\left(1-\frac{F}{K_F}\right)-\gamma C(t)F$$

| Symbol | Meaning |
| --- | --- |
| `T` | Trust index |
| `I` | Innovation capacity index |
| `F` | ICT FDI stock, in EUR millions |
| `E` | GDPR enforcement intensity |
| `R` | Regulatory credibility |
| `C` | Compliance cost proxy |
| `P` | ICT infrastructure and talent index |
| `A` | Investment attractiveness index |

Parameters are fitted to annual changes using ordinary least squares, implemented through two-variable normal equations and Cramer's rule. The dashboard uses explicit Euler integration with a time step of 0.05 years and interpolation of historical inputs.

In the coupled simulation, enforcement affects both regulatory credibility and compliance costs. Non-enforcement components of those inputs are held at their final historical levels; infrastructure and talent are also held constant after 2024. Investment attractiveness is clipped to the interval [0, 1].

## Data

The main historical analysis covers **2018–2024**. A **2017 baseline row** is also required to calculate baseline investment attractiveness.

The CSV must contain these exact column names:

```text
Year,T,R,I,C,FDI,E,P,A
```

Use numeric values without thousands separators for compatibility across the calibration routines. Arrange rows chronologically, with one observation per year. Retain the complete annual series: the calibration code uses adjacent retained rows as annual changes and does not adjust for missing years.

| Column | Description / source used in the thesis |
| --- | --- |
| `Year` | Calendar year |
| `T` | Trust proxy based on Eurostat's Services Confidence Indicator |
| `R` | Regulatory credibility constructed from enforcement and complaint/breach resolution indicators |
| `I` | Normalised European Innovation Scoreboard index |
| `C` | Compliance cost proxy combining enforcement intensity and estimated DPO costs |
| `FDI` | Irish ICT FDI stock from the Central Statistics Office, in EUR millions |
| `E` | Enforcement index constructed from Irish Data Protection Commission annual reports |
| `P` | Infrastructure and talent index based on IMD digital competitiveness indicators |
| `A` | Attractiveness index based on EY attractiveness survey indicators |

These are operational proxies for the model's concepts; for example, the trust series is not a direct survey measure of trust in GDPR. Consult the thesis for indicator construction, normalisation, assumptions and source references.

## Run locally

From Terminal, navigate to your project folder:

```bash
cd "/path/to/Thesis Code"
```

Create and activate a virtual environment (macOS/Linux):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies and launch the dashboard:

```bash
python -m pip install -r requirements.txt
python -m streamlit run mainapp.py
```

Open the local URL printed by Streamlit. Run these commands from the project root because `calibrate.py` reads `data/historical_data.csv` using a relative path.

The dependencies are Streamlit, NumPy, pandas and Plotly. Versions in `requirements.txt` are minimum versions rather than a frozen environment.

## Interpretation and limitations

The thesis's model-based results suggest that compliance costs can affect investment more immediately, while trust builds gradually and becomes relatively more prominent over time. These are conditional outcomes of the specified model, rather than causal estimates or validated forecasts.

Key limitations include a short calibration period, proxy indicators, assumed attractiveness weights, fixed future inputs and sensitivity to parameters and model structure. The innovation channel should be interpreted as a structural assumption rather than an empirically validated mechanism. Calibration R² and RMSE describe the fit to annual changes; they do not establish out-of-sample predictive performance.

The current code fixes the FDI carrying capacity at `213000 × 2.04 = 434520` EUR millions during calibration. Calibration values are the dashboard defaults, and sliders allow users to explore alternatives. Scenario curves should be read in light of those assumptions.

## Author

**Sona Maharjan**  
Erasmus Mundus MSc, Transition, Innovation and Sustainability Environments (TISE)

**Thesis:** *Trust, Fines and Foreign Investment: A System Dynamics of GDPR Enforcement in Ireland* (2026).

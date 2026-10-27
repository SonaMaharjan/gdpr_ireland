"""
Calibration of the Trust equation
"""

import numpy as np
import pandas as pd

# ----------------------------------------X---------------------------------------
# 1. Calibrate for Trust
# ----------------------------------------X---------------------------------------
# *************************** a) Load the historical data ***************************

def calibrate_trust():

    df = pd.read_csv("data/historical_data.csv")

    # Keep only the rows where both T and R have values
    df = df.dropna(subset=["T", "R"])

    T = df["T"].to_numpy()       # Trust Index T(t)         
    R = df["R"].to_numpy()       # Regulatory Credibility R(t)
    years = df["Year"].to_numpy() # Years from 2018 to 2024

    print("Loaded data:")
    print(df[["Year", "T", "R"]])


    # *********************** b) Build the regression columns ***********************

    # T_now and R_now are the values at the START of each year-pair.
    # T_next is the value at the END (used to compute Delta_T).
    T_now  = T[:-1]
    print (T_now)
    T_next = T[1:]
    print (T_next)
    R_now  = R[:-1]

    change_T = T_next - T_now          # change in T from year t to year t+1
    X1 = R_now * (1 - T_now)          # multiplier for alpha_T
    X2 = -T_now                       # multiplier for change_T

    print("\nRegression table:")
    print("  pair          T(t)    R(t)    dT     X1=R(1-T)    X2=-T")
    for i in range(len(change_T)):
        print(f"  {years[i]}->{years[i+1]}  "
              f"{T_now[i]:.4f}  {R_now[i]:.4f}  "
              f"{change_T[i]:+.4f}   {X1[i]:.4f}     {X2[i]:+.4f}")

    # ************** c) Compute the five sums for the normal equations **************

    sum_X1_sq = np.sum(X1 * X1)
    sum_X2_sq = np.sum(X2 * X2)
    sum_X1_X2 = np.sum(X1 * X2)
    sum_X1_dT = np.sum(X1 * change_T)
    sum_X2_dT = np.sum(X2 * change_T)

    print("\nSums for the normal equations:")
    print(f"  sum X1*X1   = {sum_X1_sq:.4f}")
    print(f"  sum X2*X2   = {sum_X2_sq:.4f}")
    print(f"  sum X1*X2   = {sum_X1_X2:.4f}")
    print(f"  sum X1*dT   = {sum_X1_dT:.4f}")
    print(f"  sum X2*dT   = {sum_X2_dT:.4f}")



    # ************ d) Solve the 2-by-2 normal equations by Cramer's rule ************

    # The normal equations are:
    #   sum_X1_sq * alpha + sum_X1_X2 * delta = sum_X1_dT
    #   sum_X1_X2 * alpha + sum_X2_sq * delta = sum_X2_dT

    D       = sum_X1_sq * sum_X2_sq - sum_X1_X2 * sum_X1_X2
    D_alpha = sum_X1_dT * sum_X2_sq - sum_X1_X2 * sum_X2_dT
    D_delta = sum_X1_sq * sum_X2_dT - sum_X1_X2 * sum_X1_dT

    alpha_T = D_alpha / D
    delta_T = D_delta / D      

    print("\nCalibrated constants:")
    print(f"  alpha_T = {alpha_T:.4f}")
    print(f"  delta_T = {delta_T:.4f}")


    # ******************* e) Goodness of fit (R-squared and RMSE) *******************

    predicted = alpha_T * X1 + delta_T * X2
    residuals = change_T - predicted

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((change_T - np.mean(change_T)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    rmse = np.sqrt(ss_res / len(change_T))

    print(f"  R^2     = {r_squared:.3f}")
    print(f"  RMSE    = {rmse:.3f}")

    return alpha_T, delta_T, r_squared, rmse


#alpha_T, delta_T, r_squared, rmse = calibrate_trust()

#print (f"hello= {alpha_T, delta_T, r_squared, rmse}")
# ----------------------------------------X---------------------------------------
# 2. Calibrate for Innovation
# ----------------------------------------X---------------------------------------
def calibrate_innovation():
    df = pd.read_csv("data/historical_data.csv")

    # Keep only the rows where both T and R have values
    df = df.dropna(subset=["I", "C", "FDI"])

    F_str = df["FDI"].to_numpy()       # Trust Index T(t)  
    F = F_str.astype(np.float64) 
    C = df["C"].to_numpy()      
    I = df["I"].to_numpy()       # Regulatory Credibility R(t)
    years = df["Year"].to_numpy() # Years from 2018 to 2024

    print("Loaded data:")
    print(df[["Year", "I", "FDI"]])
    
    Sum_F = np.sum(F)


    K_I = np.mean(F)
    print (f"Half Saturation Point = {K_I}")

    # *********************** b) Build the regression columns ***********************

    # T_now and R_now are the values at the START of each year-pair.
    # T_next is the value at the END (used to compute Delta_T).
    I_now, I_next = I[:-1], I[1:]
    C_now, F_now = C[:-1], F[:-1]
    change_I = I_next - I_now
    X1 = F_now / (K_I + F_now) * (1 - I_now)   # multiplier for alpha_I
    X2 = -C_now 

    # ************** c) Compute the five sums for the normal equations **************

    sum_X1_sq = np.sum(X1 * X1)
    sum_X2_sq = np.sum(X2 * X2)
    sum_X1_X2 = np.sum(X1 * X2)
    sum_X1_dI = np.sum(X1 * change_I)
    sum_X2_dI = np.sum(X2 * change_I)

    print("\nSums for the normal equations:")
    print(f"  sum X1*X1   = {sum_X1_sq:.4f}")
    print(f"  sum X2*X2   = {sum_X2_sq:.4f}")
    print(f"  sum X1*X2   = {sum_X1_X2:.4f}")
    print(f"  sum X1*dT   = {sum_X1_dI:.4f}")
    print(f"  sum X2*dT   = {sum_X2_dI:.4f}")

    print ("all values", X1, X2, sum_X1_sq)



    # ************ d) Solve the 2-by-2 normal equations by Cramer's rule ************

    # The normal equations are:
    #   sum_X1_sq * alpha + sum_X1_X2 * delta = sum_X1_dT
    #   sum_X1_X2 * alpha + sum_X2_sq * delta = sum_X2_dT

    D       = sum_X1_sq * sum_X2_sq - sum_X1_X2 * sum_X1_X2
    D_alpha = sum_X1_dI * sum_X2_sq - sum_X1_X2 * sum_X2_dI
    D_delta = sum_X1_sq * sum_X2_dI - sum_X1_X2 * sum_X1_dI

    alpha_I = D_alpha / D
    delta_I = D_delta / D      

    print("\nCalibrated constants:")
    print(f"  alpha_T = {alpha_I:.4f}")
    print(f"  delta_T = {delta_I:.4f}")

    # ---- Goodness of fit ----
    predicted = alpha_I * X1 + delta_I * X2
    residuals = change_I - predicted
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((change_I - np.mean(change_I)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    rmse = np.sqrt(ss_res / len(change_I))
    return alpha_I, delta_I, K_I, r_squared, rmse


calibrate_innovation()




# ----------------------------------------X---------------------------------------
# 3. Calibrate for Attractiveness
# ----------------------------------------X---------------------------------------
def calibrate_attractiveness():
    """Compute A_0 from the 2017 row of the dataset.
 
    The Investment Attractiveness Index is
 
        A(t) = A_0 + w_T*T + w_I*I + w_P*P - w_C*C
 
    with equal weights w_T = w_I = w_P = w_C = 0.25 (neutral prior).
 
    A_0 is calibrated from 2017 data, when GDPR did not yet exist:
        T(2017) = 0, C(2017) = 0  (no GDPR -> no trust signal, no compliance cost)
        I(2017), P(2017)          (taken from historical data)
        A(2017)                   (from EY attractiveness survey)
 
    Solving the formula for A_0:
        A_0 = A(2017) - 0.25 * (T + I + P) + 0.25 * C   (evaluated at 2017)
 
    Returns
    -------
    A_0 : float
    w_T : float   (= 0.25)
    w_I : float   (= 0.25)
    w_P : float   (= 0.25)
    w_C : float   (= 0.25)
    """
    df = pd.read_csv("data/historical_data.csv")

    row_2017 = df[df["Year"] == 2017].iloc[0]
 
    # Pre-GDPR values (with sensible defaults if cells are blank in the CSV)
    # T_2017 = 0.0 if pd.isna(row_2017.get("T"))  else float(row_2017["T"])
    # C_2017 = 0.0 if pd.isna(row_2017.get("C"))  else float(row_2017["C"])
    T_2017 = 0.0
    C_2017 = 0.0
    I_2017 = float(row_2017["I"])
    P_2017 = float(row_2017["P"])
    A_2017 = float(row_2017["A"])
 
    # Equal weights (neutral prior)
    w_T = w_I = w_P = w_C = 1/4

    # w_C = 0.4
 
    # Solve for A_0
    A_0 = A_2017 - w_T*T_2017 - w_I*I_2017 - w_P*P_2017 + w_C*C_2017

    print (A_0)
 
    return A_0, w_T, w_I, w_P, w_C

calibrate_attractiveness()
# ----------------------------------------X---------------------------------------
# 2. Calibrate for FDI
# ----------------------------------------X---------------------------------------
def calibrate_fdi():
    """Fit beta and gamma with K_F fixed.
 
    The FDI equation is
 
        dF/dt = beta * A(t) * (1 - F/K_F) - gamma * C(t) * F
 
    K_F (carrying capacity) is fixed at K_F=213,000 by default — the
    empirically observed maximum of Irish ICT FDI stock (2021). With
    K_F fixed, beta and gamma are linear coefficients and we fit them
    by OLS via the same Cramer's rule procedure as Trust and Innovation.
 
    The attractiveness A(t) used here is the empirical EY-based column
    from the dataset, not the model-computed A — we fit how observed FDI
    responded to observed attractiveness.
 
    Returns
    -------
    beta : float
    gamma : float
    K_F : float
    r_squared : float
    rmse : float
    """
    
    df = pd.read_csv("data/historical_data.csv")
    df = df.dropna(subset=["FDI", "A", "C"])

    for col in ["FDI", "A", "C"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ""),
                errors="coerce"
            )
    F = df["FDI"].to_numpy()
    A = df["A"].to_numpy()
    C = df["C"].to_numpy()
    K_F = 213000.0*2.04


 
    F_now, F_next = F[:-1], F[1:]
    A_now, C_now = A[:-1], C[:-1]
    change_F = F_next - F_now
 
    X1 = A_now * (1 - F_now / K_F)       # multiplier for beta
    X2 = -C_now * F_now                  # multiplier for gamma
    print (X1, X2, C)

    sum_X1_sq = np.sum(X1 * X1)
    sum_X2_sq = np.sum(X2 * X2)
    sum_X1_X2 = np.sum(X1 * X2)
    sum_X1_dF = np.sum(X1 * change_F)
    sum_X2_dF = np.sum(X2 * change_F)
 
    D       = sum_X1_sq * sum_X2_sq - sum_X1_X2 * sum_X1_X2
    D_beta  = sum_X1_dF * sum_X2_sq - sum_X1_X2 * sum_X2_dF
    D_gamma = sum_X1_sq * sum_X2_dF - sum_X1_X2 * sum_X1_dF
    beta  = D_beta  / D
    gamma = D_gamma / D
 
    predicted = beta * X1 + gamma * X2
    residuals = change_F - predicted
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((change_F - np.mean(change_F)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    rmse = np.sqrt(ss_res / len(change_F))

    print (beta, gamma, D, D_beta, D_gamma)
 
    return beta, gamma, K_F, r_squared, rmse
 

calibrate_trust()


if __name__ == "__main__":
    print("This is the calibrate.py file.")

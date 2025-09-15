#!/usr/bin/env python3
"""
hypersonic_transition_demo.py

Runnable end-to-end prototype:
- improved atmosphere (ISA piecewise up to ~100 km)
- fixed dynamics (mass, acceleration = force/mass)
- simple Mach-dependent drag & lift approximations (replace with CFD lookup for fidelity)
- synthetic surrogate to produce N-factor (replace with LST/CFD-trained surrogate for real use)
- RandomForest classifier trained on synthetic dataset
- trajectory integration and transition detection

This is a demo/prototype. Do NOT treat predictions as validated results until surrogate and models are replaced with physics-derived data.
"""

import numpy as np
from scipy.integrate import odeint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------
# Constants / vehicle props
# -------------------------
R = 287.058            # J/(kg·K)
gamma = 1.4
mu0 = 1.716e-5         # kg/(m·s) at T0
T0_ref = 273.15        # K reference for mu0
S = 110.4              # Sutherland's constant (K)

# Vehicle
cone_half_angle = 10.0         # degrees (for reference)
nose_radius = 0.01             # m
x_sensor = 1.0                 # m along surface from stagnation (demo)
T_w = 300.0                    # K wall temperature (assumed)
m_vehicle = 500.0              # kg (must set realistically)
A_ref = np.pi * nose_radius**2 # reference frontal area (m^2) - change if needed

# -------------------------
# Atmosphere (piecewise ISA up to ~100 km)
# -------------------------
def atmosphere(h):
    """
    Simple piecewise ISA-like model (valid qualitatively up to ~100 km).
    h: altitude in meters
    returns: T (K), p (Pa), rho (kg/m^3)
    """
    # Layers simplified: troposphere up to 11km, lower stratosphere 11-20km, isothermal above
    if h < 11000:
        T = 288.15 - 0.0065 * h
        p = 101325.0 * (T / 288.15) ** 5.255877
    elif h < 20000:
        T = 216.65
        p11 = 101325.0 * (216.65 / 288.15) ** 5.255877
        p = p11 * np.exp(- (h - 11000) / 6341.97)
    else:
        # above 20 km, approximate exponential decay (very rough)
        T = 216.65
        rho0 = 0.08803  # approx rho at 20 km
        H =  scale_height =  scale_h = 7000.0
        rho = rho0 * np.exp(-(h - 20000)/scale_h)
        p = rho * R * T
        return T, p, rho
    rho = p / (R * T)
    return T, p, rho

# -------------------------
# Sutherland viscosity
# -------------------------
def viscosity_sutherland(T):
    """Sutherland's law for dynamic viscosity (kg/m s)."""
    return mu0 * (T / T0_ref)**1.5 * (T0_ref + S) / (T + S)

# -------------------------
# Simple drag and lift models (placeholder)
# Replace with CFD/lookup for production
# -------------------------
def drag_coeff(M, cone_half_angle_deg):
    """
    Very simple Mach-dependent drag coefficient model for demonstration.
    Not accurate for design use. Replace with tables or CFD.
    """
    # baseline CD for slender cone at low M
    base = 0.2 + 0.005 * cone_half_angle_deg
    if M < 0.8:
        return base + 0.3
    elif M < 2.0:
        return base + 0.6
    elif M < 5.0:
        return base + 0.9
    else:
        # hypersonic blunt-body trend: CD increases; this is heuristic
        return base + 1.2

def lift_coeff(M, cone_half_angle_deg):
    """Assume small lift for symmetric cone; return small value."""
    # For a symmetric cone with small angle, CL is small; model as near zero
    return 0.01 * np.tan(np.radians(cone_half_angle_deg))

# -------------------------
# Trajectory model (fixed physics)
# -------------------------
def trajectory_model(state, t, params):
    """
    state: [h, U, theta]
      h: altitude (m)
      U: velocity magnitude (m/s)
      theta: flight-path angle (rad), positive is climbing
    returns state derivatives [dhdt, dUdt, dthetadt]
    """
    h, U, theta = state
    m = params['m']
    T_inf, p_inf, rho_inf = atmosphere(h)
    # sanity clamping
    if T_inf <= 0 or rho_inf <= 0:
        # Return zeros to avoid blowing up; real solution should handle this differently
        return [0.0, 0.0, 0.0]
    a = np.sqrt(gamma * R * T_inf)
    M = max(U / a, 1e-6)

    C_D = drag_coeff(M, params['cone_half_angle'])
    C_L = lift_coeff(M, params['cone_half_angle'])

    D = 0.5 * rho_inf * U**2 * C_D * params['A_ref']
    L = 0.5 * rho_inf * U**2 * C_L * params['A_ref']

    # gravity variation
    g0 = 9.80665
    R_earth = 6371e3
    g = g0 * (R_earth / (R_earth + h))**2

    dUdt = -D / m - g * np.sin(theta)
    # Avoid division by zero
    dthetadt = 0.0
    if U > 1e-3:
        dthetadt = (L / m) / U - (g * np.cos(theta)) / U
    dhdt = U * np.sin(theta)
    return [dhdt, dUdt, dthetadt]

# -------------------------
# Synthetic N-factor generator (DEMO only)
# Replace with LST solver or surrogate trained on LST/CFD data
# -------------------------
def synthetic_N_from_physics(Re_x, M_e, T_w_over_T_e):
    """
    Create a synthetic, physically-motivated N-factor:
    - higher Re tends to promote instability growth (but scaled)
    - higher Mach has stabilizing/destabilizing effects; we include mild dependency
    - higher wall-to-edge temperature ratio tends to stabilize (higher wall temp reduces gradient)
    This is a synthetic formula for demo/training only.
    """
    # scaled non-dimensional inputs to keep ranges reasonable
    Re_s = np.log10(max(Re_x, 1e3))
    M_s = np.clip(M_e, 0.1, 50)
    Tw_ratio = T_w_over_T_e

    # synthetic formula (not physical truth)
    N_mean = 0.5 * Re_s / (M_s + 1.0) - 1.0 * (Tw_ratio - 1.0)
    # center near typical transition N ~ 9..12 by shifting scale
    N = 0.8 * N_mean + 8.5
    # add small noise
    N = N + np.random.normal(scale=0.5)
    return float(N)

# -------------------------
# Calculate N, Re_x, M_e function (uses synthetic N)
# -------------------------
def calculate_N_factor(h, U, T_inf, rho_inf, x_sensor_local, T_w_local):
    a = np.sqrt(gamma * R * T_inf)
    M_e = max(U / a, 1e-6)
    T_e = T_inf
    # recovery temperature estimate (approx)
    Pr = 0.72
    r = np.sqrt(Pr)
    T_r = T_e * (1.0 + (gamma - 1.0)/2.0 * r * M_e**2)
    # reference temperature
    T_star = 0.5 * (T_w_local + T_e) + 0.22 * (T_r - T_e)
    mu_star = viscosity_sutherland(max(T_star, 1.0))
    Re_x = rho_inf * U * max(1e-6, x_sensor_local) / max(mu_star, 1e-12)
    # For demo, use synthetic function to compute N
    N = synthetic_N_from_physics(Re_x, M_e, T_w_local / T_e)
    return N, Re_x, M_e

# -------------------------
# Build synthetic dataset for classifier (DEMO)
# Replace this block with dataset derived from LST/CFD/wind-tunnel
# -------------------------
def build_synthetic_dataset(n_samples=5000, random_state=0):
    rng = np.random.default_rng(random_state)
    X = []
    y = []
    for _ in range(n_samples):
        # sample free-stream conditions covering a wide envelope
        h = 1e3 * rng.uniform(1.0, 100.0)     # 1 km to 100 km
        U = rng.uniform(800.0, 9000.0)        # 0.8 km/s to 9 km/s (~Mach range)
        T_inf, p_inf, rho_inf = atmosphere(h)
        # avoid bad states
        if rho_inf <= 0 or T_inf <= 0:
            continue
        N, Re_x, M_e = calculate_N_factor(h, U, T_inf, rho_inf, x_sensor, T_w)
        # label assignment (0: laminar, 1: in-transition, 2: turbulent)
        # use threshold on N (example): laminar if N < 7.5, transition if 7.5<=N<=11.5, turbulent if >11.5
        if N < 7.5:
            label = 0
        elif N <= 11.5:
            label = 1
        else:
            label = 2
        # features: [N, Re_x, M_e, T_w/T_e]
        feat = [N, Re_x, M_e, T_w / max(T_inf, 1.0)]
        X.append(feat)
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X, y

# -------------------------
# Train classifier (Random Forest)
# -------------------------
def train_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Classifier evaluation on synthetic test set:")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    return clf

# -------------------------
# Main simulation & detection
# -------------------------
def run_simulation_and_detect_transition(clf):
    # initial state (example)
    # start at 100 km, 7 km/s, -20 deg flight-path (descending)
    state0 = [100000.0, 7000.0, -20.0 * np.pi / 180.0]
    t = np.linspace(0.0, 1000.0, 2000)  # s
    params = {'m': m_vehicle, 'cone_half_angle': cone_half_angle, 'A_ref': A_ref}
    states = odeint(trajectory_model, state0, t, args=(params,), atol=1e-6, rtol=1e-6)

    h_vec = states[:, 0]
    U_vec = states[:, 1]
    theta_vec = states[:, 2]

    h_trans = None
    M_trans = None
    transition_index = None

    for i, h in enumerate(h_vec):
        U = U_vec[i]
        # stop if vehicle has slowed drastically or crashed
        if U <= 50 or h <= 0:
            break
        T_inf, p_inf, rho_inf = atmosphere(h)
        if rho_inf <= 0 or T_inf <= 0:
            continue
        N, Re_x, M_e = calculate_N_factor(h, U, T_inf, rho_inf, x_sensor, T_w)
        features = np.array([[N, Re_x, M_e, T_w / T_inf]])
        state_pred = clf.predict(features)[0]
        # state: 1 means in-transition (per synthetic labels)
        if state_pred == 1:
            h_trans = h
            M_trans = U / np.sqrt(gamma * R * T_inf)
            transition_index = i
            break

    return {
        'h_trans': h_trans,
        'M_trans': M_trans,
        'states': states,
        't': t,
        'transition_index': transition_index
    }

# -------------------------
# Run everything
# -------------------------
def main():
    print("Building synthetic dataset...")
    X, y = build_synthetic_dataset(n_samples=6000, random_state=1)
    print("Dataset shape:", X.shape, y.shape)
    print("Training classifier...")
    clf = train_classifier(X, y)
    print("Running trajectory simulation and detecting transition (synthetic)...")
    results = run_simulation_and_detect_transition(clf)

    if results['h_trans'] is not None:
        print(f"Predicted transition altitude: {results['h_trans']:.1f} m")
        print(f"Predicted transition Mach number: {results['M_trans']:.3f}")
    else:
        print("No transition detected during the simulated trajectory (with current surrogate/classifier).")

if __name__ == "__main__":
    main()

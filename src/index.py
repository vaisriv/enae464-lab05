"""
ENAE464 Lab05: Ludwieg-Tube Experiments on a Free-Flying Cone
Analysis of a free-flying cone in a Mach-4 Ludwieg tube
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# --- Paths -------------------------------------------------------------------

DATA_DIR = Path("./data/mfg_m4_cone")
OUTPUT_TEXT_DIR = Path("./outputs/text")
OUTPUT_FIGS_DIR = Path("./outputs/figures")

OUTPUT_TEXT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGS_DIR.mkdir(parents=True, exist_ok=True)

# --- Header ------------------------------------------------------------------

print("=" * 80)
print("ENAE464 Lab05: Ludwieg-Tube Experiments on a Free-Flying Cone")
print("=" * 80)
print()

# --- Constants ---------------------------------------------------------------

# Physical constants
GAMMA = 1.4  # Specific heat ratio for air
R_GAS = 287.0  # Gas constant for air [J/(kg·K)]

# Experimental parameters (from experimental_setup.txt)
M_EXPERIMENTAL = 3.79  # Experimental test section Mach number
P_FILL = 0.7e5  # Fill pressure [Pa] (0.7 bar)
T_FILL = 296.0  # Fill temperature [K]
FRAME_RATE = 10000.0  # Camera frame rate [Hz]
CONE_MASS = 0.0018  # Cone mass [kg] (1.8 g)
PIXEL_TO_MM = 12.6  # Conversion factor [pixels/mm]
MM_PER_PIXEL = 1.0 / PIXEL_TO_MM  # Conversion factor [mm/pixel]
M_PER_PIXEL = MM_PER_PIXEL / 1000.0  # Conversion factor [m/pixel]

# Facility geometry
AREA_RATIO_TUBE = 11.4  # At/A* for charge tube

# Derived constants
DT = 1.0 / FRAME_RATE  # Time step between frames [s]

# ---  Functions --------------------------------------------------------------


def calculate_mt_from_area_ratio(area_ratio, gamma=GAMMA, tol=1e-8):
    """
    Solve for subsonic Mach number Mt in charge tube from area ratio.

    Uses Eq. 1: (A/A*)^2 = (1/M^2) * [2/(γ+1) * (1 + (γ-1)/2 * M^2)]^[(γ+1)/(γ-1)]

    Parameters:
    -----------
    area_ratio : float
        At/A* ratio
    gamma : float
        Specific heat ratio
    tol : float
        Convergence tolerance for numerical solver

    Returns:
    --------
    float : Mt (subsonic solution)
    """
    from scipy.optimize import brentq

    def area_mach_relation(M, A_ratio, g):
        term1 = (2.0 / (g + 1.0)) * (1.0 + ((g - 1.0) / 2.0) * M**2)
        term2 = (g + 1.0) / (g - 1.0)
        return (1.0 / M**2) * term1**term2 - A_ratio**2

    # Use Brent's method with bounds to ensure positive subsonic solution
    # For subsonic flow, M is between a very small number and 1.0
    Mt = brentq(area_mach_relation, 0.001, 0.999, args=(area_ratio, gamma))

    return Mt


def calculate_expansion_conditions(Mt, p_fill, T_fill, gamma=GAMMA):
    """
    Calculate pressure and temperature behind expansion wave.

    Uses Eqs. 2-3:
    pt/p_fill = [1 + (γ-1)/2 * Mt^2]^[-2γ/(γ-1)]
    Tt/T_fill = [1 + (γ-1)/2 * Mt^2]^[-2]

    Parameters:
    -----------
    Mt : float
        Mach number in charge tube
    p_fill, T_fill : float
        Fill pressure [Pa] and temperature [K]
    gamma : float
        Specific heat ratio

    Returns:
    --------
    tuple : (pt, Tt)
    """
    term = 1.0 + ((gamma - 1.0) / 2.0) * Mt**2

    pt = p_fill * term ** (-2.0 * gamma / (gamma - 1.0))
    Tt = T_fill * term ** (-2.0)

    return pt, Tt


def calculate_stagnation_conditions(Mt, pt, Tt, gamma=GAMMA):
    """
    Calculate stagnation pressure and temperature.

    Uses Eqs. 4-5:
    p0/pt = [1 + (γ-1)/2 * Mt^2]^[γ/(γ-1)]
    T0/Tt = 1 + (γ-1)/2 * Mt^2

    Parameters:
    -----------
    Mt : float
        Mach number in charge tube
    pt, Tt : float
        Pressure [Pa] and temperature [K] behind expansion
    gamma : float
        Specific heat ratio

    Returns:
    --------
    tuple : (p0, T0)
    """
    term = 1.0 + ((gamma - 1.0) / 2.0) * Mt**2

    p0 = pt * term ** (gamma / (gamma - 1.0))
    T0 = Tt * term

    return p0, T0


def calculate_freestream_conditions(M, p0, T0, R=R_GAS, gamma=GAMMA):
    """
    Calculate freestream static conditions from stagnation values.

    Uses Eqs. 7-8 plus ideal gas law and speed of sound.

    Parameters:
    -----------
    M : float
        Freestream Mach number
    p0, T0 : float
        Stagnation pressure [Pa] and temperature [K]
    R : float
        Gas constant [J/(kg·K)]
    gamma : float
        Specific heat ratio

    Returns:
    --------
    dict : {'p': p_inf, 'T': T_inf, 'rho': rho_inf, 'V': V_inf, 'a': a_inf}
    """
    term = 1.0 + ((gamma - 1.0) / 2.0) * M**2

    p_inf = p0 / (term ** (gamma / (gamma - 1.0)))
    T_inf = T0 / term
    rho_inf = p_inf / (R * T_inf)
    a_inf = np.sqrt(gamma * R * T_inf)
    V_inf = M * a_inf

    return {"p": p_inf, "T": T_inf, "rho": rho_inf, "a": a_inf, "V": V_inf}


def load_edge_data(frame_num, edges_dir=DATA_DIR / "edges"):
    """Load upper and lower edge coordinates for a single frame."""
    upper_file = edges_dir / f"frame{frame_num}_upper.txt"
    lower_file = edges_dir / f"frame{frame_num}_lower.txt"

    upper = pd.read_csv(upper_file, sep="\t")
    lower = pd.read_csv(lower_file, sep="\t")

    return upper, lower


def fit_cone_edges_and_find_vertex(upper, lower):
    """
    Fit straight lines to upper and lower edges, find vertex.

    Returns:
    --------
    dict with keys:
        'vertex_x', 'vertex_y': vertex coordinates [pixels]
        'upper_slope', 'lower_slope': edge slopes
        'upper_intercept', 'lower_intercept': edge y-intercepts
        'cone_angle_rad', 'cone_angle_deg': half-angle
        'base_height_pixels': vertical distance between edges at back
    """
    # Fit lines: y = slope * x + intercept
    upper_coeffs = np.polyfit(upper["x"], upper["y"], 1)
    lower_coeffs = np.polyfit(lower["x"], lower["y"], 1)

    m_upper, b_upper = upper_coeffs
    m_lower, b_lower = lower_coeffs

    # Find intersection (vertex)
    # m_upper * x + b_upper = m_lower * x + b_lower
    vertex_x = (b_lower - b_upper) / (m_upper - m_lower)
    vertex_y = m_upper * vertex_x + b_upper

    # Cone half-angle from average slope
    avg_slope = (abs(m_upper) + abs(m_lower)) / 2.0
    cone_angle_rad = np.arctan(avg_slope)
    cone_angle_deg = np.degrees(cone_angle_rad)

    # Base height (at rightmost x position) - use actual edge points, not extrapolation
    x_max = min(upper["x"].max(), lower["x"].max())

    # Find actual edge points near the base (within 5 pixels of x_max)
    upper_base_points = upper[upper["x"] >= x_max - 5]
    lower_base_points = lower[lower["x"] >= x_max - 5]

    # Get actual y values at the base
    if len(upper_base_points) > 0 and len(lower_base_points) > 0:
        y_upper_base = upper_base_points["y"].mean()
        y_lower_base = lower_base_points["y"].mean()
        base_height_pixels = abs(y_upper_base - y_lower_base)
    else:
        # Fallback to extrapolation if no points found
        y_upper_base = m_upper * x_max + b_upper
        y_lower_base = m_lower * x_max + b_lower
        base_height_pixels = abs(y_upper_base - y_lower_base)

    return {
        "vertex_x": vertex_x,
        "vertex_y": vertex_y,
        "upper_slope": m_upper,
        "lower_slope": m_lower,
        "upper_intercept": b_upper,
        "lower_intercept": b_lower,
        "cone_angle_rad": cone_angle_rad,
        "cone_angle_deg": cone_angle_deg,
        "base_height_pixels": base_height_pixels,
    }


def rasmussen_pressure_coefficient(M, theta_c, gamma=GAMMA):
    """
    Calculate pressure coefficient using Rasmussen's approximation (Eq. 11).

    Cp/θc² ≈ 1 + [(γ+1)M²]/(θc²+2) + [(γ-1)M²]/(θc²+2)*ln[(γ+1)/2 + 1/(M²θc²)]

    Parameters:
    -----------
    M : float
        Freestream Mach number
    theta_c : float
        Cone half-angle [radians]
    gamma : float
        Specific heat ratio

    Returns:
    --------
    float : Cp (pressure coefficient)
    """
    theta_c_sq = theta_c**2

    term1 = 1.0
    term2 = ((gamma + 1.0) * M**2) / (theta_c_sq + 2.0)

    log_arg = (gamma + 1.0) / 2.0 + 1.0 / (M**2 * theta_c_sq)
    term3 = ((gamma - 1.0) * M**2) / (theta_c_sq + 2.0) * np.log(log_arg)

    Cp_over_theta_sq = term1 + term2 + term3
    Cp = Cp_over_theta_sq * theta_c_sq

    return Cp


def rasmussen_shock_angle(M, theta_c, gamma=GAMMA):
    """
    Calculate shock angle using Rasmussen's approximation (Eq. 10).

    sin(β)/sin(θc) ≈ √[(γ+1)/2 + 1/(M² sin²θc)]

    Parameters:
    -----------
    M : float
        Freestream Mach number
    theta_c : float
        Cone half-angle [radians]
    gamma : float
        Specific heat ratio

    Returns:
    --------
    float : β (shock angle) [radians]
    """
    sin_theta_c = np.sin(theta_c)

    term = (gamma + 1.0) / 2.0 + 1.0 / (M**2 * sin_theta_c**2)
    sin_beta = sin_theta_c * np.sqrt(term)

    beta = np.arcsin(sin_beta)

    return beta


# --- Analysis ----------------------------------------------------------------

print("1. Calculating Freestream Conditions")
print("-" * 80)

# Step 1: Tube Mach number
Mt = calculate_mt_from_area_ratio(AREA_RATIO_TUBE, GAMMA)
print(f"  Mt (charge tube Mach number): {Mt:.6f}")

# Step 2: Expansion wave conditions
pt, Tt = calculate_expansion_conditions(Mt, P_FILL, T_FILL, GAMMA)
print(f"  pt (pressure behind expansion): {pt:.2f} Pa")
print(f"  Tt (temperature behind expansion): {Tt:.2f} K")

# Step 3: Stagnation conditions
p0, T0 = calculate_stagnation_conditions(Mt, pt, Tt, GAMMA)
print(f"  p0 (stagnation pressure): {p0:.2f} Pa")
print(f"  T0 (stagnation temperature): {T0:.2f} K")

# Step 4: Freestream conditions
freestream = calculate_freestream_conditions(M_EXPERIMENTAL, p0, T0, R_GAS, GAMMA)
print(f"  M∞ (freestream Mach number): {M_EXPERIMENTAL}")
print(f"  p∞ (freestream pressure): {freestream['p']:.2f} Pa")
print(f"  T∞ (freestream temperature): {freestream['T']:.2f} K")
print(f"  ρ∞ (freestream density): {freestream['rho']:.6f} kg/m³")
print(f"  a∞ (freestream sound speed): {freestream['a']:.2f} m/s")
print(f"  V∞ (freestream velocity): {freestream['V']:.2f} m/s")
print()

print("2. Loading and Processing Edge Detection Data")
print("-" * 80)

# Determine available frames
edges_dir = DATA_DIR / "edges"
frame_files = sorted(edges_dir.glob("frame*_upper.txt"))
frame_numbers = [int(f.stem.split("_")[0].replace("frame", "")) for f in frame_files]
n_frames = len(frame_numbers)

print(f"  Found {n_frames} frames with edge data")
print(f"  Frame range: {min(frame_numbers)} to {max(frame_numbers)}")
print()

print("3. Extracting Cone Geometry")
print("-" * 80)

# Extract geometry from all frames
geometries = []
for frame_num in frame_numbers:
    upper, lower = load_edge_data(frame_num)
    geom = fit_cone_edges_and_find_vertex(upper, lower)
    geom["frame"] = frame_num
    geom["time"] = frame_num * DT
    geometries.append(geom)

geom_df = pd.DataFrame(geometries)

# Calculate median cone angle from all frames
median_cone_angle_deg = geom_df["cone_angle_deg"].median()
median_cone_angle_rad = geom_df["cone_angle_rad"].median()

# Use base height from FIRST 20 frames (cone is closest to camera, least perspective error)
initial_base_height_px = geom_df.iloc[:20]["base_height_pixels"].median()
base_radius_m = (initial_base_height_px / 2.0) * M_PER_PIXEL

print(
    f"  Median cone half-angle: {median_cone_angle_deg:.3f}° ({median_cone_angle_rad:.6f} rad)"
)
print(f"  Initial base diameter (first 20 frames): {initial_base_height_px:.2f} pixels")
print(f"  Base radius: {base_radius_m * 1000:.3f} mm ({base_radius_m:.6f} m)")
print(f"  Base area: {np.pi * base_radius_m**2:.6e} m²")
print()

# Store for later use
THETA_C = median_cone_angle_rad
R_BASE = base_radius_m
A_BASE = np.pi * R_BASE**2

print("4. Analyzing Cone Motion and Calculating Drag")
print("-" * 80)

# Extract vertex positions over time (convert to meters)
time_array = geom_df["time"].values
x_position_pixels = geom_df["vertex_x"].values
x_position_m = x_position_pixels * M_PER_PIXEL

# Fit quadratic polynomial: x = a0 + a1*t + a2*t^2
coeffs = np.polyfit(time_array, x_position_m, 2)
a2, a1, a0 = coeffs  # Note: polyfit returns highest degree first

print("  Quadratic fit coefficients:")
print(f"    a0 = {a0:.6f} m")
print(f"    a1 = {a1:.6f} m/s")
print(f"    a2 = {a2:.6f} m/s²")
print()

# Calculate acceleration
ax_experimental = 2.0 * a2

print(f"  Experimental acceleration: ax = {ax_experimental:.3f} m/s²")
print()

# Calculate drag force
drag_force_experimental = CONE_MASS * ax_experimental

print(f"  Experimental drag force: D = {drag_force_experimental:.6f} N")
print()

# Calculate experimental drag coefficient
q_inf = 0.5 * freestream["rho"] * freestream["V"] ** 2  # Dynamic pressure
CD_experimental = drag_force_experimental / (q_inf * A_BASE)

print(f"  Dynamic pressure: q∞ = {q_inf:.2f} Pa")
print(f"  Experimental drag coefficient: CD = {CD_experimental:.6f}")
print()

# Generate fit curve for visualization
x_fit = a0 + a1 * time_array + a2 * time_array**2

print("5. Calculating Theoretical Drag (Rasmussen Approximation)")
print("-" * 80)

# Calculate Cp
Cp_theory = rasmussen_pressure_coefficient(M_EXPERIMENTAL, THETA_C, GAMMA)
print(f"  Rasmussen Cp (inviscid): {Cp_theory:.6f}")
print()

# For slender cones, Cp ≈ CD_inviscid
CD_inviscid = Cp_theory

# Calculate theoretical drag force
drag_force_inviscid = CD_inviscid * q_inf * A_BASE

print(f"  Inviscid drag coefficient: CD_inviscid = {CD_inviscid:.6f}")
print(f"  Inviscid drag force: D_inviscid = {drag_force_inviscid:.6f} N")
print()

# Calculate theoretical acceleration
ax_inviscid = drag_force_inviscid / CONE_MASS

print(f"  Inviscid acceleration: ax_inviscid = {ax_inviscid:.3f} m/s²")
print()

print("6. Estimating Viscous Drag Component")
print("-" * 80)

# Viscous drag is the difference between experimental and inviscid
drag_force_viscous = drag_force_experimental - drag_force_inviscid
CD_viscous = CD_experimental - CD_inviscid

# Percentage breakdown
if drag_force_experimental != 0:
    viscous_fraction = drag_force_viscous / drag_force_experimental * 100.0
    inviscid_fraction = drag_force_inviscid / drag_force_experimental * 100.0
else:
    viscous_fraction = 0.0
    inviscid_fraction = 0.0

print(f"  Total drag (experimental): {drag_force_experimental:.6f} N")
print(
    f"  Inviscid drag (theoretical): {drag_force_inviscid:.6f} N ({inviscid_fraction:.1f}%)"
)
print(
    f"  Viscous drag (estimated): {drag_force_viscous:.6f} N ({viscous_fraction:.1f}%)"
)
print()
print(f"  CD_experimental: {CD_experimental:.6f}")
print(f"  CD_inviscid: {CD_inviscid:.6f}")
print(f"  CD_viscous: {CD_viscous:.6f}")
print()

print("7. Calculating Theoretical Shock Angle")
print("-" * 80)

# Calculate shock angle
beta_rad = rasmussen_shock_angle(M_EXPERIMENTAL, THETA_C, GAMMA)
beta_deg = np.degrees(beta_rad)

print(f"  Cone half-angle: θc = {np.degrees(THETA_C):.3f}°")
print(f"  Shock angle: β = {beta_deg:.3f}° ({beta_rad:.6f} rad)")
print()

# --- Visualization -----------------------------------------------------------

print("8. Generating Figures")
print("-" * 80)

# Figure 1: Position vs Time
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    time_array * 1000,
    x_position_m * 1000,
    "o",
    label="Measured vertex position",
    markersize=4,
    alpha=0.6,
)
ax.plot(
    time_array * 1000,
    x_fit * 1000,
    "-",
    label=f"Quadratic fit: $x = {a0:.3f} + {a1:.3f}t + {a2:.3f}t^2$",
    linewidth=2,
    color="red",
)
ax.set_xlabel("Time [ms]", fontsize=12)
ax.set_ylabel("Position [mm]", fontsize=12)
ax.set_title("Cone Vertex Position vs. Time", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_FIGS_DIR / "position_vs_time.png", dpi=300)
plt.close()
print("  Saved: position_vs_time.png")

# Figure 2: Velocity vs Time
velocity_measured = np.gradient(x_position_m, time_array)
velocity_fit = a1 + 2 * a2 * time_array

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    time_array * 1000,
    velocity_measured,
    "o",
    label="Numerical derivative",
    markersize=3,
    alpha=0.5,
)
ax.plot(
    time_array * 1000,
    velocity_fit,
    "-",
    label=f"From fit: $v = {a1:.3f} + 2({a2:.3f})t$",
    linewidth=2,
    color="red",
)
ax.set_xlabel("Time [ms]", fontsize=12)
ax.set_ylabel("Velocity [m/s]", fontsize=12)
ax.set_title("Cone Velocity vs. Time", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_FIGS_DIR / "velocity_vs_time.png", dpi=300)
plt.close()
print("  Saved: velocity_vs_time.png")

# Figure 3: Acceleration vs Time
acceleration_measured = np.gradient(velocity_measured, time_array)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    time_array * 1000,
    acceleration_measured,
    "o",
    label="From numerical derivatives",
    markersize=3,
    alpha=0.5,
)
ax.axhline(
    y=ax_experimental,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"From quadratic fit: $a_x = {ax_experimental:.3f}$ m/s²",
)
ax.set_xlabel("Time [ms]", fontsize=12)
ax.set_ylabel("Acceleration [m/s²]", fontsize=12)
ax.set_title("Cone Acceleration vs. Time", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_FIGS_DIR / "acceleration_vs_time.png", dpi=300)
plt.close()
print("  Saved: acceleration_vs_time.png")

# Figure 4: Theoretical shock shape overlay on experimental image
# Select a frame near the middle with low angle of attack
middle_frame = frame_numbers[len(frame_numbers) // 2]
upper_mid, lower_mid = load_edge_data(middle_frame)
geom_mid = geom_df[geom_df["frame"] == middle_frame].iloc[0]

# Load the actual shadowgraph image
image_path = DATA_DIR / "images" / f"cone_{middle_frame:03d}.tif"
img = Image.open(image_path)
img_array = np.array(img)

fig, ax = plt.subplots(figsize=(14, 10))

# Display the shadowgraph image
ax.imshow(
    img_array,
    cmap="gray",
    alpha=0.8,
    extent=(0, img_array.shape[1], img_array.shape[0], 0),
)

# Plot cone edges
ax.plot(
    upper_mid["x"], upper_mid["y"], "b-", linewidth=2.5, label="Detected cone edges"
)
ax.plot(lower_mid["x"], lower_mid["y"], "b-", linewidth=2.5)

# Plot vertex
vx, vy = geom_mid["vertex_x"], geom_mid["vertex_y"]
ax.plot(
    vx,
    vy,
    "go",
    markersize=12,
    label="Cone vertex",
    zorder=5,
    markeredgecolor="white",
    markeredgewidth=1.5,
)

# Plot theoretical shock rays
x_shock_range = np.linspace(vx, upper_mid["x"].max(), 200)

# Upper shock: y = vy + (x - vx) * tan(β)
y_shock_upper = vy + (x_shock_range - vx) * np.tan(beta_rad)

# Lower shock: y = vy - (x - vx) * tan(β)
y_shock_lower = vy - (x_shock_range - vx) * np.tan(beta_rad)

ax.plot(
    x_shock_range,
    y_shock_upper,
    "r--",
    linewidth=3,
    label=f"Theoretical shock (β = {beta_deg:.2f}°)",
)
ax.plot(x_shock_range, y_shock_lower, "r--", linewidth=3)

# Annotations
ax.text(
    vx + 50,
    vy + 60,
    f"θc = {np.degrees(THETA_C):.2f}°",
    fontsize=12,
    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8),
    fontweight="bold",
)
ax.text(
    vx + 100,
    vy + (x_shock_range[50] - vx) * np.tan(beta_rad) + 15,
    f"β = {beta_deg:.2f}°",
    fontsize=12,
    color="red",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    fontweight="bold",
)

ax.set_xlabel("x [pixels]", fontsize=13)
ax.set_ylabel("y [pixels]", fontsize=13)
ax.set_title(
    f"Theoretical Shock Overlay on Experimental Shadowgraph (Frame {middle_frame})",
    fontsize=14,
    fontweight="bold",
)
ax.legend(fontsize=11, loc="upper left", framealpha=0.9)
ax.set_xlim(0, img_array.shape[1])
ax.set_ylim(img_array.shape[0], 0)
plt.tight_layout()
plt.savefig(OUTPUT_FIGS_DIR / "theoretical-shock.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: theoretical-shock.png (updated with shadowgraph background)")

# Figure 5: Drag coefficient comparison bar chart
fig, ax = plt.subplots(figsize=(8, 6))

categories = ["Experimental\n(Total)", "Inviscid\n(Theory)", "Viscous\n(Estimated)"]
values = [CD_experimental, CD_inviscid, CD_viscous]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor="black")

# Add value labels on bars
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{val:.4f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

ax.set_ylabel("Drag Coefficient", fontsize=12)
ax.set_title("Drag Coefficient Comparison", fontsize=14)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_FIGS_DIR / "drag_coefficient_comparison.png", dpi=300)
plt.close()
print("  Saved: drag_coefficient_comparison.png")

# Figure 6: Cone angle variation (quality check)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(geom_df["time"] * 1000, geom_df["cone_angle_deg"], "o-", markersize=3)
ax.axhline(
    y=median_cone_angle_deg,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Median: {median_cone_angle_deg:.3f}°",
)
ax.set_xlabel("Time [ms]", fontsize=12)
ax.set_ylabel("Measured Cone Half-Angle [°]", fontsize=12)
ax.set_title("Cone Half-Angle Over Time (Measurement Consistency)", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_FIGS_DIR / "cone_angle_consistency.png", dpi=300)
plt.close()
print("  Saved: cone_angle_consistency.png")

print()

# --- Output ------------------------------------------------------------------

print("9. Exporting Data to CSV")
print("-" * 80)

# Table 1: Freestream conditions
freestream_data = {
    "Parameter": [
        "M_t",
        "p_t [Pa]",
        "T_t [K]",
        "p_0 [Pa]",
        "T_0 [K]",
        "M_∞ [experimental]",
        "p_∞ [Pa]",
        "T_∞ [K]",
        "ρ_∞ [kg/m³]",
        "a_∞ [m/s]",
        "V_∞ [m/s]",
    ],
    "Value": [
        Mt,
        pt,
        Tt,
        p0,
        T0,
        M_EXPERIMENTAL,
        freestream["p"],
        freestream["T"],
        freestream["rho"],
        freestream["a"],
        freestream["V"],
    ],
}
df_freestream = pd.DataFrame(freestream_data)
df_freestream.to_csv(OUTPUT_TEXT_DIR / "freestream_conditions.csv", index=False)
print("  Saved: freestream_conditions.csv")

# Table 2: Cone geometry
cone_geom_data = {
    "Parameter": [
        "Half-angle [rad]",
        "Half-angle [°]",
        "Base radius [m]",
        "Base radius [mm]",
        "Base area [m²]",
    ],
    "Value": [
        THETA_C,
        np.degrees(THETA_C),
        R_BASE,
        R_BASE * 1000,
        A_BASE,
    ],
}
df_cone_geom = pd.DataFrame(cone_geom_data)
df_cone_geom.to_csv(OUTPUT_TEXT_DIR / "cone_geometry.csv", index=False)
print("  Saved: cone_geometry.csv")

# Table 3: Drag analysis summary
drag_summary_data = {
    "Parameter": [
        "Quadratic fit a0 [m]",
        "Quadratic fit a1 [m/s]",
        "Quadratic fit a2 [m/s²]",
        "Experimental acceleration [m/s²]",
        "Experimental drag force [N]",
        "Experimental drag coefficient",
        "Rasmussen Cp (inviscid)",
        "Inviscid drag force [N]",
        "Inviscid drag coefficient",
        "Viscous drag force [N]",
        "Viscous drag coefficient",
        "Viscous fraction [%]",
    ],
    "Value": [
        a0,
        a1,
        a2,
        ax_experimental,
        drag_force_experimental,
        CD_experimental,
        Cp_theory,
        drag_force_inviscid,
        CD_inviscid,
        drag_force_viscous,
        CD_viscous,
        viscous_fraction,
    ],
}
df_drag = pd.DataFrame(drag_summary_data)
df_drag.to_csv(OUTPUT_TEXT_DIR / "drag_analysis.csv", index=False)
print("  Saved: drag_analysis.csv")

# Table 4: Shock angle
shock_data = {
    "Parameter": [
        "Cone half-angle [rad]",
        "Cone half-angle [°]",
        "Shock angle [rad]",
        "Shock angle [°]",
    ],
    "Value": [THETA_C, np.degrees(THETA_C), beta_rad, beta_deg],
}
df_shock = pd.DataFrame(shock_data)
df_shock.to_csv(OUTPUT_TEXT_DIR / "shock_angle.csv", index=False)
print("  Saved: shock_angle.csv")

# Table 5: Full trajectory data
geom_df.to_csv(OUTPUT_TEXT_DIR / "trajectory_data.csv", index=False)
print("  Saved: trajectory_data.csv")

print()
print("=" * 80)
print("Analysis Complete!")
print("=" * 80)

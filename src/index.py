"""
ENAE464 Lab05: Particle Imaging Velocimetry (PIV) Analysis
Analysis of vortex shedding downstream of a circular cylinder in water flow
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy.optimize import curve_fit

# ── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR = Path("./data/test_2p1hz_20ms")
OUTPUT_TEXT_DIR = Path("./outputs/text")
OUTPUT_FIGS_DIR = Path("./outputs/figures")

OUTPUT_TEXT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────

# Experiment parameters
PUMP_FREQ = 2.1  # Hz (from folder name test_2p1hz_20ms)
CYLINDER_DIAMETER = 0.0127  # m (0.5 inches)
TOTAL_DURATION = 0.38  # seconds
NUM_FRAMES = 20  # number of velocity field snapshots

# Particle properties
PARTICLE_DENSITY = 950  # kg/m³ (polyamide particles)
PARTICLE_DIAMETER = 20e-6  # m (20 microns)

# Fluid properties (water at room temperature)
WATER_DENSITY = 998  # kg/m³
WATER_VISCOSITY = 4.04e-4  # kg/(m·s) dynamic viscosity
WATER_KINEMATIC_VISCOSITY = WATER_VISCOSITY / WATER_DENSITY  # m²/s

# Flow velocity calculation
U_INF = 0.0185 * PUMP_FREQ - 0.0038  # m/s

# Strouhal number for circular cylinder (typical value for Re ~ 1000-10000)
STROUHAL_NUMBER = 0.21  # dimensionless (from literature)

print("=" * 80)
print("ENAE464 Lab05: Particle Imaging Velocimetry Analysis")
print("=" * 80)
print()

# ── Load and Process PIV Data ────────────────────────────────────────────────

print("Loading PIV velocity field data...")
print(f"  Data directory: {DATA_DIR}")
print(f"  Number of frames: {NUM_FRAMES}")
print()

# Get list of velocity data files
vb_files = sorted(DATA_DIR.glob("VB*.txt"))
print(f"Found {len(vb_files)} velocity field files")

# Time array for each frame
time_array = np.linspace(0, TOTAL_DURATION, NUM_FRAMES)

# Storage for velocity fields
all_x = []
all_y = []
all_u = []
all_v = []

for i, vb_file in enumerate(vb_files):
    # Load the raw data file
    data = np.loadtxt(vb_file)

    # Reshape according to the MATLAB pattern provided in lab instructions
    # Data format: rows 1-120 = x, 121-240 = y, 241-360 = u, 361-480 = v
    # Each section has 120 rows × 67 columns = 8040 points
    x_data = data[0:120, :].reshape(-1)  # Flatten to 1D array
    y_data = data[120:240, :].reshape(-1)
    u_data = data[240:360, :].reshape(-1)
    v_data = data[360:480, :].reshape(-1)

    all_x.append(x_data)
    all_y.append(y_data)
    all_u.append(u_data)
    all_v.append(v_data)

# Convert to numpy arrays
all_x = np.array(all_x)  # Shape: (num_frames, num_points)
all_y = np.array(all_y)
all_u = np.array(all_u)
all_v = np.array(all_v)

print(f"  Data shape: {all_x.shape} (frames × spatial points)")
print(f"  Time range: 0 to {TOTAL_DURATION:.3f} seconds")
print()

# ── Results 1: Flow Velocity ──────────────────────────────────────────────────

print("=" * 80)
print("RESULTS 1: Flow Velocity")
print("=" * 80)

print(f"Pump frequency: {PUMP_FREQ} Hz")
print(f"Incoming flow velocity: U∞ = 0.0185 × {PUMP_FREQ} - 0.0038")
print(f"                        U∞ = {U_INF:.4f} m/s")
print()

# Calculate mean velocities from PIV data
mean_u = np.mean(all_u)
mean_v = np.mean(all_v)
mean_velocity_magnitude = np.sqrt(mean_u**2 + mean_v**2)

print("PIV-measured mean velocities:")
print(f"  Mean u-velocity: {mean_u:.4f} m/s")
print(f"  Mean v-velocity: {mean_v:.4f} m/s")
print(f"  Mean velocity magnitude: {mean_velocity_magnitude:.4f} m/s")
print()

# Save flow velocity results
flow_velocity_results = pd.DataFrame(
    {
        "Parameter": [
            "Pump Frequency [Hz]",
            "Incoming Flow Velocity U∞ [m/s]",
            "Mean u-velocity (PIV) [m/s]",
            "Mean v-velocity (PIV) [m/s]",
            "Mean Velocity Magnitude (PIV) [m/s]",
        ],
        "Value": [
            f"{PUMP_FREQ:.2f}",
            f"{U_INF:.5f}",
            f"{mean_u:.5f}",
            f"{mean_v:.5f}",
            f"{mean_velocity_magnitude:.5f}",
        ],
    }
)

flow_velocity_results.to_csv(OUTPUT_TEXT_DIR / "flow_velocity.csv", index=False)
print(f"Saved flow velocity results → {OUTPUT_TEXT_DIR / 'flow_velocity.csv'}")
print()

# ── Results 2: Reynolds Number ────────────────────────────────────────────────

print("=" * 80)
print("RESULTS 2: Reynolds Number")
print("=" * 80)

reynolds_number = (U_INF * CYLINDER_DIAMETER) / WATER_KINEMATIC_VISCOSITY

print("Reynolds number calculation:")
print("  Re = (U∞ × D) / ν")
print(
    f"  Re = ({U_INF:.4f} × {CYLINDER_DIAMETER:.4f}) / {WATER_KINEMATIC_VISCOSITY:.6e}"
)
print(f"  Re = {reynolds_number:.1f}")
print()

# Save Reynolds number results
reynolds_results = pd.DataFrame(
    {
        "Parameter": [
            "Flow Velocity U∞ [m/s]",
            "Cylinder Diameter D [m]",
            "Water Density [kg/m³]",
            "Water Dynamic Viscosity [kg/(m·s)]",
            "Water Kinematic Viscosity [m²/s]",
            "Reynolds Number [-]",
        ],
        "Value": [
            f"{U_INF:.5f}",
            f"{CYLINDER_DIAMETER:.4f}",
            f"{WATER_DENSITY:.1f}",
            f"{WATER_VISCOSITY:.6e}",
            f"{WATER_KINEMATIC_VISCOSITY:.6e}",
            f"{reynolds_number:.1f}",
        ],
    }
)

reynolds_results.to_csv(OUTPUT_TEXT_DIR / "reynolds_number.csv", index=False)
print(f"Saved Reynolds number results → {OUTPUT_TEXT_DIR / 'reynolds_number.csv'}")
print()

# ── Results 3: Velocity Fluctuation at (x,y) ≈ (0,0) ─────────────────────────

print("=" * 80)
print("RESULTS 3: Velocity Fluctuation at (x,y) ≈ (0,0)")
print("=" * 80)

# Find the point closest to (x,y) = (0,0)
# Use first frame to find spatial location
x_first = all_x[0, :]
y_first = all_y[0, :]

# Distance from origin
distance_from_origin = np.sqrt(x_first**2 + y_first**2)
origin_idx = np.argmin(distance_from_origin)

print("Point closest to origin:")
print(f"  Index: {origin_idx}")
print(f"  Position: (x={x_first[origin_idx]:.4f}, y={y_first[origin_idx]:.4f}) cm")
print(f"  Distance from (0,0): {distance_from_origin[origin_idx]:.4f} cm")
print()

# Extract v-velocity time series at this point
v_at_origin = all_v[:, origin_idx]
u_at_origin = all_u[:, origin_idx]

print("V-velocity time series at near-origin point:")
for i, (t, v) in enumerate(zip(time_array, v_at_origin)):
    if i < 5 or i >= NUM_FRAMES - 2:
        print(f"  t = {t:.4f} s, v = {v:.4f} m/s")
    elif i == 5:
        print("  ...")
print()

# Calculate v-velocity fluctuation (subtract mean)
v_mean_at_origin = np.mean(v_at_origin)
v_fluctuation = v_at_origin - v_mean_at_origin

# Save velocity fluctuation data
velocity_fluctuation_data = pd.DataFrame(
    {
        "Time [s]": time_array,
        "v-velocity [m/s]": v_at_origin,
        "v-fluctuation [m/s]": v_fluctuation,
        "u-velocity [m/s]": u_at_origin,
    }
)

velocity_fluctuation_data.to_csv(
    OUTPUT_TEXT_DIR / "velocity_fluctuation.csv", index=False
)
print(
    f"Saved velocity fluctuation data → {OUTPUT_TEXT_DIR / 'velocity_fluctuation.csv'}"
)
print()

# ── Results 4: Expected Shedding Frequency & Strouhal Number ──────────────────

print("=" * 80)
print("RESULTS 4: Expected Shedding Frequency & Strouhal Number")
print("=" * 80)

print(f"Strouhal number for circular cylinder: St = {STROUHAL_NUMBER}")
print(f"  (Typical value for Re ≈ {reynolds_number:.0f})")
print()

# Calculate expected shedding frequency
# St = f × D / U∞
# Therefore: f = St × U∞ / D
f_expected = STROUHAL_NUMBER * U_INF / CYLINDER_DIAMETER

print("Expected shedding frequency:")
print("  f = St × U∞ / D")
print(f"  f = {STROUHAL_NUMBER} × {U_INF:.4f} / {CYLINDER_DIAMETER:.4f}")
print(f"  f = {f_expected:.2f} Hz")
print()

# Save expected frequency results
expected_frequency_results = pd.DataFrame(
    {
        "Parameter": [
            "Strouhal Number [-]",
            "Flow Velocity U∞ [m/s]",
            "Cylinder Diameter D [m]",
            "Expected Shedding Frequency [Hz]",
            "Expected Shedding Period [s]",
        ],
        "Value": [
            f"{STROUHAL_NUMBER:.2f}",
            f"{U_INF:.5f}",
            f"{CYLINDER_DIAMETER:.4f}",
            f"{f_expected:.3f}",
            f"{1 / f_expected:.4f}",
        ],
    }
)

expected_frequency_results.to_csv(
    OUTPUT_TEXT_DIR / "expected_shedding_frequency.csv", index=False
)
print(
    f"Saved expected frequency results → {OUTPUT_TEXT_DIR / 'expected_shedding_frequency.csv'}"
)
print()

# ── Results 5: Stokes Number ──────────────────────────────────────────────────

print("=" * 80)
print("RESULTS 5: Stokes Number for Particle Seeding")
print("=" * 80)

# Particle reaction time: τ_r = ρ_p × D_p² / (18 × μ)
particle_reaction_time = (
    PARTICLE_DENSITY * PARTICLE_DIAMETER**2 / (18 * WATER_VISCOSITY)
)

print("Particle reaction time:")
print("  τ_r = ρ_p × D_p² / (18 × μ)")
print(
    f"  τ_r = {PARTICLE_DENSITY} × ({PARTICLE_DIAMETER:.2e})² / (18 × {WATER_VISCOSITY:.2e})"
)
print(f"  τ_r = {particle_reaction_time:.6e} s")
print()

# Characteristic flow time scale: τ_f = 1 / f_shedding
flow_time_scale = 1 / f_expected

print("Characteristic flow time scale:")
print("  τ_f = 1 / f_shedding")
print(f"  τ_f = 1 / {f_expected:.3f}")
print(f"  τ_f = {flow_time_scale:.4f} s")
print()

# Stokes number: St_k = τ_r / τ_f
stokes_number = particle_reaction_time / flow_time_scale

print("Stokes number:")
print("  St_k = τ_r / τ_f")
print(f"  St_k = {particle_reaction_time:.6e} / {flow_time_scale:.4f}")
print(f"  St_k = {stokes_number:.6e}")
print()

if stokes_number < 0.1:
    print("✓ St_k << 1: Particles faithfully follow the flow")
else:
    print("⚠ St_k not << 1: Particles may not perfectly follow the flow")
print()

# Save Stokes number results
stokes_results = pd.DataFrame(
    {
        "Parameter": [
            "Particle Density [kg/m³]",
            "Particle Diameter [m]",
            "Water Dynamic Viscosity [kg/(m·s)]",
            "Particle Reaction Time τ_r [s]",
            "Shedding Frequency [Hz]",
            "Flow Time Scale τ_f [s]",
            "Stokes Number St_k [-]",
        ],
        "Value": [
            f"{PARTICLE_DENSITY:.0f}",
            f"{PARTICLE_DIAMETER:.2e}",
            f"{WATER_VISCOSITY:.2e}",
            f"{particle_reaction_time:.6e}",
            f"{f_expected:.3f}",
            f"{flow_time_scale:.4f}",
            f"{stokes_number:.6e}",
        ],
    }
)

stokes_results.to_csv(OUTPUT_TEXT_DIR / "stokes_number.csv", index=False)
print(f"Saved Stokes number results → {OUTPUT_TEXT_DIR / 'stokes_number.csv'}")
print()

# ── Analysis 1: Measured Shedding Frequency ───────────────────────────────────

print("=" * 80)
print("ANALYSIS 1: Measured Shedding Frequency from Sinusoidal Fit")
print("=" * 80)


# Define sinusoidal function for curve fitting
def sinusoid(t, amplitude, frequency, phase, offset):
    """Sinusoidal function: A × sin(2π × f × t + φ) + C"""
    return amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset


# Initial guess for curve fitting parameters
# Start with expected frequency
p0_amplitude = np.std(v_fluctuation) * np.sqrt(2)  # Estimate amplitude
p0_frequency = f_expected  # Use expected frequency as initial guess
p0_phase = 0  # Initial phase guess
p0_offset = v_mean_at_origin  # Mean value

print("Fitting sinusoid to v-velocity fluctuation...")
print(f"  Initial guess for frequency: {p0_frequency:.2f} Hz")
print()

try:
    # Perform curve fitting
    popt, pcov = curve_fit(
        sinusoid,
        time_array,
        v_at_origin,
        p0=[p0_amplitude, p0_frequency, p0_phase, p0_offset],
        maxfev=10000,
    )

    amplitude_fit, frequency_fit, phase_fit, offset_fit = popt

    # Calculate standard errors
    perr = np.sqrt(np.diag(pcov))

    print("Fitted parameters:")
    print(f"  Amplitude: {amplitude_fit:.5f} ± {perr[0]:.5f} m/s")
    print(f"  Frequency: {frequency_fit:.3f} ± {perr[1]:.3f} Hz")
    print(f"  Phase: {phase_fit:.3f} ± {perr[2]:.3f} rad")
    print(f"  Offset: {offset_fit:.5f} ± {perr[3]:.5f} m/s")
    print()

    # Calculate fitted curve
    v_fitted = sinusoid(time_array, *popt)

    # Calculate R² value
    ss_res = np.sum((v_at_origin - v_fitted) ** 2)
    ss_tot = np.sum((v_at_origin - np.mean(v_at_origin)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    print(f"Goodness of fit: R² = {r_squared:.4f}")
    print()

    # Measured shedding frequency
    f_measured = frequency_fit

except Exception as e:
    print(f"Warning: Curve fitting failed ({e})")
    print("Using FFT-based frequency estimation as fallback...")

    # Fallback: Use FFT to estimate dominant frequency
    from scipy.fft import fft, fftfreq

    # Perform FFT on v-velocity fluctuation
    N = len(v_fluctuation)
    yf = fft(v_fluctuation - np.mean(v_fluctuation))
    xf = fftfreq(N, TOTAL_DURATION / N)

    # Find dominant frequency (positive frequencies only)
    positive_freq_idx = xf > 0
    dominant_idx = np.argmax(np.abs(yf[positive_freq_idx]))
    f_measured = xf[positive_freq_idx][dominant_idx]

    amplitude_fit, phase_fit, offset_fit = None, None, None
    v_fitted = None
    r_squared = None

    print(f"  Dominant frequency from FFT: {f_measured:.3f} Hz")
    print()

# Save measured frequency results
measured_frequency_results = pd.DataFrame(
    {
        "Parameter": [
            "Measured Shedding Frequency [Hz]",
            "Fitted Amplitude [m/s]" if amplitude_fit is not None else "N/A",
            "Fitted Phase [rad]" if phase_fit is not None else "N/A",
            "Fitted Offset [m/s]" if offset_fit is not None else "N/A",
            "R² (goodness of fit)" if r_squared is not None else "N/A",
        ],
        "Value": [
            f"{f_measured:.4f}",
            f"{amplitude_fit:.5f}" if amplitude_fit is not None else "N/A",
            f"{phase_fit:.3f}" if phase_fit is not None else "N/A",
            f"{offset_fit:.5f}" if offset_fit is not None else "N/A",
            f"{r_squared:.4f}" if r_squared is not None else "N/A",
        ],
    }
)

measured_frequency_results.to_csv(
    OUTPUT_TEXT_DIR / "measured_shedding_frequency.csv", index=False
)
print(
    f"Saved measured frequency results → {OUTPUT_TEXT_DIR / 'measured_shedding_frequency.csv'}"
)
print()

# ── Analysis 2: Comparison to Theory ──────────────────────────────────────────

print("=" * 80)
print("ANALYSIS 2: Comparison of Measured vs. Expected Shedding Frequency")
print("=" * 80)

percent_error = abs(f_measured - f_expected) / f_expected * 100

print(f"Expected shedding frequency: {f_expected:.3f} Hz")
print(f"Measured shedding frequency: {f_measured:.3f} Hz")
print(f"Absolute difference: {abs(f_measured - f_expected):.3f} Hz")
print(f"Percent error: {percent_error:.2f}%")
print()

if percent_error < 10:
    print("✓ Excellent agreement (< 10% error)")
elif percent_error < 30:
    print("✓ Good agreement (< 30% error)")
else:
    print("⚠ Significant discrepancy (> 30% error)")
print()

# Save comparison results
comparison_results = pd.DataFrame(
    {
        "Metric": [
            "Expected Shedding Frequency [Hz]",
            "Measured Shedding Frequency [Hz]",
            "Absolute Difference [Hz]",
            "Percent Error [%]",
        ],
        "Value": [
            f"{f_expected:.4f}",
            f"{f_measured:.4f}",
            f"{abs(f_measured - f_expected):.4f}",
            f"{percent_error:.2f}",
        ],
    }
)

comparison_results.to_csv(OUTPUT_TEXT_DIR / "frequency_comparison.csv", index=False)
print(f"Saved comparison results → {OUTPUT_TEXT_DIR / 'frequency_comparison.csv'}")
print()

# ── Visualization 1: Velocity Fluctuation Plot ────────────────────────────────

print("Generating velocity fluctuation plot...")

fig, ax = plt.subplots(figsize=(10, 6))

# Plot measured data
ax.plot(
    time_array,
    v_at_origin,
    "o",
    markersize=8,
    color="steelblue",
    label="Measured v-velocity",
    zorder=3,
)

# Plot fitted sinusoid if available
if v_fitted is not None:
    # Create smoother curve for visualization
    t_smooth = np.linspace(0, TOTAL_DURATION, 200)
    v_smooth = sinusoid(t_smooth, amplitude_fit, frequency_fit, phase_fit, offset_fit)

    ax.plot(
        t_smooth,
        v_smooth,
        "-",
        linewidth=2.5,
        color="tomato",
        label=f"Fitted sinusoid (f = {frequency_fit:.2f} Hz)",
        zorder=2,
    )

# Add mean line
ax.axhline(
    v_mean_at_origin,
    color="gray",
    linestyle="--",
    linewidth=1.5,
    alpha=0.7,
    label=f"Mean v-velocity = {v_mean_at_origin:.4f} m/s",
    zorder=1,
)

ax.set_xlabel("Time [s]", fontsize=12)
ax.set_ylabel("v-velocity [m/s]", fontsize=12)
ax.set_title(
    f"Transverse Velocity Fluctuation at (x,y) ≈ (0,0)\n"
    f"Measured Frequency: {f_measured:.2f} Hz, Expected: {f_expected:.2f} Hz",
    fontsize=13,
    fontweight="bold",
)

ax.grid(which="major", linestyle="--", alpha=0.4)
ax.grid(which="minor", linestyle=":", alpha=0.2)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.legend(loc="best", fontsize=10)

fig.tight_layout()
fig.savefig(OUTPUT_FIGS_DIR / "velocity_fluctuation.png", dpi=300)
plt.close(fig)

print(
    f"Saved velocity fluctuation plot → {OUTPUT_FIGS_DIR / 'velocity_fluctuation.png'}"
)
print()

# ── Visualization 2: Sample Velocity Vector Field ─────────────────────────────

print("Generating sample velocity vector field plot...")

# Use the middle frame for visualization
frame_idx = NUM_FRAMES // 2

# Get spatial grid (reshape back to 2D grid)
# Original data is 120 rows × 67 columns
x_grid = all_x[frame_idx, :].reshape(120, 67)
y_grid = all_y[frame_idx, :].reshape(120, 67)
u_grid = all_u[frame_idx, :].reshape(120, 67)
v_grid = all_v[frame_idx, :].reshape(120, 67)

# Downsample for clearer visualization (plot every 4th point)
skip = 4
x_plot = x_grid[::skip, ::skip]
y_plot = y_grid[::skip, ::skip]
u_plot = u_grid[::skip, ::skip]
v_plot = v_grid[::skip, ::skip]

# Calculate velocity magnitude
velocity_magnitude = np.sqrt(u_grid**2 + v_grid**2)

fig, ax = plt.subplots(figsize=(12, 8))

# Plot velocity magnitude as contour
contour = ax.contourf(
    x_grid, y_grid, velocity_magnitude, levels=20, cmap="viridis", alpha=0.7
)

# Plot velocity vectors
quiver = ax.quiver(
    x_plot,
    y_plot,
    u_plot,
    v_plot,
    scale=1.5,
    scale_units="inches",
    width=0.003,
    color="white",
    edgecolor="black",
    linewidth=0.5,
    alpha=0.8,
)

# Add colorbar
cbar = fig.colorbar(contour, ax=ax, label="Velocity Magnitude [m/s]")

# Mark the origin
ax.plot(0, 0, "r*", markersize=15, label="Origin (0,0)", zorder=10)

ax.set_xlabel("x-position [cm]", fontsize=12)
ax.set_ylabel("y-position [cm]", fontsize=12)
ax.set_title(
    f"Velocity Vector Field at t = {time_array[frame_idx]:.3f} s\n"
    f"(Frame {frame_idx + 1}/{NUM_FRAMES})",
    fontsize=13,
    fontweight="bold",
)
ax.set_aspect("equal")
ax.legend(loc="upper right", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.3)

fig.tight_layout()
fig.savefig(OUTPUT_FIGS_DIR / "velocity_vector_field.png", dpi=300)
plt.close(fig)

print(
    f"Saved velocity vector field plot → {OUTPUT_FIGS_DIR / 'velocity_vector_field.png'}"
)
print()

# ── Visualization 3: Velocity Magnitude Contour Over Time ─────────────────────

print("Generating velocity magnitude time evolution plot...")

# Create a figure with multiple subplots showing velocity field evolution
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Select 6 evenly-spaced frames to visualize
frame_indices = np.linspace(0, NUM_FRAMES - 1, 6, dtype=int)

for idx, frame_idx in enumerate(frame_indices):
    ax = axes[idx]

    # Reshape data for this frame
    x_grid = all_x[frame_idx, :].reshape(120, 67)
    y_grid = all_y[frame_idx, :].reshape(120, 67)
    u_grid = all_u[frame_idx, :].reshape(120, 67)
    v_grid = all_v[frame_idx, :].reshape(120, 67)
    velocity_magnitude = np.sqrt(u_grid**2 + v_grid**2)

    # Plot contour
    contour = ax.contourf(x_grid, y_grid, velocity_magnitude, levels=15, cmap="jet")

    # Mark origin
    ax.plot(0, 0, "k*", markersize=10)

    ax.set_xlabel("x [cm]", fontsize=10)
    ax.set_ylabel("y [cm]", fontsize=10)
    ax.set_title(f"t = {time_array[frame_idx]:.3f} s", fontsize=11)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.3)

# Add shared colorbar
# reserve space and position manually
fig.tight_layout(
    rect=[0, 0, 0.88, 0.96]
)  # leave room on right for colorbar, top for suptitle
cbar_ax = fig.add_axes([0.90, 0.08, 0.02, 0.84])  # [left, bottom, width, height]
fig.colorbar(contour, cax=cbar_ax, label="Velocity Magnitude [m/s]")

fig.suptitle(
    "Velocity Field Evolution: Vortex Shedding Behind Circular Cylinder",
    fontsize=14,
    fontweight="bold",
    y=0.995,
)

fig.savefig(OUTPUT_FIGS_DIR / "velocity_field_evolution.png", dpi=300)
plt.close(fig)

print(
    f"Saved velocity field evolution plot → {OUTPUT_FIGS_DIR / 'velocity_field_evolution.png'}"
)
print()

# ── Summary Statistics ────────────────────────────────────────────────────────

print("=" * 80)
print("SUMMARY OF ALL RESULTS")
print("=" * 80)
print()

summary_df = pd.DataFrame(
    {
        "Parameter": [
            "Pump Frequency",
            "Incoming Flow Velocity U∞",
            "Reynolds Number",
            "Strouhal Number",
            "Expected Shedding Frequency",
            "Measured Shedding Frequency",
            "Frequency Error",
            "Particle Reaction Time",
            "Flow Time Scale",
            "Stokes Number",
        ],
        "Value": [
            f"{PUMP_FREQ} Hz",
            f"{U_INF:.4f} m/s",
            f"{reynolds_number:.1f}",
            f"{STROUHAL_NUMBER}",
            f"{f_expected:.3f} Hz",
            f"{f_measured:.3f} Hz",
            f"{percent_error:.2f}%",
            f"{particle_reaction_time:.4e} s",
            f"{flow_time_scale:.4f} s",
            f"{stokes_number:.4e}",
        ],
    }
)

print(summary_df.to_string(index=False))
print()

summary_df.to_csv(OUTPUT_TEXT_DIR / "summary.csv", index=False)
print(f"Saved summary → {OUTPUT_TEXT_DIR / 'summary.csv'}")
print()

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print()
print("All outputs saved to:")
print(f"  Figures: {OUTPUT_FIGS_DIR}")
print(f"  Data:    {OUTPUT_TEXT_DIR}")
print()

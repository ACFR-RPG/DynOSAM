import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Data (same as before)
# ============================================================

# plt.rcdefaults()

from dynosam_utils.evaluation.core.plotting import startup_plotting
plt.rcdefaults()
startup_plotting(font_size=24, text_size_scalar=0.9, line_width=3.0)

import seaborn as sns
sns.set_style("whitegrid")

# plt.rcdefaults()



sequences = [
    "K00","K01","K02","K03","K04","K05","K06","K18","K20",
    "OMD",
    "L1","L2","S1","S2",
    "CD_M","CD_H","CN_M","CN_H","PL_M","PL_H",
    "IV","V","VI","VII"
]

parallel_hybrid = np.array([
    247,232,250,333,133,332,420,397,1252,
    494,
    453,486,124,78,
    1332,881,1164,1046,791,838,
    426,276,149,75
], dtype=float)

ihybrid_10 = np.array([
    601,611,648,665,733,550,1013,np.nan,np.nan,
    np.nan,
    1268,930,322,244,
    np.nan,2536,np.nan,2881,np.nan,1947,
    np.nan,760,360,142
], dtype=float)

ihybrid_1 = np.array([
    4771,1683,2146,2202,2287,2931,1527,8359,np.nan,
    np.nan,
    2105,1823,862,589,
    15822,16262,np.nan,9325,3648,4085,
    3286,1673,663,476
], dtype=float)

ibaseline_10 = np.array([
    np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,
    np.nan,
    np.nan,1119,330,361,
    np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,
    np.nan,np.nan,np.nan,146
], dtype=float)

ibaseline_1 = np.array([
    2632,2340,np.nan,1502,3252,3130,np.nan,7166,np.nan,
    np.nan,
    3100,2158,868,518,
    np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,
    np.nan,1031,835,342
], dtype=float)

# ============================================================
# Style: consistent color per method
# ============================================================

colors = {
    "parallel": "#1f77b4",   # blue
    "ihybrid":  "#ff7f0e",   # orange
    "ibaseline":"#2ca02c"    # green
}

plt.rcParams.update({
    "font.size": 25,
    "axes.labelsize": 22,
    "axes.titlesize": 24,
    "legend.fontsize": 18,
    "xtick.labelsize": 15,
    # "ytick.labelsize": 14,
    "font.family": "serif"
})

fig, ax = plt.subplots(figsize=(18, 7))
x = np.arange(len(sequences))

groups = [
    ("KITTI", 0, 8),
    ("OMD", 9, 9),
    ("Outdoor Cluster", 10, 13),
    ("VIODE", 14, 19),
    ("TartanAir", 20, 23)
]

# ============================================================
# Draw underbraces in axis coordinates (clean + stable)
# ============================================================
import matplotlib.transforms as transforms

def partial_circle(x_center, y_center,
                   radius_x, radius_y,
                   start_deg, end_deg,
                   n=100):
    t = np.linspace(start_deg/180*np.pi, end_deg/180*np.pi, n)
    return x_center + radius_x * np.cos(t), y_center + radius_y * np.sin(t)

def underbrace(ax,
               x_start, x_end,
               y=-0.15,
               label=None,
               radius_y=0.02,
               radius_x=0.1,
               lw=1.0,
               color="black",
               text_offset=-0.04,
               **text_kwargs):

    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

    x_mid = 0.5 * (x_start + x_end)

    y_top = y + radius_y
    y_mid = y
    y_bottom = y - radius_y

    # left arc
    X_circ_left, Y_circ_left = partial_circle(
        x_start + radius_x, y_top,
        radius_x, radius_y,
        180, 270
    )

    X_line_left = np.linspace(x_start + radius_x, x_mid - radius_x)
    Y_line_left = np.full_like(X_line_left, y_mid)

    # middle kink
    X_circ_mid_left, Y_circ_mid_left = partial_circle(
        x_mid - radius_x, y_bottom,
        radius_x, radius_y,
        90, 0
    )

    X_circ_mid_right, Y_circ_mid_right = partial_circle(
        x_mid + radius_x, y_bottom,
        radius_x, radius_y,
        180, 90
    )

    # right side
    X_line_right = np.linspace(x_mid + radius_x, x_end - radius_x)
    Y_line_right = np.full_like(X_line_right, y_mid)

    X_circ_right, Y_circ_right = partial_circle(
        x_end - radius_x, y_top,
        radius_x, radius_y,
        270, 360
    )

    Xs = np.concatenate([
        X_circ_left, X_line_left,
        X_circ_mid_left, X_circ_mid_right,
        X_line_right, X_circ_right
    ])

    Ys = np.concatenate([
        Y_circ_left, Y_line_left,
        Y_circ_mid_left, Y_circ_mid_right,
        Y_line_right, Y_circ_right
    ])

    ax.plot(Xs, Ys, color=color, lw=lw, transform=trans, clip_on=False)

    if label is not None:
        ax.text(
            x_mid,
            y + text_offset,
            label,
            ha="center",
            va="top",
            transform=trans,
            color=color,
            **text_kwargs
        )

y_brace = -0.08   # below x-axis (in axes coords)
y_tick = -0.04

for name, start, end in groups:
    underbrace(
        ax,
        x_start=start - 0.4,
        x_end=end + 0.4,
        y=-0.12,              # below axis (adjust if needed)
        label=name,
        radius_x=0.25,
        radius_y=0.015,
        lw=1.2,
        color="black",
        fontsize=19
    )

# ============================================================
# Plot (color = method, linestyle = lambda)
# ============================================================

# Parallel-Hybrid (single)
ax.plot(
    x, parallel_hybrid,
    color=colors["parallel"],
    marker='o',
    linewidth=3.5,
    label="Parallel-Hybrid"
)

# iHybrid (same color, different λ styles)
ax.plot(
    x, ihybrid_10,
    color=colors["ihybrid"],
    linestyle='-',
    marker='s',
    linewidth=2.5,
    label=r"iHybrid ($\lambda_{rs}=10$)"
)

ax.plot(
    x, ihybrid_1,
    color=colors["ihybrid"],
    linestyle='--',
    marker='s',
    linewidth=2.5,
    label=r"iHybrid ($\lambda_{rs}=1$)"
)

# iBaseline (same color, different λ styles)
ax.plot(
    x, ibaseline_10,
    color=colors["ibaseline"],
    linestyle='-',
    marker='D',
    linewidth=2.5,
    label=r"iBaseline ($\lambda_{rs}=10$)"
)

ax.plot(
    x, ibaseline_1,
    color=colors["ibaseline"],
    linestyle='--',
    marker='D',
    linewidth=2.5,
    label=r"iBaseline ($\lambda_{rs}=1$)"
)

# ============================================================
# Axes formatting
# ============================================================

ax.set_yscale("log")
ax.set_ylabel("Average Update Time (ms)")
# ax.set_xlabel("Sequence")

ax.set_xticks(x)
ax.set_xticklabels(sequences, rotation=45)

ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.35)
ax.set_axisbelow(True)

# Dataset separators
for b in [9, 10, 14, 20]:
    ax.axvline(b - 0.5, linestyle='--', linewidth=1.0, alpha=0.3)



# Legend
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False)
# ax.legend(loc='upper center', ncol=3, frameon=False)
ax.legend()


# ax.set_title("Per-Sequence Incremental SLAM Runtime Comparison")
ax.set_title("Incremental Dynamic SLAM Runtime Comparison")


ax.grid(
    which="minor",
    axis="y",
    linestyle="--",
    linewidth=1.0,
    alpha=0.3,
    color="tab:gray",
    zorder = 5
)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.0)
    spine.set_color("black")

plt.tight_layout()
plt.show()


# file_path = f'/root/results/ral2025_timing.pdf'
# fig.savefig(file_path, format="pdf")

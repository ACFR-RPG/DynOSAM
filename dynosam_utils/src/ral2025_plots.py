import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import transforms
import numpy as np
import matplotlib

# =========================================================
# CONFIGURATION
# =========================================================

from dynosam_utils.evaluation.core.plotting import startup_plotting
plt.rcdefaults()
startup_plotting(font_size=20, text_size_scalar=0.9, line_width=3.0)

import seaborn as sns
sns.set_style("whitegrid")

CATEGORY_TO_PLOT = "camera"     # camera | object
PLOT_TYPE = "box"              # bar | box | normalized | line | mean

# ---------------------------------------------------------
# Aggregation
# ---------------------------------------------------------
AGGREGATE_PER_DATASET = True

# mean | median
AGGREGATION_FN = "mean"

# Show std region
SHOW_STD = True
STD_ALPHA = 0.20

sns.set_style("whitegrid")

# plt.rcParams.update({
#     "font.size": 14,
#     "axes.titlesize": 16,
#     "axes.labelsize": 14
# })

def partial_circle(x_center, y_center,
                   radius_x, radius_y,
                   start_deg, end_deg,
                   n=100):
    t = np.linspace(start_deg/180*np.pi, end_deg/180*np.pi, n)
    return x_center + radius_x * np.cos(t), y_center + radius_y * np.sin(t)


def underbrace(ax,
               x_start, x_end,       # horizontal position in data coords
               y=-0.15,              # vertical position in axes coords (0 = bottom, 1 = top)
               label=None,
               radius_y=0.02,        # vertical radius in axes coords
               radius_x=0.1,         # horizontal radius in data coords
               lw=1.0,
               color="black",
               text_offset=-0.04,    # extra offset for the label in axes coords
               **text_kwargs):

    # x in data coords, y in axes coords
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

    x_mid = 0.5 * (x_start + x_end)

    y_top = y + radius_y
    y_mid = y
    y_bottom = y - radius_y

    # left brace
    X_circ_left, Y_circ_left = partial_circle(x_start + radius_x, y_top, radius_x, radius_y, 180, 270)
    X_line_left = np.linspace(x_start + radius_x, x_mid - radius_x)
    Y_line_left = np.full_like(X_line_left, y_mid)
    # middle kink
    X_circ_mid_left,  Y_circ_mid_left  = partial_circle(x_mid - radius_x, y_bottom, radius_x, radius_y, 90, 0)
    X_circ_mid_right, Y_circ_mid_right = partial_circle(x_mid + radius_x, y_bottom, radius_x, radius_y, 180, 90)
    # right brace
    X_line_right = np.linspace(x_mid + radius_x, x_end - radius_x)
    Y_line_right = np.full_like(X_line_right, y_mid)
    X_circ_right, Y_circ_right = partial_circle(x_end - radius_x, y_top, radius_x, radius_y, 270, 360)

    Xs = np.concatenate([X_circ_left, X_line_left,
                         X_circ_mid_left, X_circ_mid_right,
                         X_line_right, X_circ_right])
    Ys = np.concatenate([Y_circ_left, Y_line_left,
                         Y_circ_mid_left, Y_circ_mid_right,
                         Y_line_right, Y_circ_right])

    ax.plot(Xs, Ys, color=color, lw=lw, transform=trans, clip_on=False)

    if label is not None:
        ax.text(x_mid, y + text_offset, label,
                ha="center", va="top",
                transform=trans, color=color, **text_kwargs)



# =========================================================
# DATA STORAGE
# =========================================================

rows = []

def add(category, dataset, metric, method, sequences, values, misc=None):

    for seq,val in zip(sequences, values):

        rows.append({
            "category": category,
            "dataset": dataset,
            "metric": metric,
            "method": method,
            "sequence": seq,
            "value": val,
            "misc": misc
        })


def get_dataset_ranges(rows, category="camera"):
    """
    Returns a dictionary mapping dataset names to their start and end sequences
    for a given category, based on the flat 'rows' list of dicts.

    Example output:
    {'KITTI': ['00','20'], 'OutdoorCluster': ['L1','S2'], 'OMD': ['S4U','S4U']}
    """
    dataset_ranges = {}

    # Filter rows by category
    filtered_rows = [r for r in rows if r["category"] == category]

    # Group sequences by dataset
    from collections import defaultdict
    dataset_seqs = defaultdict(set)
    for r in filtered_rows:
        dataset_seqs[r["dataset"]].add(r["sequence"])

    # Compute start and end sequence per dataset
    for dataset, seqs in dataset_seqs.items():
        # Sort numeric first, then alphanumeric
        seqs_sorted = sorted(seqs, key=lambda x: (not x.isdigit(), x))
        dataset_ranges[dataset] = [seqs_sorted[0], seqs_sorted[-1]]

    return dataset_ranges

def get_dataset_positions(rows, category="camera"):
    """
    Returns a dictionary mapping dataset names to start/end x-axis positions
    for a given category, based on the flat 'rows' list of dicts.

    If a dataset has only one sequence, its span is expanded by 1 on each side
    to make it visible on plots.

    Example output:
    {'KITTI': [0, 8], 'OutdoorCluster': [9, 12], 'OMD': [12, 13]}
    """
    dataset_ranges = get_dataset_ranges(rows, category)

    # Get sequences in order of appearance (unique)
    seq_order = []
    seen = set()
    for r in rows:
        if r["category"] == category and r["sequence"] not in seen:
            seq_order.append(r["sequence"])
            seen.add(r["sequence"])

    dataset_positions = {}
    n = len(seq_order)
    for dataset, (start_label, end_label) in dataset_ranges.items():
        try:
            start_idx = seq_order.index(start_label)
            end_idx = seq_order.index(end_label)

            # If start == end, expand span by 1 on each side if possible
            if start_idx == end_idx:
                start_idx = max(0, start_idx - 1)
                end_idx = min(n - 1, end_idx + 1)
        except ValueError:
            start_idx = end_idx = None

        dataset_positions[dataset] = [start_idx, end_idx]

    return dataset_positions


# =========================================================
# CAMERA RESULTS
# =========================================================

# ------------------------
# KITTI sequences
# ------------------------

# NOTE: all the NON-Baseline results are reported as relative to the Baseline

kitti_seq = ["00", "01", "02", "03", "04", "05", "06", "18", "20"]
outdoor_seq = ["L1", "L2", "S1", "S2"]
omd_seq = ["S4U"]


# --- OBJECT ERROR ---

# --- Metric: ME_r (Degrees) ---
# KITTI
add("object", "KITTI", "ME_r", "Baseline", kitti_seq, [1.11, 1.04, 0.97, 0.26, 1.24, 0.85, 0.39, 0.57, 0.52])
add("object", "KITTI", "ME_r", "Hybrid", kitti_seq, [-0.23, -0.06, 0.1, 0.01, 0.15, 0.3, 0.19, 0.27, -0.3])
add("object", "KITTI", "ME_r", "iHybrid", kitti_seq, [0.12, 0.08, -0.35, -0.03, 0.71, -0.10, 0.07, 0.24, None])
add("object", "KITTI", "ME_r", "Parallel-Hybrid", kitti_seq, [-0.06, -0.41, -1.29, -0.12, -0.23, -0.27, -0.09, -0.34, -0.14])

# Outdoor Cluster
add("object", "Outdoor Cluster", "ME_r", "Baseline", outdoor_seq, [0.82, 0.70, 0.69, 2.36])
add("object", "Outdoor Cluster", "ME_r", "Hybrid", outdoor_seq, [-0.34, 0.19, 0.16, 0.03])
add("object", "Outdoor Cluster", "ME_r", "iHybrid", outdoor_seq, [-0.46, -0.4, 0.3, 0.71])
add("object", "Outdoor Cluster", "ME_r", "Parallel-Hybrid", outdoor_seq, [-0.25, -0.18, -0.06, 0.3])

# OMD
add("object", "OMD", "ME_r", "Baseline", omd_seq, [0.67])
add("object", "OMD", "ME_r", "Hybrid", omd_seq, [0.08])
add("object", "OMD", "ME_r", "iHybrid", omd_seq, [None])
add("object", "OMD", "ME_r", "Parallel-Hybrid", omd_seq, [0.07])


# --- Metric: ME_t (Meters) ---
# KITTI
add("object", "KITTI", "ME_t", "Baseline", kitti_seq, [0.15, 0.32, 0.51, 0.11, 0.12, 0.27, 0.09, 0.11, 0.11])
add("object", "KITTI", "ME_t", "Hybrid", kitti_seq, [-0.08, -0.05, 0.0, 0.0, 0.0, -0.16, -0.02, 0.03, 0.0])
add("object", "KITTI", "ME_t", "iHybrid", kitti_seq, [-0.08, -0.08, 0.06, -0.01, -0.18, -0.16, -0.11, -0.01, None])
add("object", "KITTI", "ME_t", "Parallel-Hybrid", kitti_seq, [0.01, -0.06, 0.0, 0.0, -0.04, -0.09, 0.01, -0.04, 0.0])

# Outdoor Cluster
add("object", "Outdoor Cluster", "ME_t", "Baseline", outdoor_seq, [0.08, 0.06, 0.04, 0.15])
add("object", "Outdoor Cluster", "ME_t", "Hybrid", outdoor_seq, [-0.04, -0.04, 0.0, 0.11])
add("object", "Outdoor Cluster", "ME_t", "iHybrid", outdoor_seq, [-0.04, -0.05, 0.0, 0.09])
add("object", "Outdoor Cluster", "ME_t", "Parallel-Hybrid", outdoor_seq, [0.02, -0.01, -0.02, -0.04])

# OMD
add("object", "OMD", "ME_t", "Baseline", omd_seq, [0.02])
add("object", "OMD", "ME_t", "Hybrid", omd_seq, [0.0])
add("object", "OMD", "ME_t", "iHybrid", omd_seq, [None])
add("object", "OMD", "ME_t", "Parallel-Hybrid", omd_seq, [0.0])


# --- CAMERA ERROR ---

# --- Metric: ATE (Meters) ---
# KITTI
add("camera", "KITTI", "ATE", "Static Baseline", kitti_seq, [1.57, 2.10, 0.72, 1.67, 1.30, 1.99, 0.70, 2.15, 2.33])
add("camera", "KITTI", "ATE", "Baseline", kitti_seq, [1.54, 2.10, 0.74, 1.64, 1.28, 2.01, 0.41, 2.30, 2.30])
add("camera", "KITTI", "ATE", "Hybrid", kitti_seq, [0.0, 0.0, 0.0, -0.02, -0.01, 0.0, 0.0, 0.0, 0.0])
add("camera", "KITTI", "ATE", "iHybrid", kitti_seq, [-0.01, 0.0, 0.0, -0.19, -0.02, 0.01, -0.01, -0.06, None])
add("camera", "KITTI", "ATE", "Parallel-Hybrid", kitti_seq, [-0.01, 0.0, 0.0, -0.03, -0.04, 0.01, -0.02, -0.1, -0.12])

# Outdoor Cluster
add("camera", "Outdoor Cluster", "ATE", "Static Baseline", outdoor_seq, [0.60, 0.52, 0.12, 1.15])
add("camera", "Outdoor Cluster", "ATE", "Baseline", outdoor_seq, [0.62, 0.52, 0.09, 1.08])
add("camera", "Outdoor Cluster", "ATE", "Hybrid", outdoor_seq, [0.0, 0.0, -0.06, 0.81])
add("camera", "Outdoor Cluster", "ATE", "iHybrid", outdoor_seq, [-0.02, -0.03, -0.05, 0.46])
add("camera", "Outdoor Cluster", "ATE", "Parallel-Hybrid", outdoor_seq, [0.16, -0.23, -0.02, -0.3])

# OMD
add("camera", "OMD", "ATE", "Static Baseline", omd_seq, [0.12])
add("camera", "OMD", "ATE", "Baseline", omd_seq, [0.10])
add("camera", "OMD", "ATE", "Hybrid", omd_seq, [0.0])
add("camera", "OMD", "ATE", "iHybrid", omd_seq, [None])
add("camera", "OMD", "ATE", "Parallel-Hybrid", omd_seq, [0.0])


# --- Metric: RPE_r (Degrees) ---
# KITTI
add("camera", "KITTI", "RPE_r", "Static Baseline", kitti_seq, [0.05, 0.03, 0.02, 0.04, 0.08, 0.04, 0.05, 0.02, 0.05])
add("camera", "KITTI", "RPE_r", "Baseline", kitti_seq, [0.05, 0.03, 0.02, 0.07, 0.07, 0.07, 0.05, 0.04, 0.03])
add("camera", "KITTI", "RPE_r", "Hybrid", kitti_seq, [-0.01, 0.0, 0.0, 0.1, 0.0, -0.01, 0.0, 0.0, 0.0])
add("camera", "KITTI", "RPE_r", "iHybrid", kitti_seq, [0.0, -0.01, 0.0, 0.2, 0.0, -0.01, 0.0, 0.0, None])
add("camera", "KITTI", "RPE_r", "Parallel-Hybrid", kitti_seq, [0.0, 0.0, 0.0, 0.01, 0.02, 0.0, 0.01, 0.0, 0.0])

# Outdoor Cluster
add("camera", "Outdoor Cluster", "RPE_r", "Static Baseline", outdoor_seq, [0.02, 0.03, 0.02, 0.03])
add("camera", "Outdoor Cluster", "RPE_r", "Baseline", outdoor_seq, [0.02, 0.02, 0.01, 0.02])
add("camera", "Outdoor Cluster", "RPE_r", "Hybrid", outdoor_seq, [-0.01, -0.01, 0.0, 0.0])
add("camera", "Outdoor Cluster", "RPE_r", "iHybrid", outdoor_seq, [0.0, 0.0, 0.0, 0.0])
add("camera", "Outdoor Cluster", "RPE_r", "Parallel-Hybrid", outdoor_seq, [0.0, 0.01, 0.01, 0.0])

# OMD
add("camera", "OMD", "RPE_r", "Static Baseline", omd_seq, [0.71])
add("camera", "OMD", "RPE_r", "Baseline", omd_seq, [0.66])
add("camera", "OMD", "RPE_r", "Hybrid", omd_seq, [0.0])
add("camera", "OMD", "RPE_r", "iHybrid", omd_seq, [None])
add("camera", "OMD", "RPE_r", "Parallel-Hybrid", omd_seq, [0.0])


# --- Metric: RPE_t (Meters) ---
# KITTI
add("camera", "KITTI", "RPE_t", "Static Baseline", kitti_seq, [0.04, 0.06, 0.04, 0.1, 0.03, 0.01, 0.01, 0.05, 0.03])
add("camera", "KITTI", "RPE_t", "Baseline", kitti_seq, [0.04, 0.06, 0.06, 0.07, 0.06, 0.08, 0.01, 0.05, 0.04])
add("camera", "KITTI", "RPE_t", "Hybrid", kitti_seq, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
add("camera", "KITTI", "RPE_t", "iHybrid", kitti_seq, [0.00, 0.01, 0.01, -0.01, -0.01, -0.02, 0.00, 0.00, None])
add("camera", "KITTI", "RPE_t", "Parallel-Hybrid", kitti_seq, [0.00, 0.00, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])

# Outdoor Cluster
add("camera", "Outdoor Cluster", "RPE_t", "Static Baseline", outdoor_seq, [0.02, 0.02, 0.008, 0.05])
add("camera", "Outdoor Cluster", "RPE_t", "Baseline", outdoor_seq, [0.02, 0.01, 0.01, 0.02])
add("camera", "Outdoor Cluster", "RPE_t", "Hybrid", outdoor_seq, [0.0, 0.0, 0.0, 0.01])
add("camera", "Outdoor Cluster", "RPE_t", "iHybrid", outdoor_seq, [0.00, 0.00, 0.00, 0.00])
add("camera", "Outdoor Cluster", "RPE_t", "Parallel-Hybrid", outdoor_seq, [0.01, 0.00, 0.00, -0.01])

# OMD
add("camera", "OMD", "RPE_t", "Static Baseline", omd_seq, [0.006])
add("camera", "OMD", "RPE_t", "Baseline", omd_seq, [0.01])
add("camera", "OMD", "RPE_t", "Hybrid", omd_seq, [0.0])
add("camera", "OMD", "RPE_t", "iHybrid", omd_seq, [None])
add("camera", "OMD", "RPE_t", "Parallel-Hybrid", omd_seq, [0.0])


def convert_relative_to_absolute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts relative error values of non-baseline methods back into absolute values
    by adding them to their respective sequence's Baseline value.

    Leaves 'Baseline' and 'Static Baseline' values unchanged.
    """
    # 1. Create a copy so we don't mutate the original data frame unexpectedly
    df_absolute = df.copy()

    # 2. Extract only the baseline rows to build a lookup mapping
    # We group by the unique identifiers that define a specific experiment run
    baseline_df = df_absolute[df_absolute['method'] == 'Baseline']

    # Create a mapping dictionary: {(category, dataset, metric, sequence): baseline_value}
    baseline_lookup = baseline_df.set_index(['category', 'dataset', 'metric', 'sequence'])['value'].to_dict()

    # 3. Define an internal row-by-row mapping function
    def calculate_absolute(row):
        # We only alter rows that are relative variations (exclude original baselines)
        if row['method'] in ['Baseline', 'Static Baseline']:
            return row['value']

        # If the value is missing due to a system failure (None / NaN), leave it as-is
        if pd.isna(row['value']):
            return row['value']

        # Define the key matching this row's exact experiment criteria
        lookup_key = (row['category'], row['dataset'], row['metric'], row['sequence'])

        # Grab the baseline anchor value
        baseline_val = baseline_lookup.get(lookup_key)

        if baseline_val is not None and not pd.isna(baseline_val):
            # Absolute = Baseline + Relative offset
            # Rounding handles float precision noise (e.g., 0.15 + -0.08 resulting in 0.0700000000001)
            # subtract result as -0.04 actually means 0.04 worse than the baseline so we should make the error higher
            return round(baseline_val - row['value'], 4)

        return row['value']

    # 4. Apply the calculation down the dataframe axis
    df_absolute['value'] = df_absolute.apply(calculate_absolute, axis=1)

    return df_absolute

# =========================================================
# DATAFRAME
# =========================================================

df = pd.DataFrame(rows)
df = convert_relative_to_absolute(df)

def build_method_color_map(df):
    """
    Build a globally consistent color mapping for methods.

    Ensures:
    - every method always gets the same color
    - colors come from matplotlib rcParams cycle
    - consistent across ALL plots/subplots
    """

    base_colors = matplotlib.rcParams[
        'axes.prop_cycle'
    ].by_key()['color']

    # preserve insertion order from dataframe
    methods = list(dict.fromkeys(df["method"].values))

    method_to_color = {}

    for i, method in enumerate(methods):
        method_to_color[method] = base_colors[i % len(base_colors)]

    return method_to_color


# =========================================================
# BUILD GLOBAL COLOR MAP ONCE
# =========================================================

METHOD_COLORS = build_method_color_map(df)


# =========================================================
# NORMALIZATION
# =========================================================

def normalize(group):

    best = group.value.min()

    group["value"] = group.value / best

    return group


# =========================================================
# AGGREGATION
# =========================================================

def aggregate_per_dataset(df):

    dataset_order = list(dict.fromkeys(df["dataset"]))

    line_rows = []
    box_rows = []

    for category in df["category"].unique():
        for metric in df["metric"].unique():
            for method in df["method"].unique():

                sub = df[
                    (df["category"] == category) &
                    (df["metric"] == metric) &
                    (df["method"] == method)
                ]

                accumulated = []

                for dataset in dataset_order:

                    d = sub[sub["dataset"] == dataset]
                    if d.empty:
                        continue

                    vals = d["value"].values

                    # -------------------------
                    # LINE (mean + cumulative std)
                    # -------------------------
                    accumulated.extend(vals.tolist())

                    line_rows.append({
                        "category": category,
                        "dataset": dataset,
                        "metric": metric,
                        "method": method,
                        "value": np.mean(vals),
                        "std": np.std(accumulated, ddof=0),
                        "sequence": dataset
                    })

                    # -------------------------
                    # BOX (preserve distribution)
                    # -------------------------
                    for v in vals:
                        box_rows.append({
                            "category": category,
                            "dataset": dataset,
                            "metric": metric,
                            "method": method,
                            "value": v,
                            "sequence": dataset
                        })

    return pd.DataFrame(line_rows), pd.DataFrame(box_rows)

def remap_metric_to_title(metric):

    if metric == "ME_r":
        return r"ME$_r(^\circ)$"

    if metric == "ME_t":
        return r"ME$_t$(m)"

    if metric == "ATE":
        return r"ATE(m)"

    if metric == "RPE_r":
        return r"RPE$_r(^\circ)$"

    if metric == "RPE_t":
        return r"RPE$_t$(m)"

    return metric


def remap_method_to_title(method):

    if method == "WCME" or method == "WCPE":
        return method + " (ours)"

    return method

def remap_dataset_to_title(dataset):
    if AGGREGATE_PER_DATASET:
        if dataset == "KITTI":
            return "KITTI Tracking"
        if dataset == "TartanAir":
            return "TartanAir (Shibuya)"
    return dataset

def plot_box(ax, df, metric, global_legend_handles, global_added_legend):

    sub = df[df["metric"] == metric]

    labels = list(dict.fromkeys(sub["dataset"].values))

    face_alpha = 0.6

    # =====================================================
    # Layout parameters
    # =====================================================

    # BOX_WIDTH = 0.18
    # METHOD_SPACING = 0.22
    # GROUP_SPACING = 1.4

    BOX_WIDTH = 0.14
    METHOD_SPACING = 0.22
    GROUP_SPACING = 1.2
    # METHOD_SPACING = 0.1
    # GROUP_SPACING = 0.5

    dataset_centers = np.arange(len(labels)) * GROUP_SPACING

    # legend_handles = []
    # added_legend = set()

    # =====================================================
    # Plot dataset groups
    # =====================================================

    for dataset_idx, dataset in enumerate(labels):

        dataset_data = sub[sub["dataset"] == dataset]

        # -------------------------------------------------
        # Sort methods by performance
        #
        # largest on left
        # smallest on right
        # -------------------------------------------------

        method_means = []

        for method in dataset_data["method"].unique():

            vals = dataset_data[
                dataset_data["method"] == method
            ]["value"].values

            if len(vals) == 0:
                continue

            method_means.append((
                method,
                np.mean(vals)
            ))

        # Sort descending:
        # largest error -> left
        # smallest error -> right
        method_means.sort(
            key=lambda x: x[1],
            reverse=True
        )

        methods_present = [
            m[0] for m in method_means
        ]

        n_methods = len(methods_present)

        if n_methods == 0:
            continue

        # -------------------------------------------------
        # Center methods around dataset center
        # -------------------------------------------------

        offsets = (
            np.arange(n_methods) - (n_methods - 1) / 2
        ) * METHOD_SPACING

        for method_idx, method in enumerate(methods_present):

            method_data = dataset_data[
                dataset_data["method"] == method
            ]

            values = method_data["value"].values

            if len(values) == 0:
                continue

            pos = dataset_centers[dataset_idx] + offsets[method_idx]

            color = METHOD_COLORS[method]

            rgba = (
                *matplotlib.colors.to_rgb(color),
                face_alpha
            )

            # =================================================
            # Draw box
            # =================================================

            if PLOT_TYPE == "box":
                box = ax.boxplot(
                    [values],
                    positions=[pos],
                    widths=BOX_WIDTH,
                    patch_artist=True,
                    showfliers=False,
                    boxprops=dict(linewidth=2),
                    capprops=dict(linewidth=2),
                    whiskerprops=dict(linewidth=2),
                    medianprops=dict(linewidth=2)
                )

                # =================================================
                # Styling
                # =================================================

                box["boxes"][0].set_facecolor(rgba)
                box["boxes"][0].set_edgecolor(color)

                box["whiskers"][0].set_color(color)
                box["whiskers"][1].set_color(color)

                box["caps"][0].set_color(color)
                box["caps"][1].set_color(color)

                box["medians"][0].set_color(color)
            elif PLOT_TYPE == "mean":
                mean = np.mean(values)
                std = np.std(values)

                # Mean marker
                ax.scatter(
                    pos,
                    mean,
                    color=color,
                    marker="D",
                    s=50,
                    zorder=3,
                    label="_nolegend_"
                )

                if std != 0.0:
                    # Std error bar
                    ax.errorbar(
                        pos,
                        mean,
                        yerr=std,
                        fmt="none",
                        ecolor=color,
                        elinewidth=2,
                        capsize=4,
                        zorder=2
                    )

                # if metric == "ATE" and CATEGORY_TO_PLOT == "camera":
                #     print(f"{} STD {std}")

            # =================================================
            # Legend
            # =================================================

            # if method not in added_legend:

            #     legend_handles.append(
            #         matplotlib.patches.Patch(
            #             facecolor=rgba,
            #             edgecolor=color,
            #             label=remap_method_to_title(method)
            #         )
            #     )

            #     added_legend.add(method)
            if method not in global_added_legend:
                global_legend_handles.append(
                    matplotlib.patches.Patch(
                        facecolor=rgba,
                        edgecolor=color,
                        label=remap_method_to_title(method)
                    )
                )

                global_added_legend.add(method)

    # =====================================================
    # Axis formatting
    # =====================================================
    ax.set_xticks(dataset_centers)

    remapped_labels = [remap_dataset_to_title(label) for label in labels]
    ax.set_xticklabels(remapped_labels)

    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment('center')

    ax.set_ylabel(remap_metric_to_title(metric))
    # ax.set_title(metric)

    # ax.grid(True, alpha=0.4)

    # =====================================================
    # Vertical separators between dataset groups
    # =====================================================
    loosley_dashed = (0, (5, 10))

    for i in range(len(dataset_centers) - 1):

        boundary = (
            dataset_centers[i] +
            dataset_centers[i + 1]
        ) / 2

        ax.axvline(
            boundary,
            linestyle=loosley_dashed,
            linewidth=1,
            alpha=0.25,
            color="black"
        )


    # Define your threshold N
    # N = 4
    # # Determine ncols: if greater than N, split into 2 lines (round up for odd numbers)
    # num_handles = len(legend_handles)
    # ncols = (num_handles + 1) // 2 if num_handles > N else 1

    # # Replace your original line with this
    # ax.legend(handles=legend_handles, loc="best", ncol=ncols)

    formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))  # Force scientific notation outside this range
    # ax.yaxis.set_major_formatter(formatter)

    # if metric == "ATE" and CATEGORY_TO_PLOT == "camera":
    if CATEGORY_TO_PLOT == "camera":
        ax.set_yscale("log")
        from matplotlib.ticker import LogLocator
        # Minor ticks between powers of 10
        ax.yaxis.set_minor_locator(
            LogLocator(
                base=10.0,
                subs=np.arange(2, 10) * 0.1
            )
        )
        # ax.set_ylim(bottom=3e-3)
    else:
        ax.set_yscale("linear")

        ymin, ymax = ax.get_ylim()
        yrange = ymax - ymin
        minor_step = yrange / 8.0

        from matplotlib.ticker import MultipleLocator
        ax.yaxis.set_minor_locator(
            MultipleLocator(minor_step)
        )
        ax.set_ylim(bottom=0)



    ax.grid(
        which="minor",
        axis="y",
        linestyle="--",
        linewidth=1.0,
        alpha=0.3,
        color="tab:gray",
        zorder = 5
    )




# =========================================================
# LABEL REMAPPING
# =========================================================



# =========================================================
# PREPARE DATA
# =========================================================


df_plot = df[df.category == CATEGORY_TO_PLOT].copy()

if AGGREGATE_PER_DATASET:
    df_line, df_box = aggregate_per_dataset(df_plot)
else:
    df_line = df_plot.copy()
    df_line["std"] = 0.0
    df_box = df_plot.copy()

metrics = df_line.metric.unique()


# =========================================================
# PLOTTING
# =========================================================

def plot_line(ax, df, metric):

    sub = df[df["metric"] == metric]

    labels = list(dict.fromkeys(sub["dataset"]))
    x = np.arange(len(labels))

    methods = sub["method"].unique()
    color_list = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']

    for i, method in enumerate(methods):

        m = sub[sub["method"] == method]

        ys, stds = [], []

        for d in labels:

            r = m[m["dataset"] == d]
            if r.empty:
                ys.append(np.nan)
                stds.append(0)
            else:
                ys.append(r["value"].values[0])
                stds.append(r["std"].values[0])

        color = color_list[i % len(color_list)]

        ax.plot(
            x,
            ys,
            marker="o",
            linewidth=2,
            label=method,
            color=color
        )

        ax.fill_between(
            x,
            np.array(ys) - np.array(stds),
            np.array(ys) + np.array(stds),
            alpha=0.2,
            color=color
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(metric)
    ax.set_ylabel("Error")
    ax.grid(True, alpha=0.7)
    ax.legend()


def aggregate(df):

    dataset_order = list(dict.fromkeys(df["dataset"]))

    line_rows = []
    box_rows = []

    for category in df["category"].unique():
        for metric in df["metric"].unique():
            for method in df["method"].unique():

                sub = df[
                    (df["category"] == category) &
                    (df["metric"] == metric) &
                    (df["method"] == method)
                ]

                acc = []

                for dataset in dataset_order:

                    d = sub[sub["dataset"] == dataset]
                    if d.empty:
                        continue

                    vals = np.array(d["value"].values.tolist())
                    vals = vals[~np.isnan(vals)]

                    acc.extend(vals)

                    print(f"{method} {metric} {vals}")

                    # LINE
                    line_rows.append({
                        "category": category,
                        "metric": metric,
                        "method": method,
                        "dataset": dataset,
                        "value": np.mean(vals),
                        "std": np.std(acc)
                    })

                    # BOX (preserve distribution)
                    for v in vals:
                        box_rows.append({
                            "category": category,
                            "metric": metric,
                            "method": method,
                            "dataset": dataset,
                            "value": v
                        })

    return pd.DataFrame(line_rows), pd.DataFrame(box_rows)


from matplotlib.gridspec import GridSpec

def plot_all(df):

    global_legend_handles = []
    global_added_legend = set()

    df = df[df["category"] == CATEGORY_TO_PLOT]

    if AGGREGATE_PER_DATASET:
        df_line, df_box = aggregate(df)
    else:
        df_line = df.copy()
        df_box = df.copy()

    metrics = df_line["metric"].unique()
    n_metrics = len(metrics)

    # =========================================================
    # GRID SPEC LAYOUT
    # =========================================================

    fig = plt.figure(figsize=(15, 3 * n_metrics))

    gs = GridSpec(
        nrows=n_metrics + 2,   # +1 title row +1 legend row
        ncols=1,
        height_ratios=[0.07, 0.05] + [1.0] * n_metrics,
        figure=fig
    )

    # =========================================================
    # TITLE ROW
    # =========================================================
    ax_title = fig.add_subplot(gs[0])
    ax_title.axis("off")

    ax_title.text(
        0.5, 0.5,
        "Camera Pose Error Comparisons Per Sequence"
        if CATEGORY_TO_PLOT == "camera"
        else "Object Motion Error Comparisons Per Sequence",
        ha="center",
        va="center",
        # fontsize=18,
        fontweight="bold"
    )

    # =========================================================
    # LEGEND ROW
    # =========================================================

    # =========================================================
    # PLOTS
    # =========================================================

    axes = []
    for i, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[i + 2])
        axes.append(ax)

        print(f"Plotting metric: {metric}")

        if PLOT_TYPE == "box" or PLOT_TYPE == "mean":
            plot_box(ax, df_box, metric, global_legend_handles, global_added_legend)
        elif PLOT_TYPE == "line":
            plot_line(ax, df_line, metric)
        else:
            raise ValueError("Unknown PLOT_TYPE")

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color("black")

    ax_leg = fig.add_subplot(gs[1])
    ax_leg.axis("off")

    # NOTE: legend is figure-level but visually anchored here
    legend = ax_leg.legend(
        handles=global_legend_handles,
        loc="center",
        ncol=min(len(global_legend_handles), 6),
        frameon=True
    )

    # =========================================================
    # FINAL SPACING CONTROL
    # =========================================================
    # fig.tight_layout(pad=1.5)
    # fig.subplots_adjust(hspace=0.05)
    # fig.tight_layout()
    # fig.subplots_adjust(
    #     hspace=0.05,  # optional, but now safe to use
    #     wspace=0.01
    # )
    # for ax in axes:
    #     ax.margins(x=0.01, y=0.05)
    fig.subplots_adjust(
        left=0.06,
        right=0.995,
        top=0.95,
        bottom=0.08
    )

    # fig.subplots_adjust(
    #     hspace=0.11,   # ONLY knob you now need
    #     top=0.95,
    #     bottom=0.05
    # )

    return fig

# =========================================================
# FINALIZE
# =========================================================

if not AGGREGATE_PER_DATASET:
    plt.subplots_adjust(bottom=0.5)

# plt.tight_layout()

fig = plot_all(df)


# fig.tight_layout()
plt.show()

# file_path = f'/root/results/{CATEGORY_TO_PLOT}_errors_TRO.jpg'
# fig.savefig(file_path)

# plt.savefig(file_path)

# print(f"Saved plot to: {file_path}")

# plt.show()

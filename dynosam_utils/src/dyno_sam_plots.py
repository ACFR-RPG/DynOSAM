# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib import transforms
# import numpy as np

# # =========================================================
# # CONFIGURATION
# # =========================================================

# from dynosam_utils.evaluation.core.plotting import startup_plotting
# startup_plotting()


# CATEGORY_TO_PLOT = "camera"     # camera | object
# PLOT_TYPE = "line"              # bar | box | normalized | line

# sns.set_style("whitegrid")

# # plt.rcParams.update({
# #     "font.size": 14,
# #     "axes.titlesize": 16,
# #     "axes.labelsize": 14
# # })

# def partial_circle(x_center, y_center,
#                    radius_x, radius_y,
#                    start_deg, end_deg,
#                    n=100):
#     t = np.linspace(start_deg/180*np.pi, end_deg/180*np.pi, n)
#     return x_center + radius_x * np.cos(t), y_center + radius_y * np.sin(t)


# def underbrace(ax,
#                x_start, x_end,       # horizontal position in data coords
#                y=-0.15,              # vertical position in axes coords (0 = bottom, 1 = top)
#                label=None,
#                radius_y=0.02,        # vertical radius in axes coords
#                radius_x=0.1,         # horizontal radius in data coords
#                lw=1.0,
#                color="black",
#                text_offset=-0.04,    # extra offset for the label in axes coords
#                **text_kwargs):

#     # x in data coords, y in axes coords
#     trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

#     x_mid = 0.5 * (x_start + x_end)

#     y_top = y + radius_y
#     y_mid = y
#     y_bottom = y - radius_y

#     # left brace
#     X_circ_left, Y_circ_left = partial_circle(x_start + radius_x, y_top, radius_x, radius_y, 180, 270)
#     X_line_left = np.linspace(x_start + radius_x, x_mid - radius_x)
#     Y_line_left = np.full_like(X_line_left, y_mid)
#     # middle kink
#     X_circ_mid_left,  Y_circ_mid_left  = partial_circle(x_mid - radius_x, y_bottom, radius_x, radius_y, 90, 0)
#     X_circ_mid_right, Y_circ_mid_right = partial_circle(x_mid + radius_x, y_bottom, radius_x, radius_y, 180, 90)
#     # right brace
#     X_line_right = np.linspace(x_mid + radius_x, x_end - radius_x)
#     Y_line_right = np.full_like(X_line_right, y_mid)
#     X_circ_right, Y_circ_right = partial_circle(x_end - radius_x, y_top, radius_x, radius_y, 270, 360)

#     Xs = np.concatenate([X_circ_left, X_line_left,
#                          X_circ_mid_left, X_circ_mid_right,
#                          X_line_right, X_circ_right])
#     Ys = np.concatenate([Y_circ_left, Y_line_left,
#                          Y_circ_mid_left, Y_circ_mid_right,
#                          Y_line_right, Y_circ_right])

#     ax.plot(Xs, Ys, color=color, lw=lw, transform=trans, clip_on=False)

#     if label is not None:
#         ax.text(x_mid, y + text_offset, label,
#                 ha="center", va="top",
#                 transform=trans, color=color, **text_kwargs)



# # =========================================================
# # DATA STORAGE
# # =========================================================

# rows = []

# def add(category, dataset, metric, method, sequences, values):

#     for seq,val in zip(sequences, values):

#         rows.append({
#             "category": category,
#             "dataset": dataset,
#             "metric": metric,
#             "method": method,
#             "sequence": seq,
#             "value": val
#         })


# def get_dataset_ranges(rows, category="camera"):
#     """
#     Returns a dictionary mapping dataset names to their start and end sequences
#     for a given category, based on the flat 'rows' list of dicts.

#     Example output:
#     {'KITTI': ['00','20'], 'OutdoorCluster': ['L1','S2'], 'OMD': ['S4U','S4U']}
#     """
#     dataset_ranges = {}

#     # Filter rows by category
#     filtered_rows = [r for r in rows if r["category"] == category]

#     # Group sequences by dataset
#     from collections import defaultdict
#     dataset_seqs = defaultdict(set)
#     for r in filtered_rows:
#         dataset_seqs[r["dataset"]].add(r["sequence"])

#     # Compute start and end sequence per dataset
#     for dataset, seqs in dataset_seqs.items():
#         # Sort numeric first, then alphanumeric
#         seqs_sorted = sorted(seqs, key=lambda x: (not x.isdigit(), x))
#         dataset_ranges[dataset] = [seqs_sorted[0], seqs_sorted[-1]]

#     return dataset_ranges

# def get_dataset_positions(rows, category="camera"):
#     """
#     Returns a dictionary mapping dataset names to start/end x-axis positions
#     for a given category, based on the flat 'rows' list of dicts.

#     If a dataset has only one sequence, its span is expanded by 1 on each side
#     to make it visible on plots.

#     Example output:
#     {'KITTI': [0, 8], 'OutdoorCluster': [9, 12], 'OMD': [12, 13]}
#     """
#     dataset_ranges = get_dataset_ranges(rows, category)

#     # Get sequences in order of appearance (unique)
#     seq_order = []
#     seen = set()
#     for r in rows:
#         if r["category"] == category and r["sequence"] not in seen:
#             seq_order.append(r["sequence"])
#             seen.add(r["sequence"])

#     dataset_positions = {}
#     n = len(seq_order)
#     for dataset, (start_label, end_label) in dataset_ranges.items():
#         try:
#             start_idx = seq_order.index(start_label)
#             end_idx = seq_order.index(end_label)

#             # If start == end, expand span by 1 on each side if possible
#             if start_idx == end_idx:
#                 start_idx = max(0, start_idx - 1)
#                 end_idx = min(n - 1, end_idx + 1)
#         except ValueError:
#             start_idx = end_idx = None

#         dataset_positions[dataset] = [start_idx, end_idx]

#     return dataset_positions


# # =========================================================
# # CAMERA RESULTS
# # =========================================================

# # ------------------------
# # KITTI sequences
# # ------------------------

# kitti_seq = ["00","01","02","03","04","05","06","18","20"]

# add("camera","KITTI","ATE","DynaSLAM II",kitti_seq,
# [1.29,2.31,0.91,0.69,1.42,1.34,0.19,1.09,1.36])

# add("camera","KITTI","ATE","VDO-SLAM",kitti_seq,
# [3.37,6.74,2.47,2.12,4.53,3.8,0.45,9.94,7.82])

# add("camera","KITTI","ATE","WCPE",kitti_seq,
# [0.82,2.00,0.73,0.82,2.01,1.58,0.31,1.84,1.26])

# add("camera","KITTI","ATE","WCME",kitti_seq,
# [0.82,2.00,0.73,0.82,2.01,1.58,0.31,1.84,1.26])


# add("camera","KITTI","RPE_r","DynaSLAM II",kitti_seq,
# [0.06,0.04,0.02,0.06,0.06,0.03,0.04,0.02,0.04])

# add("camera","KITTI","RPE_r","VDO-SLAM",kitti_seq,
# [0.08,0.05,0.03,0.03,0.06,0.03,0.10,0.03,0.04])

# add("camera","KITTI","RPE_r","WCPE",kitti_seq,
# [0.05,0.03,0.02,0.05,0.06,0.06,0.05,0.04,0.04])

# add("camera","KITTI","RPE_r","WCME",kitti_seq,
# [0.04,0.03,0.02,0.05,0.06,0.05,0.05,0.04,0.04])


# add("camera","KITTI","RPE_t","DynaSLAM II",kitti_seq,
# [0.04,0.05,0.04,0.04,0.07,0.06,0.02,0.05,0.07])

# add("camera","KITTI","RPE_t","VDO-SLAM",kitti_seq,
# [0.09,0.15,0.05,0.09,0.14,0.11,0.04,0.09,0.30])

# add("camera","KITTI","RPE_t","WCPE",kitti_seq,
# [0.04,0.04,0.03,0.05,0.07,0.05,0.01,0.04,0.04])

# add("camera","KITTI","RPE_t","WCME",kitti_seq,
# [0.04,0.04,0.03,0.05,0.06,0.05,0.01,0.04,0.02])


# # ------------------------
# # Outdoor Cluster
# # ------------------------

# cluster_seq = ["L1","L2","S1","S2"]

# add("camera","OutdoorCluster","ATE","WCPE",cluster_seq,[0.61,0.52,0.09,0.13])
# add("camera","OutdoorCluster","ATE","WCME",cluster_seq,[0.61,0.52,0.09,0.13])

# add("camera","OutdoorCluster","RPE_r","WCPE",cluster_seq,[0.02,0.02,0.01,0.02])
# add("camera","OutdoorCluster","RPE_r","WCME",cluster_seq,[0.02,0.02,0.01,0.02])

# add("camera","OutdoorCluster","RPE_t","WCPE",cluster_seq,[0.04,0.02,0.01,0.02])
# add("camera","OutdoorCluster","RPE_t","WCME",cluster_seq,[0.04,0.01,0.01,0.02])


# # ------------------------
# # OMD
# # ------------------------

# add("camera","OMD","ATE","DynaSLAM II",["S4U"],[0.21])
# add("camera","OMD","ATE","VDO-SLAM",["S4U"],[0.19])
# add("camera","OMD","ATE","MVO",["S4U"],[0.05])
# add("camera","OMD","ATE","WCPE",["S4U"],[0.11])
# add("camera","OMD","ATE","WCME",["S4U"],[0.11])

# viode_seq = ["CD-Mid","CD-High","CN-Mid","CN-High","PL-Mid","PL-High"]

# # -------------------------------------------------
# # ATE (meters)
# # -------------------------------------------------

# add("camera","VIODE","ATE","DynaVINS",viode_seq,
# [0.104,0.150,0.194,0.147,0.056,0.065])

# add("camera","VIODE","ATE","ORB-SLAM3",
# ["CD-Mid","CN-Mid","CN-High"],
# [0.217,1.693,3.006])

# add("camera","VIODE","ATE","WCPE",viode_seq,
# [2.515,2.128,1.360,2.560,1.377,0.764])

# add("camera","VIODE","ATE","WCME",viode_seq,
# [2.515,2.128,1.360,2.560,1.377,0.764])


# # -------------------------------------------------
# # RPE_t (meters)
# # -------------------------------------------------

# add("camera","VIODE","RPE_t","DynaVINS",viode_seq,
# [0.024,0.027,0.019,0.023,0.019,0.015])

# add("camera","VIODE","RPE_t","WCPE",viode_seq,
# [0.008,0.014,0.015,0.020,0.006,0.005])

# add("camera","VIODE","RPE_t","WCME",viode_seq,
# [0.008,0.014,0.015,0.020,0.006,0.005])


# # -------------------------------------------------
# # RPE_r (degrees)
# # -------------------------------------------------

# add("camera","VIODE","RPE_r","DynaVINS",viode_seq,
# [0.087,0.09,0.102,0.096,0.126,0.111])

# add("camera","VIODE","RPE_r","WCPE",viode_seq,
# [0.049,0.105,0.070,0.190,0.036,0.040])

# add("camera","VIODE","RPE_r","WCME",viode_seq,
# [0.049,0.105,0.070,0.190,0.036,0.040])


# # ------------------------
# # TartanAir Shibuya
# # ------------------------

# tas_seq = ["I","II","III","IV","V","VI","VII"]

# add("camera","TartanAir","ATE","AirDOS",tas_seq,
# [0.06,0.02,0.10,0.03,0.02,0.22,0.56])

# add("camera","TartanAir","ATE","VDO-SLAM",tas_seq,
# [0.10,0.61,0.38,0.39,0.22,0.24,0.66])

# add("camera","TartanAir","ATE","WCPE",tas_seq,
# [0.03,0.03,0.02,0.02,0.03,0.04,0.18])

# add("camera","TartanAir","ATE","WCME",tas_seq,
# [0.02,0.04,0.02,0.02,0.04,0.03,0.19])



# # =========================================================
# # OBJECT RESULTS
# # =========================================================

# add("object","KITTI","ME_r","VDO-SLAM",kitti_seq,
# [1.38,2.15,1.68,0.39,2.8,0.48,2.8,0.36,0.47])

# add("object","KITTI","ME_r","MVO",["00"],[3.36])

# add("object","KITTI","ME_r","WCPE",kitti_seq,
# [1.23,0.91,0.95,0.27,0.76,0.56,2.8,1.15,0.39])

# add("object","KITTI","ME_r","WCME",kitti_seq,
# [1.29,0.86,1.06,0.26,1.01,0.49,0.39,0.6,0.33])


# add("object","KITTI","ME_t","VDO-SLAM",kitti_seq,
# [0.11,0.35,0.43,0.15,0.38,0.19,0.11,0.16,0.57])

# add("object","KITTI","ME_t","MVO",["00"],[0.27])

# add("object","KITTI","ME_t","WCPE",kitti_seq,
# [0.09,0.40,0.73,0.15,0.10,0.14,0.22,0.31,0.39])

# add("object","KITTI","ME_t","WCME",kitti_seq,
# [0.15,0.34,0.40,0.15,0.09,0.13,0.11,0.20,0.05])


# # ------------------------
# # OMD object (per-object)
# # ------------------------

# obj_ids = ["1","2","3","4"]

# add("object","OMD","ME_r","VDO-SLAM",obj_ids,
# [1.256,0.770,0.907,0.927])

# add("object","OMD","ME_r","MVO",obj_ids,
# [0.542,0.843,1.648,0.854])

# add("object","OMD","ME_r","WCME",obj_ids,
# [1.138,0.544,0.443,0.474])

# add("object","OMD","ME_t","VDO-SLAM",obj_ids,
# [0.0243,0.0234,0.0148,0.0293])

# add("object","OMD","ME_t","MVO",obj_ids,
# [0.0169,0.0269,0.0232,0.0309])

# add("object","OMD","ME_t","WCME",obj_ids,
# [0.0214,0.0233,0.0086,0.0291])


# cluster_obj_seq = ["L1","L2","S1","S2"]

# add("object","OutdoorCluster","ME_r","VDO-SLAM",cluster_obj_seq,
# [1.41, 1.11, 0.29, 0.35])

# add("object","OutdoorCluster","ME_r","WCPE",cluster_obj_seq,
# [0.93, 0.72, 0.18, 0.21])

# add("object","OutdoorCluster","ME_r","WCME",cluster_obj_seq,
# [0.88, 0.65, 0.17, 0.20])


# add("object","OutdoorCluster","ME_t","VDO-SLAM",cluster_obj_seq,
# [0.33, 0.28, 0.08, 0.09])

# add("object","OutdoorCluster","ME_t","WCPE",cluster_obj_seq,
# [0.21, 0.19, 0.06, 0.07])

# add("object","OutdoorCluster","ME_t","WCME",cluster_obj_seq,
# [0.18, 0.16, 0.05, 0.06])


# # =========================================================
# # DATAFRAME
# # =========================================================

# df = pd.DataFrame(rows)



# # =========================================================
# # NORMALIZATION
# # =========================================================

# def normalize(group):

#     best = group.value.min()

#     group["value"] = group.value / best

#     return group


# # =========================================================
# # PLOTTING
# # =========================================================

# df_plot = df[df.category == CATEGORY_TO_PLOT]

# metrics = df_plot.metric.unique()

# # fig, axes = plt.subplots(len(metrics),1, figsize=(10,4*len(metrics)))
# fig, axes = plt.subplots(len(metrics),1, figsize=(17,10))

# if CATEGORY_TO_PLOT == "object":
#     fig.suptitle("Object Motion Errors Per Sequence")
# if CATEGORY_TO_PLOT == "camera":
#     fig.suptitle("Camera Pose Errors Per Sequence")

# category_positions = get_dataset_positions(rows, CATEGORY_TO_PLOT)


# if len(metrics) == 1:
#     axes = [axes]

# def remap_metric_to_title(metric):
#     if metric == "ME_r":
#         return r"ME$_r(^\circ)$"
#     if metric == "ME_t":
#         return r"ME$_t$(m)"
#     if metric == "ATE":
#         return r"ATE(m)"
#     if metric == "RPE_r":
#         return r"RPE$_r(^\circ)$"
#     if metric == "RPE_t":
#         return r"RPE$_t$(m)"
#     return metric

# def remap_method_to_title(method):
#     if method == "WCME" or method == "WCPE":
#         return method + " (ours)"
#     return method

# for ax, metric in zip(axes, metrics):

#     sub = df_plot[df_plot.metric == metric]

#     metric = remap_metric_to_title(metric)

#     if PLOT_TYPE == "normalized":
#         sub = sub.groupby(["dataset","metric","sequence"]).apply(normalize)

#     if PLOT_TYPE == "bar":

#         sns.barplot(
#             data=sub,
#             x="dataset",
#             y="value",
#             hue="method",
#             ax=ax
#         )

#     elif PLOT_TYPE == "box":

#         sns.boxplot(
#             data=sub,
#             x="dataset",
#             y="value",
#             hue="method",
#             ax=ax
#         )

#     elif PLOT_TYPE == "line":
#         seq_order = []
#         seen = set()

#         # for r in rows:
#         #     if r["category"] == CATEGORY_TO_PLOT and r["sequence"] not in seen:
#         #         seq_order.append(r["sequence"])
#         #         seen.add(r["sequence"])

#         # x_positions = {s: i for i, s in enumerate(seq_order)}
#         seq_order = list(dict.fromkeys(sub["sequence"]))
#         x_positions = {s: i for i, s in enumerate(seq_order)}

#         # -------------------------------------------------
#         # Plot each method manually
#         # -------------------------------------------------
#         methods = sub["method"].unique()

#         for method in methods:

#             method_data = sub[sub["method"] == method]


#             xs = []
#             ys = []

#             for seq in seq_order:

#                 seq_rows = method_data[method_data["sequence"] == seq]

#                 if not seq_rows.empty:
#                     xs.append(x_positions[seq])
#                     ys.append(seq_rows["value"].values[0])

#                 print(seq)

#             if xs:
#                 marker = "x"
#                 if method == "WCME" or method == "WCPE":
#                     marker = 'o'
#                 ax.plot(
#                     xs,
#                     ys,
#                     marker=marker,
#                     linewidth=2,
#                     label=remap_method_to_title(method)
#                 )
#         # print(range)

#         # # -------------------------------------------------
#         # # Axis formatting
#         # # -------------------------------------------------
#         ax.set_xticks(range(len(seq_order)))
#         ax.set_xticklabels(seq_order)

#         ax.set_ylabel("Error")
#         ax.legend()


#         # sns.lineplot(
#         #     data=sub,
#         #     x="sequence",
#         #     y="value",
#         #     hue="method",
#         #     marker="o",
#         #     ax=ax
#         # )

#         # # ax.set_xlabel("Sequence", labelpad=20)
#         # ax.set_ylabel("Error")
#         dataset_positions = {}

#         for dataset in sub["dataset"].unique():

#             dataset_rows = sub[sub["dataset"] == dataset]

#             dataset_seqs = list(dict.fromkeys(dataset_rows["sequence"]))

#             start = x_positions[dataset_seqs[0]]
#             end = x_positions[dataset_seqs[-1]]

#             dataset_positions[dataset] = [start, end]

#         # -------------------------------------------------
#         # Draw dataset braces
#         # -------------------------------------------------

#         for dataset, seq_range in dataset_positions.items():

#             underbrace(ax, seq_range[0], seq_range[1], label=dataset)

#         # for dataset, r in category_positions.items():
#         #     if dataset in sub["dataset"].values:
#         #         underbrace(ax, r[0], r[1], label=dataset)


#     elif PLOT_TYPE == "normalized":

#         sns.barplot(
#             data=sub,
#             x="dataset",
#             y="value",
#             hue="method",
#             ax=ax
#         )

#         ax.axhline(1, linestyle="--")


#     ax.set_title(metric)



# plt.subplots_adjust(bottom=0.5)

# plt.tight_layout()
# # plt.show()

# file_path = f'/root/results/{CATEGORY_TO_PLOT}_errors_TRO.jpg'
# plt.savefig(file_path)

# # Array of values
# # values = np.array([[1,2,3], [1,2,3]])
# # values = [["a","b","c"], [1,2,3]]

# # # Line plot
# # fig, ax = plt.subplots()
# # # plt.plot(values[0,:], values[1,:], 'bo-',label='$P_{1}$')
# # plt.plot(values[0], values[1], 'bo-',label='$P_{1}$')

# # # make room for braces + labels
# # plt.subplots_adjust(bottom=0.25)

# # # add braces
# # # underbrace(ax, 1.0, 2.0, label="s", color="red")
# # underbrace(ax, 0.0, 2.0, label="s", color="red")
# # underbrace(ax, 2.0, 3, label="t - s", color="green")

# # plt.legend()
# # plt.show()


#################### ORIGINAL ########################






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
startup_plotting(font_size=16, text_size_scalar=0.9)


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

kitti_seq = ["00","01","02","03","04","05","06","18","20"]

add("camera","KITTI","ATE","DynaSLAM II",kitti_seq,
[1.29,2.31,0.91,0.69,1.42,1.34,0.19,1.09,1.36])

add("camera","KITTI","ATE","VDO-SLAM",kitti_seq,
[3.37,6.74,2.47,2.12,4.53,3.8,0.45,9.94,7.82])

add("camera","KITTI","ATE","WCPE",kitti_seq,
[0.82,2.00,0.73,0.82,2.01,1.58,0.31,1.84,1.26])

add("camera","KITTI","ATE","WCME",kitti_seq,
[0.82,2.00,0.73,0.82,2.01,1.58,0.31,1.84,1.26])


add("camera","KITTI","RPE_r","DynaSLAM II",kitti_seq,
[0.06,0.04,0.02,0.06,0.06,0.03,0.04,0.02,0.04])

add("camera","KITTI","RPE_r","VDO-SLAM",kitti_seq,
[0.08,0.05,0.03,0.03,0.06,0.03,0.10,0.03,0.04])

add("camera","KITTI","RPE_r","WCPE",kitti_seq,
[0.05,0.03,0.02,0.05,0.06,0.06,0.05,0.04,0.04])

add("camera","KITTI","RPE_r","WCME",kitti_seq,
[0.04,0.03,0.02,0.05,0.06,0.05,0.05,0.04,0.04])


add("camera","KITTI","RPE_t","DynaSLAM II",kitti_seq,
[0.04,0.05,0.04,0.04,0.07,0.06,0.02,0.05,0.07])

add("camera","KITTI","RPE_t","VDO-SLAM",kitti_seq,
[0.09,0.15,0.05,0.09,0.14,0.11,0.04,0.09,0.30])

add("camera","KITTI","RPE_t","WCPE",kitti_seq,
[0.04,0.04,0.03,0.05,0.07,0.05,0.01,0.04,0.04])

add("camera","KITTI","RPE_t","WCME",kitti_seq,
[0.04,0.04,0.03,0.05,0.06,0.05,0.01,0.04,0.02])


# ------------------------
# Outdoor Cluster
# ------------------------

cluster_seq = ["L1","L2","S1","S2"]

add("camera","OutdoorCluster","ATE","WCPE",cluster_seq,[0.61,0.52,0.09,0.13])
add("camera","OutdoorCluster","ATE","WCME",cluster_seq,[0.61,0.52,0.09,0.13])
# only average provided
add("camera","OutdoorCluster","ATE","ClusterSLAM",cluster_seq,[0.53,np.nan,np.nan,np.nan])

add("camera","OutdoorCluster","RPE_r","WCPE",cluster_seq,[0.02,0.02,0.01,0.02])
add("camera","OutdoorCluster","RPE_r","WCME",cluster_seq,[0.02,0.02,0.01,0.02])

add("camera","OutdoorCluster","RPE_t","WCPE",cluster_seq,[0.04,0.02,0.01,0.02])
add("camera","OutdoorCluster","RPE_t","WCME",cluster_seq,[0.04,0.01,0.01,0.02])

# only average provided
add("camera","OutdoorCluster","RPE_t","ClusterSLAM",cluster_seq,[1.10,np.nan,np.nan,np.nan])
add("camera","OutdoorCluster","RPE_r","ClusterSLAM",cluster_seq,[1.15,np.nan,np.nan,np.nan])


# ------------------------
# OMD
# ------------------------

add("camera","OMD","ATE","DynaSLAM II",["S4U"],[0.21])
add("camera","OMD","ATE","VDO-SLAM",["S4U"],[0.19])
add("camera","OMD","ATE","MVO",["S4U"],[0.05])
add("camera","OMD","ATE","WCPE",["S4U"],[0.11])
add("camera","OMD","ATE","WCME",["S4U"],[0.11])

viode_seq = ["CD-Mid","CD-High","CN-Mid","CN-High","PL-Mid","PL-High"]

# -------------------------------------------------
# ATE (meters)
# -------------------------------------------------

add("camera","VIODE","ATE","DynaVINS",viode_seq,
[0.104,0.150,0.194,0.147,0.056,0.065])

add("camera","VIODE","ATE","ORB-SLAM3",
["CD-Mid","CN-Mid","CN-High"],
[0.217,1.693,3.006])

add("camera","VIODE","ATE","WCPE",viode_seq,
[2.515,2.128,1.360,2.560,1.377,0.764])

add("camera","VIODE","ATE","WCME",viode_seq,
[2.515,2.128,1.360,2.560,1.377,0.764])


# -------------------------------------------------
# RPE_t (meters)
# -------------------------------------------------

add("camera","VIODE","RPE_t","DynaVINS",viode_seq,
[0.024,0.027,0.019,0.023,0.019,0.015])

add("camera","VIODE","RPE_t","WCPE",viode_seq,
[0.008,0.014,0.015,0.020,0.006,0.005])

add("camera","VIODE","RPE_t","WCME",viode_seq,
[0.008,0.014,0.015,0.020,0.006,0.005])


# -------------------------------------------------
# RPE_r (degrees)
# -------------------------------------------------

add("camera","VIODE","RPE_r","DynaVINS",viode_seq,
[0.087,0.09,0.102,0.096,0.126,0.111])

add("camera","VIODE","RPE_r","WCPE",viode_seq,
[0.049,0.105,0.070,0.190,0.036,0.040])

add("camera","VIODE","RPE_r","WCME",viode_seq,
[0.049,0.105,0.070,0.190,0.036,0.040])


# ------------------------
# TartanAir Shibuya
# ------------------------

tas_seq = ["I","II","III","IV","V","VI","VII"]

add("camera","TartanAir","ATE","AirDOS",tas_seq,
[0.06,0.02,0.10,0.03,0.02,0.22,0.56])

add("camera","TartanAir","ATE","VDO-SLAM",tas_seq,
[0.10,0.61,0.38,0.39,0.22,0.24,0.66])

add("camera","TartanAir","ATE","WCPE",tas_seq,
[0.03,0.03,0.02,0.02,0.03,0.04,0.18])

add("camera","TartanAir","ATE","WCME",tas_seq,
[0.02,0.04,0.02,0.02,0.04,0.03,0.19])



# =========================================================
# OBJECT RESULTS
# =========================================================

add("object","KITTI","ME_r","VDO-SLAM",kitti_seq,
[1.38,2.15,1.68,0.39,2.8,0.48,2.8,0.36,0.47])

add("object","KITTI","ME_r","MVO",["00"],[3.36])

add("object","KITTI","ME_r","WCPE",kitti_seq,
[1.23,0.91,0.95,0.27,0.76,0.56,2.8,1.15,0.39])

add("object","KITTI","ME_r","WCME",kitti_seq,
[1.29,0.86,1.06,0.26,1.01,0.49,0.39,0.6,0.33])


add("object","KITTI","ME_t","VDO-SLAM",kitti_seq,
[0.11,0.35,0.43,0.15,0.38,0.19,0.11,0.16,0.57])

add("object","KITTI","ME_t","MVO",["00"],[0.27])

add("object","KITTI","ME_t","WCPE",kitti_seq,
[0.09,0.40,0.73,0.15,0.10,0.14,0.22,0.31,0.39])

add("object","KITTI","ME_t","WCME",kitti_seq,
[0.15,0.34,0.40,0.15,0.09,0.13,0.11,0.20,0.05])


# ------------------------
# OMD object (per-object)
# ------------------------

obj_ids = ["1","2","3","4"]

add("object","OMD","ME_r","VDO-SLAM",obj_ids,
[1.256,0.770,0.907,0.927])

add("object","OMD","ME_r","MVO",obj_ids,
[0.542,0.843,1.648,0.854])

add("object","OMD","ME_r","WCME",obj_ids,
[1.138,0.544,0.443,0.474])

add("object","OMD","ME_t","VDO-SLAM",obj_ids,
[0.0243,0.0234,0.0148,0.0293])

add("object","OMD","ME_t","MVO",obj_ids,
[0.0169,0.0269,0.0232,0.0309])

add("object","OMD","ME_t","WCME",obj_ids,
[0.0214,0.0233,0.0086,0.0291])


cluster_obj_seq = ["L1","L2","S1","S2"]

add("object","OutdoorCluster","ME_r","VDO-SLAM",cluster_obj_seq,
[1.41, 1.11, 0.29, 0.35])

add("object","OutdoorCluster","ME_r","WCPE",cluster_obj_seq,
[0.93, 0.72, 0.18, 0.21])

add("object","OutdoorCluster","ME_r","WCME",cluster_obj_seq,
[0.88, 0.65, 0.17, 0.20])


add("object","OutdoorCluster","ME_t","VDO-SLAM",cluster_obj_seq,
[0.33, 0.28, 0.08, 0.09])

add("object","OutdoorCluster","ME_t","WCPE",cluster_obj_seq,
[0.21, 0.19, 0.06, 0.07])

add("object","OutdoorCluster","ME_t","WCME",cluster_obj_seq,
[0.18, 0.16, 0.05, 0.06])




# =========================================================
# DATAFRAME
# =========================================================

df = pd.DataFrame(rows)

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

def plot_box(ax, df, metric):

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

    legend_handles = []
    added_legend = set()

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

            if method not in added_legend:

                legend_handles.append(
                    matplotlib.patches.Patch(
                        facecolor=rgba,
                        edgecolor=color,
                        label=remap_method_to_title(method)
                    )
                )

                added_legend.add(method)

    # =====================================================
    # Axis formatting
    # =====================================================
    ax.set_xticks(dataset_centers)

    remapped_labels = [remap_dataset_to_title(label) for label in labels]
    ax.set_xticklabels(remapped_labels)

    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment('center')

    # ax.tick_params(
    #     axis="x",
    #     length=0   # removes tick marks entirely
    # )

    # ax.set_ylabel("Error")
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
    N = 4
    # Determine ncols: if greater than N, split into 2 lines (round up for odd numbers)
    num_handles = len(legend_handles)
    ncols = (num_handles + 1) // 2 if num_handles > N else 1

    # Replace your original line with this
    ax.legend(handles=legend_handles, loc="best", ncol=ncols)

    formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))  # Force scientific notation outside this range
    ax.yaxis.set_major_formatter(formatter)

    if metric == "ATE" and CATEGORY_TO_PLOT == "camera":
        ax.set_yscale("log")
        from matplotlib.ticker import LogLocator
        # Minor ticks between powers of 10
        ax.yaxis.set_minor_locator(
            LogLocator(
                base=10.0,
                subs=np.arange(2, 10) * 0.1
            )
        )
        ax.set_ylim(bottom=0.0001)
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


# =========================================================
# MAIN PLOT DRIVER
# =========================================================

def plot_all(df):

    df = df[df["category"] == CATEGORY_TO_PLOT]

    if AGGREGATE_PER_DATASET:
        df_line, df_box = aggregate(df)
    else:
        df_line = df.copy()
        df_box = df.copy()

    metrics = df_line["metric"].unique()

    # fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 4 * len(metrics)))
    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 3 * len(metrics)))

    if CATEGORY_TO_PLOT == "camera":
        fig.suptitle("Camera Pose Error Comparisons Per Sequence")
    if CATEGORY_TO_PLOT == "object":
        fig.suptitle("Object Motion Error Comparisons Per Sequence")

    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        print(f"Plotting metric: {metric}")

        if PLOT_TYPE == "box" or PLOT_TYPE == "mean":
            plot_box(ax, df_box, metric)

        elif PLOT_TYPE == "line":
            plot_line(ax, df_line, metric)

        else:
            raise ValueError("Unknown PLOT_TYPE")

# =========================================================
# FINALIZE
# =========================================================

if not AGGREGATE_PER_DATASET:
    plt.subplots_adjust(bottom=0.5)

# plt.tight_layout()

plot_all(df)


plt.tight_layout()
plt.show()

# file_path = f'/root/results/{CATEGORY_TO_PLOT}_errors.jpg'

# plt.savefig(file_path)

# print(f"Saved plot to: {file_path}")

# plt.show()

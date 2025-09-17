import dynosam_utils.evaluation.evaluation_lib as eval
import dynosam_utils.evaluation.core.metrics as eval_metrics
from dynosam_utils.evaluation.formatting_utils import * #for nice colours
from dynosam_utils.evaluation.core.plotting import startup_plotting, plot_object_trajectories
import evo.tools.plot as evo_plot
import matplotlib.pyplot as plt


import sys

plt.rcdefaults()
startup_plotting(30)



def make_plot(fig, directed_path, undirected_path, prefix = "dyno_mpc_backend", object=1):
    directed_dataset_eval = eval.DatasetEvaluator(directed_path)
    undirected_dataset_eval = eval.DatasetEvaluator(undirected_path)

    directed_data_files = directed_dataset_eval.make_data_files(prefix)
    undirected_data_files = undirected_dataset_eval.make_data_files(prefix)


    if not directed_data_files.check_is_dynosam_results():
        print(f"Invalid data file {directed_data_files}")
        sys.exit(0)

    if not undirected_data_files.check_is_dynosam_results():
        print(f"Invalid data file {undirected_data_files}")
        sys.exit(0)

    directed_motion_eval = directed_dataset_eval.create_motion_error_evaluator(directed_data_files)
    undirected_motion_eval = undirected_dataset_eval.create_motion_error_evaluator(undirected_data_files)

    directed_camera_eval = directed_dataset_eval.create_camera_pose_evaluator(directed_data_files)
    undirected_camera_eval = undirected_dataset_eval.create_camera_pose_evaluator(undirected_data_files)

    object_gt_trajectory = directed_motion_eval.object_poses_traj_ref[object]
    object_directed_traj = directed_motion_eval.object_poses_traj[object]
    object_undirected_traj = undirected_motion_eval.object_poses_traj[object]

    camera_gt_trajectory = directed_camera_eval.camera_pose_traj_ref
    camera_directed_traj = directed_camera_eval.camera_pose_traj
    camera_undirected_traj = undirected_camera_eval.camera_pose_traj



    traj = {}
    traj["Directed"] = object_directed_traj
    traj["Undirected"] = object_undirected_traj
    # traj["Directed"] = camera_directed_traj
    # traj["Undirected"] = camera_undirected_traj

    plot_object_trajectories(
        fig,
        traj,
        plot_mode=evo_plot.PlotMode.xy,
        plot_axis_est=False,
        plot_start_end_markers=True,
        axis_marker_scale=0.1,
        downscale=0.1,
        colours=[get_nice_green(), get_nice_blue()],
        traj_zorder=30,
        traj_linewidth=3.0)

    plot_object_trajectories(
        fig,
        {
        "Ground Truth": object_gt_trajectory,
        #  "Ground Truth": camera_gt_trajectory
         },
        plot_mode=evo_plot.PlotMode.xy,
        plot_axis_est=False,
        plot_start_end_markers=True,
        axis_marker_scale=0.1,
        downscale=0.1,
        colours=[get_nice_red()],
        traj_zorder=30,
        est_style="--",
        traj_linewidth=3.0)

if __name__ == "__main__":
    map_fig = plt.figure(figsize=(8,8))
    # ax = evo_plot.prepare_axis(map_fig, evo_plot.PlotMode.xyz)
    # ax = map_fig.add_subplot(111, projection="3d")
    ax = map_fig.gca()
    ax.set_ylabel(r"Y(m)")
    ax.set_xlabel(r"X(m)")
    # ax.set_zlabel(r"Z(m)")

    ax.set_title("Estimated Object Trajectory")

    make_plot(
        map_fig,
        "/root/results/Dynosam_icra2026/estimation_dynamic_obj_1_directed",
        "/root/results/Dynosam_icra2026/estimation_dynamic_obj_1_undirected"
    )
    ax.legend()
    # ax.patch.set_facecolor('white')
    eval.tools.set_clean_background(ax)
    # ax.axis('off')
    # map_fig.tight_layout()
    ax.grid(which='major', color='#DDDDDD', linewidth=1.0)
    map_fig.tight_layout(pad=0.05)

    # map_fig.savefig("/root/results/Dynosam_icra2026/estimation_dynamic_camera_trajectory.pdf")
    map_fig.savefig("/root/results/Dynosam_icra2026/estimation_dynamic_obj1_trajectory.pdf")

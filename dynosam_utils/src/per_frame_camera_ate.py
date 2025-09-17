import dynosam_utils.evaluation.evaluation_lib as eval
import dynosam_utils.evaluation.core.metrics as eval_metrics
from dynosam_utils.evaluation.formatting_utils import * #for nice colours
from dynosam_utils.evaluation.core.plotting import startup_plotting, plot_object_trajectories
import evo.tools.plot as evo_plot
import matplotlib.pyplot as plt
from evo.core import lie_algebra, trajectory, metrics, transformations


from pathlib import Path

import sys

plt.rcdefaults()
startup_plotting(30)



def make_plot(fig, directed_path, undirected_path, start_k, end_k, prefix = "dyno_mpc_backend"):

    camera_directed_ate = []
    camera_undirected_ate = []

    camera_directed_rpe_t = []
    camera_undirected_rpe_t = []

    camera_directed_rpe_r = []
    camera_undirected_rpe_r = []

    for k in range(start_k, end_k):
        prefix = f"dyno_mpc_k_{k}_backend"

        directed_dataset_eval = eval.DatasetEvaluator(directed_path)
        directed_data_files = directed_dataset_eval.make_data_files(prefix)

        if not directed_data_files.check_is_dynosam_results():
            print(f"Invalid data file {directed_data_files}")
            sys.exit(0)

        directed_camera_eval = directed_dataset_eval.create_camera_pose_evaluator(directed_data_files)

        camera_gt_trajectory = directed_camera_eval.camera_pose_traj_ref
        camera_directed_traj = directed_camera_eval.camera_pose_traj

        data = (camera_gt_trajectory, camera_directed_traj)
        ape_trans = metrics.APE(metrics.PoseRelation.translation_part)

        # rpe_trans = metrics.RPE(metrics.PoseRelation.translation_part,
        #                 1.0, metrics.Unit.frames, 0.0, False)
        # rpe_rot = metrics.RPE(metrics.PoseRelation.rotation_angle_deg,
        #                 1.0, metrics.Unit.frames, 1.0, False)

        ape_trans.process_data(data)
        # rpe_trans.process_data(data)
        # rpe_rot.process_data(data)

        camera_directed_ate.append(ape_trans.get_all_statistics()["rmse"])
        # camera_directed_rpe_t.append(rpe_trans.get_all_statistics()["rmse"])
        # camera_directed_rpe_r.append(rpe_rot.get_all_statistics()["rmse"])

    # undirected_dataset_eval = eval.DatasetEvaluator(undirected_path)

    # directed_data_files = directed_dataset_eval.make_data_files(prefix)
    # undirected_data_files = undirected_dataset_eval.make_data_files(prefix)


    # if not directed_data_files.check_is_dynosam_results():
    #     print(f"Invalid data file {directed_data_files}")
    #     sys.exit(0)

    # if not undirected_data_files.check_is_dynosam_results():
    #     print(f"Invalid data file {undirected_data_files}")
    #     sys.exit(0)

    # directed_motion_eval = directed_dataset_eval.create_motion_error_evaluator(directed_data_files)
    # undirected_motion_eval = undirected_dataset_eval.create_motion_error_evaluator(undirected_data_files)

    # directed_camera_eval = directed_dataset_eval.create_camera_pose_evaluator(directed_data_files)
    # undirected_camera_eval = undirected_dataset_eval.create_camera_pose_evaluator(undirected_data_files)

    # object_gt_trajectory = directed_motion_eval.object_poses_traj_ref[object]
    # object_directed_traj = directed_motion_eval.object_poses_traj[object]
    # object_undirected_traj = undirected_motion_eval.object_poses_traj[object]

    # camera_gt_trajectory = directed_camera_eval.camera_pose_traj_ref
    # camera_directed_traj = directed_camera_eval.camera_pose_traj
    # camera_undirected_traj = undirected_camera_eval.camera_pose_traj

    print(camera_directed_ate)

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
        "/root/results/Dynosam_icra2026/following_disappearing_object_1_2",
        "/root/results/Dynosam_icra2026/estimation_dynamic_obj_1_undirected",
        start_k=2,
        end_k=14
    )
    ax.legend()

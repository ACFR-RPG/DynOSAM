import dynosam_utils.evaluation.evaluation_lib as eval
import dynosam_utils.evaluation.core.metrics as eval_metrics
from dynosam_utils.evaluation.formatting_utils import * #for nice colours
from dynosam_utils.evaluation.core.plotting import startup_plotting, plot_object_trajectories
import evo.tools.plot as evo_plot
import matplotlib.pyplot as plt
from evo.core import lie_algebra, trajectory, metrics, transformations
import dynosam_utils.evaluation.core.metrics as dyno_metrics
import matplotlib.animation as animation




from pathlib import Path

import sys

plt.rcdefaults()
startup_plotting(30)


def find_min_max_k(path:str):
    import re
    # Regex: capture k, allow arbitrary stuff after "_backend"
    pattern = re.compile(r"dyno_mpc_k_(\d+)_backend.*")

    k_values = []
    for f in Path(path).glob("dyno_mpc_k_*_backend*"):
        match = pattern.fullmatch(f.name)
        if match:
            k_values.append(int(match.group(1)))

    if k_values:
        return min(k_values), max(k_values)
    else:
        print("No matching files found")
        return None

def plot_single_error_tuple(ax_ate, ax_rpe_r, ax_rpe_t, errors, label):
    camera_ate, camera_rpe_r, camera_rpe_t = errors

    x_ate = list(range(len(camera_ate)))
    x_rpe_r = list(range(len(camera_rpe_r)))
    x_rpe_t = list(range(len(camera_rpe_t)))

    ax_ate.plot(x_ate, camera_ate, label=f"{label}")
    ax_rpe_r.plot(x_rpe_r, camera_rpe_r, label=f"{label}")
    ax_rpe_t.plot(x_rpe_t, camera_rpe_t, label=f"{label}")



def plot_directed_undirected_errors(ax_ate, ax_rpe_r, ax_rpe_t, directed_errors, undirected_errors):
    plot_single_error_tuple(ax_ate, ax_rpe_r, ax_rpe_t, directed_errors, "Directed")
    plot_single_error_tuple(ax_ate, ax_rpe_r, ax_rpe_t, undirected_errors, "Undirected")

def load_object_evaluation_results_from_range(path:str, start_k:int, end_k:int, object_id=1):

    rme_r = []
    rme_t = []

    print(f"Loading dataset eval from {path}")
    dataset_eval = eval.DatasetEvaluator(path)


    for k in range(start_k+3, end_k):
        prefix = f"dyno_mpc_k_{k}_backend"
        data_files = dataset_eval.make_data_files(prefix)

        if not data_files.check_is_dynosam_results():
            print(f"Invalid data file {data_files}")
            sys.exit(0)


        object_eval = dataset_eval.create_motion_error_evaluator(data_files)

        object_poses_ref_traj = object_eval.object_poses_traj_ref[object_id]
        object_motion_traj_est = object_eval.object_motion_traj[object_id]

        data = (object_poses_ref_traj, object_motion_traj_est)
        rme_trans = dyno_metrics.RME(metrics.PoseRelation.translation_part)
        rme_rot = dyno_metrics.RME(metrics.PoseRelation.rotation_angle_deg)

        rme_trans.process_data(data)
        rme_rot.process_data(data)



        rme_r.append(rme_trans.get_all_statistics()["rmse"])
        rme_t.append(rme_rot.get_all_statistics()["rmse"])

    return rme_t, rme_r


def load_camera_evaluation_results_from_range(path:str, start_k:int, end_k:int):

    camera_ate = []
    camera_rpe_t = []
    camera_rpe_r = []

    print(f"Loading dataset eval from {path}")
    dataset_eval = eval.DatasetEvaluator(path)


    for k in range(start_k, end_k):
        prefix = f"dyno_mpc_k_{k}_backend"
        data_files = dataset_eval.make_data_files(prefix)

        if not data_files.check_is_dynosam_results():
            print(f"Invalid data file {data_files}")
            sys.exit(0)


        directed_camera_eval = dataset_eval.create_camera_pose_evaluator(data_files)

        camera_gt_trajectory = directed_camera_eval.camera_pose_traj_ref
        camera_directed_traj = directed_camera_eval.camera_pose_traj

        data = (camera_gt_trajectory, camera_directed_traj)
        ape_trans = metrics.APE(metrics.PoseRelation.translation_part)


        if k > start_k:
            rpe_trans = metrics.RPE(metrics.PoseRelation.translation_part,
                            1.0, metrics.Unit.frames, 0.0, False)
            rpe_rot = metrics.RPE(metrics.PoseRelation.rotation_angle_deg,
                            1.0, metrics.Unit.frames, 1.0, False)
            rpe_trans.process_data(data)
            rpe_rot.process_data(data)

            camera_rpe_t.append(rpe_trans.get_all_statistics()["rmse"])
            camera_rpe_r.append(rpe_rot.get_all_statistics()["rmse"])


        ape_trans.process_data(data)
        camera_ate.append(ape_trans.get_all_statistics()["rmse"])

    return camera_ate, camera_rpe_r, camera_rpe_t


def collect_data(directed_path, undirected_path, evaluation_func):
    directed_start_end = find_min_max_k(directed_path)
    undirected_start_end = find_min_max_k(undirected_path)

    if directed_start_end and undirected_start_end:
        directed_min_k, directed_max_k = directed_start_end
        undirected_min_k, undirected_max_k = undirected_start_end

        start_k = max(directed_min_k, undirected_min_k)
        end_k = min(directed_max_k, undirected_max_k)

        print(f"Found start k={start_k} and end k={end_k}")


        evaluation_func(directed_path, undirected_path, start_k, end_k)

        # directed_errors = load_camera_evaluation_results_from_range(
        #     directed_path,
        #     start_k,end_k
        # )

        # undirected_errors = load_camera_evaluation_results_from_range(
        #     undirected_path,
        #     start_k,end_k
        # )

        # plot_directed_undirected_errors(
        #     ax_ate, ax_rpe_r, ax_rpe_t,
        #     directed_errors,
        #     undirected_errors
        # )

    else:
        print("Could not load files or find min and max!")

plot_camera_animation = True
plot_object_animation = True

def make_camera_plot(directed_path, undirected_path):

    ate_fig = plt.figure(figsize=(8,8))
    rpe_t_fig = plt.figure(figsize=(8,8))
    rpe_r_fig = plt.figure(figsize=(8,8))

    # ate_fig.suptitle("Robot Trajectory Estimation Errors")

    ate_ax = ate_fig.gca()
    rpe_t_ax = rpe_t_fig.gca()
    rpe_r_ax = rpe_r_fig.gca()

    rpe_r_ax.set_ylabel("RPE$_r$(\N{degree sign})")
    rpe_t_ax.set_ylabel("RPE$_t$(m)")
    ate_ax.set_ylabel("ATE(m)")

    ate_ax.set_xlabel("Frame Index [-]")
    rpe_t_ax.set_xlabel("Frame Index [-]")
    rpe_r_ax.set_xlabel("Frame Index [-]")

    def plot_camera(directed_path, undirected_path, start_k, end_k):
        directed_errors = load_camera_evaluation_results_from_range(
            directed_path,
            start_k,end_k
        )

        undirected_errors = load_camera_evaluation_results_from_range(
            undirected_path,
            start_k,end_k
        )
        if plot_camera_animation:
            directed_ate, directed_rpe_r, directed_rpe_t = directed_errors
            undirected_ate, undirected_rpe_r, undirected_rpe_t = undirected_errors
            x = list(range(len(directed_ate)))

            # Prepare empty plot lines for animation
            (line_d_ate,) = ate_ax.plot([], [], label="Directed", color="C0")
            (line_u_ate,) = ate_ax.plot([], [], label="Undirected", color="C1")

            (line_d_rpe_r,) = rpe_r_ax.plot([], [], label="Directed", color="C0")
            (line_u_rpe_r,) = rpe_r_ax.plot([], [], label="Undirected", color="C1")

            (line_d_rpe_t,) = rpe_t_ax.plot([], [], label="Directed", color="C0")
            (line_u_rpe_t,) = rpe_t_ax.plot([], [], label="Undirected", color="C1")


            # Set axis limits upfront
            ate_ax.set_xlim(0, len(x))
            ate_ax.set_ylim(0, max(max(directed_ate), max(undirected_ate)) * 1.1)

            rpe_r_ax.set_xlim(0, len(x))
            rpe_r_ax.set_ylim(0, max(max(directed_rpe_r), max(undirected_rpe_r)) * 1.1)

            rpe_t_ax.set_xlim(0, len(x))
            rpe_t_ax.set_ylim(0, max(max(directed_rpe_t), max(undirected_rpe_t)) * 1.1)

            ate_ax.legend(loc="upper left")


            # Animation update function
            def update(frame):
                line_d_ate.set_data(x[:frame], directed_ate[:frame])
                line_u_ate.set_data(x[:frame], undirected_ate[:frame])

                line_d_rpe_r.set_data(x[:frame], directed_rpe_r[:frame])
                line_u_rpe_r.set_data(x[:frame], undirected_rpe_r[:frame])

                line_d_rpe_t.set_data(x[:frame], directed_rpe_t[:frame])
                line_u_rpe_t.set_data(x[:frame], undirected_rpe_t[:frame])

                return (line_d_ate, line_u_ate,
                        line_d_rpe_r, line_u_rpe_r,
                        line_d_rpe_t, line_u_rpe_t)


            # Make animation
            ani = animation.FuncAnimation(
                ate_fig,
                update,
                frames=len(x),
                interval=100,  # ms between frames
                blit=True,
            )
            save_mp4 = True
            if save_mp4:
                writer = animation.FFMpegWriter(fps=30, bitrate=6000)
                ani.save("/root/results/Dynosam_icra2026/camera_errors_animation.mp4", writer=writer)
        else:
            plot_directed_undirected_errors(
                ate_ax, rpe_r_ax, rpe_t_ax,
                directed_errors,
                undirected_errors
            )
            set_legend(ate_ax, rpe_r_ax, rpe_t_ax)


    collect_data(
        directed_path, undirected_path,
        plot_camera
    )



    if not plot_camera_animation:
        ate_fig.savefig("/root/results/Dynosam_icra2026/badly_tuned_static_result.pdf")


def make_object_plot(directed_path, undirected_path):

    # rme_t_fig = plt.figure(figsize=(8,8))
    # rme_r_fig = plt.figure(figsize=(8,8))

    # rme_t_fig.suptitle("Object Trajectory Estimation")
    fig, (ax_me_r, ax_me_t) = plt.subplots(nrows=2, sharex=True, layout="constrained")
    fig.set_size_inches(9, 9)
    fig.dpi = 800

    # fig.suptitle("Object Trajectory Estimation Errors")

    ax_me_r.set_ylabel("ME$_r$(\N{degree sign})")
    ax_me_t.set_ylabel("ME$_t$(m)")

    ax_me_r.set_xlabel("Frame Index [-]")
    ax_me_t.set_xlabel("Frame Index [-]")

    def plot_objects(directed_path, undirected_path, start_k, end_k):
        directed_me_r, directed_me_t = load_object_evaluation_results_from_range(
            directed_path,
            start_k,end_k
        )

        undirected_me_r, undirected_me_t  = load_object_evaluation_results_from_range(
            undirected_path,
            start_k,end_k
        )

        if plot_object_animation:
            x = list(range(len(directed_me_r)))

            # Prepare empty plot lines for animation
            (line_d_me_r,) = ax_me_r.plot([], [], label="Directed", color="C0")
            (line_u_me_r,) = ax_me_r.plot([], [], label="Undirected", color="C1")

            (line_d_me_t,) = ax_me_t.plot([], [], label="Directed", color="C0")
            (line_u_me_t,) = ax_me_t.plot([], [], label="Undirected", color="C1")

            ax_me_r.legend(loc="upper left")
            ax_me_t.legend(loc="upper left")

            # Set axis limits upfront
            ax_me_r.set_xlim(0, len(x))
            ax_me_r.set_ylim(0, max(max(directed_me_r), max(undirected_me_r)) * 1.1)

            ax_me_t.set_xlim(0, len(x))
            ax_me_t.set_ylim(0, max(max(directed_me_t), max(undirected_me_t)) * 1.1)

            # Animation update function
            def update(frame):
                line_d_me_r.set_data(x[:frame], directed_me_r[:frame])
                line_u_me_r.set_data(x[:frame], undirected_me_r[:frame])

                line_d_me_t.set_data(x[:frame], directed_me_t[:frame])
                line_u_me_t.set_data(x[:frame], undirected_me_t[:frame])

                return line_d_me_r, line_u_me_r, line_d_me_t, line_u_me_t

            # Make animation
            ani = animation.FuncAnimation(
                fig,
                update,
                frames=len(x),
                interval=100,  # ms between frames
                blit=True,
            )

            save_mp4 = True
            if save_mp4:
                writer = animation.FFMpegWriter(fps=30, bitrate=6000)
                ani.save("/root/results/Dynosam_icra2026/object_errors_animation.mp4", writer=writer)
        else:

            x_me_r = list(range(len(directed_me_r)))
            x_me_t = list(range(len(directed_me_t)))

            ax_me_r.plot(x_me_r, directed_me_r, label="Directed")
            ax_me_r.plot(x_me_r, undirected_me_r, label="Undirected")

            ax_me_t.plot(x_me_t, directed_me_t, label="Directed")
            ax_me_t.plot(x_me_t, undirected_me_t, label="Undirected")

            set_legend(ax_me_r, ax_me_t)




    collect_data(
        directed_path, undirected_path,
        plot_objects
    )


    fig.savefig("/root/results/Dynosam_icra2026/badly_tuned_dynamic_result.pdf")



def set_legend(*axes):
    for ax in axes:
        ax.legend()



if __name__ == "__main__":

    make_camera_plot(
        "/root/results/Dynosam_icra2026/estimation_static_every_k_data/new_badly_tuned_estimation_static_directed",
        "/root/results/Dynosam_icra2026/estimation_static_every_k_data/new_badly_tuned_estimation_static_undirected"
    )


    make_object_plot(
        "/root/results/Dynosam_icra2026/estimation_dynamic_every_k_data/estimation_dynamic_obj_1_directed_new_all_k",
        "/root/results/Dynosam_icra2026/estimation_dynamic_every_k_data/estimation_dynamic_obj_1_undirected_new_all_k"
    )

    # plt.show()

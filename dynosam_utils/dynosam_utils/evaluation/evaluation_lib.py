import csv
import os
import logging
from typing import Optional, List, Tuple, Dict, TypeAlias
from pathlib import Path
import matplotlib.axes
import matplotlib.figure
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from evo.core import lie_algebra, trajectory, metrics, transformations
import evo.tools.plot as evo_plot

import matplotlib

import copy

from dynosam_utils.evaluation.filesystem_utils import DataFiles, read_csv
import dynosam_utils.evaluation.core.plotting as plotting
import dynosam_utils.evaluation.core.metrics as dyno_metrics

from dynosam_utils.evaluation.tools import (
    common_entries,
    transform_camera_trajectory_to_world,
    camera_coordinate_to_world,
    TrajectoryHelper,
    load_pose_from_row,
    reconstruct_trajectory_from_relative,
    calculate_omd_errors,
    load_bson
)

import dynosam_utils.evaluation.tools as tools
import dynosam_utils.evaluation.core as core


logger = logging.getLogger("dynosam_eval")
# ----> console info messages require these lines <----
# create console handler and set level to debug
_ch = logging.StreamHandler()
_ch.setLevel(logging.DEBUG)
_ch.setFormatter(logging.Formatter(
    fmt='[%(levelname)s][%(filename)s:%(lineno)d] %(name)s: %(message)s'))
# add ch to logger
logger.addHandler(_ch)


## {object_id: { frame_id: homogenous_matrix }}
ObjectPoseDict: TypeAlias = Dict[int, Dict[int, np.ndarray]]
## {object_id: trajectory.PosePath3D}
ObjectTrajDict: TypeAlias = Dict[int, trajectory.PoseTrajectory3D]


class Evaluator(ABC):

    @abstractmethod
    def process(self, plot_collection: evo_plot.PlotCollection, results: Dict):
        pass




class MiscEvaluator(Evaluator):

    def __init__(self, output_folder_path: str):
        self._output_folder_path: str = output_folder_path
        self._tracklet_length_hist_file = os.path.join(
            self._output_folder_path,
            "tracklet_length_hist.json"
        )

    def process(self, plot_collection: evo_plot.PlotCollection, results: Dict):
        self._process_tracklet_length_data(plot_collection)


    def _process_tracklet_length_data(self, plot_collection):
        if not os.path.isfile(self._tracklet_length_hist_file):
            logger.error(f"Missing tracklet length histogram file at {self._tracklet_length_hist_file}. Skipping process.")
            return

        import matplotlib.ticker as mticker
        # assume bin size is all the same!!
        tracklet_length_data = load_bson(self._tracklet_length_hist_file)[0]['data']

        bin_labels = None
        # per object average per bin
        average_histograms = {}

        def process_bin_data(tracking_data, object_id):
            bin_labels_impl = []
            values = []
            for index, bin_data in enumerate(tracking_data):
                count = bin_data["count"]
                lower = bin_data["lower"]
                upper = bin_data["upper"]
                label = f"{int(lower)} - {int(upper)}"
                bin_labels_impl.append(label)
                values.append(count)

            # assume bins are always all the same!!
            nonlocal bin_labels
            if bin_labels is None:
                bin_labels = bin_labels_impl

            values = np.array(values)
            if object_id not in average_histograms:
                average_histograms[object_id] = values
            else:
                previous_stack = average_histograms[object_id]
                average_histograms[object_id] = np.vstack((previous_stack, values))


        # print(tracklet_length_data)
        for frame_id, per_object_tracking_data in tracklet_length_data.items():
            for object_id, histogram in per_object_tracking_data.items():
                object_id = int(object_id)
                # Tracking data is a dictionary of histogram name (which really we can ignore)
                # and then a vector of bin information
                # each bin will contain ["count", "lower", "upper"] indicating the size and value for each bin
                # e.g. {'tacklet-length-0': [[{'count': 156.0, 'lower': 0.0, 'upper': 1.0}, {...}]]}
                # note that the vector is nested because thei histrogram class in C++ contains an array of each axis
                # so even if we have one axis we have a vector
                # note: get the first element as wrapping in a list adds a new list
                # e.g {'tacklet-length-0': [[[{}]]]
                # and we want to mantain the original structure
                histogram = list(histogram.values())[0]
                # check that the tracking data contains only one axis
                assert(len(histogram) == 1)
                # access the first axis of the histogram
                tracking_data = histogram[0]

                process_bin_data(tracking_data, object_id)

        # print(average_histograms)
        # # take average over histogram
        for object_id, avgs in average_histograms.items():
            # print(object_id)
            # reshape to get (N x columns) where N is the number of data points
            # print(avgs)
            avgs = avgs.reshape(-1, len(bin_labels))
            # print(object_id)
            # print(avgs)
            # avgs = np.mean(avgs, axis=0).astype(int)
            avgs = np.median(avgs, axis=0).astype(int)

            # print(avgs)

            fig = plt.figure(figsize=(8,8))
            ax = fig.gca()
            ax.bar(bin_labels, avgs)

            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
            # # Rotate x-axis labels at 90 degrees and remove default x-axis coordinates
            ax.set_xticklabels(bin_labels, rotation=90)
            # fixing xticks with FixedLocator but also using MaxNLocator to avoid cramped x-labels
            # ax.xaxis.set_major_locator(mticker.MaxNLocator(len(bin_labels) + 1))
            # ticks_loc = ax.get_xticks().tolist()
            # ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
            # ax.set_xticklabels(bin_labels, rotation=90)

            # Remove the x-axis line
            ax.spines['bottom'].set_visible(False)

            # Add labels for each bar
            for i, v in enumerate(avgs):
                ax.text(i, v + 1, str(v), ha='center', va='bottom')


            fig.suptitle(f"Tracking Length Hist {object_id}")
            plot_collection.add_figure(f"Tracking Length Hist {object_id}", fig)


class MotionErrorEvaluator(Evaluator):
    def __init__(self, object_motion_log:str, object_pose_log:str) -> None:
        self._object_motion_log = object_motion_log
        self._object_pose_log = object_pose_log

        self._object_pose_log_file = read_csv(self._object_pose_log,
                                              ["frame_id", "object_id",
                                               "tx", "ty", "tz", "qx", "qy", "qz", "qw",
                                               "gt_tx", "gt_ty", "gt_tz", "gt_qx", "gt_qy", "gt_qz", "gt_qw"])

        self._object_motion_log_file = read_csv(self._object_motion_log,
                                            ["frame_id", "object_id",
                                            "tx", "ty", "tz", "qx", "qy", "qz", "qw",
                                            "gt_tx", "gt_ty", "gt_tz", "gt_qx", "gt_qy", "gt_qz", "gt_qw"])

        self._object_poses_traj: ObjectTrajDict = None
        self._object_poses_traj_ref: ObjectTrajDict = None

        self._object_motions_traj: ObjectTrajDict = None
        self._object_motions_traj_ref: ObjectTrajDict = None

        # TODO: need to make traj's a property trajectory by using the frame id as timestamp - this allows us to synchronize
        self._object_poses_traj, self._object_poses_traj_ref = MotionErrorEvaluator._construct_object_se3_trajectories(self._object_pose_log_file, convert_to_world_coordinates=True)
        self._object_motions_traj, self._object_motions_traj_ref = MotionErrorEvaluator._construct_object_se3_trajectories(self._object_motion_log_file, convert_to_world_coordinates=True)

        # cache of objects that exist per frame taken from the estimated _object_poses_traj
        # dictionary of frame ids's to a list of objects in that frame
        self._objects_at_frame = {}
        self._calculate_objects_per_frame()



    @property
    def object_poses_traj(self) -> ObjectTrajDict:
        return self._object_poses_traj

    @property
    def object_poses_traj_ref(self) -> ObjectTrajDict:
        return self._object_poses_traj_ref

    @property
    def object_motion_traj(self) -> ObjectTrajDict:
        return self._object_motions_traj

    @property
    def object_motion_traj_ref(self) -> ObjectTrajDict:
        return self._object_motions_traj_ref

    @property
    def frames(self):
        return list(self._objects_at_frame.keys())

    def objects_at_frame(self, frame):
        if frame in self._objects_at_frame:
            return self._objects_at_frame[frame]
        return None




    def make_object_trajectory(self, object_id: int) -> Optional[tools.ObjectMotionTrajectory]:
        if object_id in self.object_poses_traj and object_id in self.object_motion_traj:
            return tools.ObjectMotionTrajectory(self.object_poses_traj[object_id], self.object_motion_traj[object_id])
        return None

    def make_object_ref_trajectory(self, object_id: int) -> Optional[tools.ObjectMotionTrajectory]:
        if object_id in self.object_poses_traj_ref and object_id in self.object_motion_traj_ref:
            return tools.ObjectMotionTrajectory(self.object_poses_traj_ref[object_id], self.object_motion_traj_ref[object_id])
        return None



    def process(self, plot_collection: evo_plot.PlotCollection, results: Dict):
        # prepare results dict
        results["objects"] = {}
        for object_id, _, _ in common_entries(self._object_motions_traj, self._object_motions_traj_ref):
            results["objects"][object_id] = {}

        for object_id, _, _ in common_entries(self._object_poses_traj, self._object_poses_traj_ref):
            results["objects"][object_id] = {}

        self._process_motion_traj(plot_collection, results)
        self._process_pose_traj(plot_collection, results)


    def _calculate_objects_per_frame(self):
        assert self._object_poses_traj is not None

        # still asuming everything is by frame id!
        for object_id, object_traj in self._object_poses_traj.items():
            # for each 'timestamp
            for idx in object_traj.timestamps:
                idx = int(idx)
                if idx not in self._objects_at_frame:
                    self._objects_at_frame[idx] = []

                self._objects_at_frame[idx].append(object_id)


    def _process_motion_traj(self,plot_collection: evo_plot.PlotCollection, results: Dict):
        for object_id, object_traj, object_traj_ref in common_entries(self._object_motions_traj, self._object_motions_traj_ref):

            # motion errors in W
            absolute_motion_errors = self._compute_motion_in_W_errors(object_id, object_traj, object_traj_ref)
            # motion errors in L
            relatvive_motion_errors = self._compute_motion_in_L_errors(object_id, object_traj, plot_collection)


            # expect results to already have results["objects"][id] prepared
            if absolute_motion_errors:
                results["objects"][object_id]["motions_W"] = absolute_motion_errors

            if relatvive_motion_errors:
                results["objects"][object_id]["motions_L"] = relatvive_motion_errors


    def _compute_motion_in_W_errors(self, object_id, object_motion_traj, object_motion_traj_ref, plots: Optional[evo_plot.PlotCollection] = None):
        # only interested in APE as this matches our error metrics
        ape_trans = metrics.APE(metrics.PoseRelation.translation_part)
        ape_rot = metrics.APE(metrics.PoseRelation.rotation_angle_deg)

        data = (object_motion_traj_ref, object_motion_traj)
        ape_trans.process_data(data)
        ape_rot.process_data(data)

        results_per_object = {}
        results_per_object["ape_translation"] = ape_trans.get_all_statistics()
        results_per_object["ape_rotation"] = ape_rot.get_all_statistics()


        if plots:
            plots.add_figure(
                f"Object_Motion_W_translation_{object_id}",
                plotting.plot_metric(ape_trans, f"AME Translation Error: {object_id}", x_axis=object_motion_traj.timestamps)
            )

            plots.add_figure(
                f"Object_Motion_W_rotation_{object_id}",
                plotting.plot_metric(ape_rot, f"AME Rotation Error: {object_id}", x_axis=object_motion_traj.timestamps)
            )

            fig = plt.figure(figsize=(11,8))
            core.plotting.plot_ame_error(
                fig,
                object_motion_traj,
                object_motion_traj_ref,
                f"Object {object_id}"
            )
            plots.add_figure(
                f"Object Motion Error{object_id}",
                fig)

        return results_per_object

    def _compute_motion_in_L_errors(self, object_id, object_motion_traj, plots: Optional[evo_plot.PlotCollection] = None) -> Dict:
        if object_id not in self._object_poses_traj_ref:
            logger.warning(f"{object_id} not found for object pose ground truth. Skipping RME calculation")
            return None

        # ground truth object poses
        object_poses_ref_traj = self._object_poses_traj_ref[object_id]
        object_motion_traj_est = object_motion_traj

        # object_poses = object_poses_ref_traj.poses_se3

        # for object_pose_k_1, object_pose_k, object_motion_k in zip(object_poses[:-1], object_poses[1:], object_motion_traj.poses_se3[1:]):
        #     object_motion_L.append(lie_algebra.se3_inverse(object_pose_k) @ object_motion_k @ object_pose_k_1)
        #     object_motion_L_gt.append(lie_algebra.se3())

        # # for motion_L, motion_L_ref in zip(object_motion_L, object_motion_L_gt):
        # object_traj_in_L = trajectory.PoseTrajectory3D(poses_se3=np.array(object_motion_L), timestamps=object_motion_traj.timestamps[1:])
        # object_traj_in_L_ref = trajectory.PoseTrajectory3D(poses_se3=np.array(object_motion_L_gt), timestamps=object_motion_traj.timestamps[1:])

        # # only interested in APE as this matches our error metrics
        # ape_trans = metrics.APE(metrics.PoseRelation.translation_part)
        # ape_rot = metrics.APE(metrics.PoseRelation.rotation_angle_deg)

        print(object_poses_ref_traj)
        print(object_motion_traj_est)

        data = (object_poses_ref_traj, object_motion_traj_est)
        rme_trans = dyno_metrics.RME(metrics.PoseRelation.translation_part)
        rme_rot = dyno_metrics.RME(metrics.PoseRelation.rotation_angle_deg)

        rme_trans.process_data(data)
        rme_rot.process_data(data)

        results_per_object = {}
        results_per_object["ape_translation"] = rme_trans.get_all_statistics()
        results_per_object["ape_rotation"] = rme_rot.get_all_statistics()

        if plots is not None:
            plots.add_figure(
                f"Object_Motion_L_translation_{object_id}",
                plotting.plot_metric(rme_trans, f"RME Translation Error: {object_id}", x_axis=rme_trans.timestamps)
            )

            plots.add_figure(
                f"Object_Motion_L_rotation_{object_id}",
                plotting.plot_metric(rme_rot, f"RME Rotation Error: {object_id}", x_axis=rme_rot.timestamps)
            )

            rme_full = dyno_metrics.RME(metrics.PoseRelation.full_transformation)
            rme_full.process_data(data)

            fig = plt.figure(figsize=(11,8))
            core.plotting.plot_per_frame_error(
                fig,
                rme_full,
                f"RME Object {object_id}",
                frames=rme_full.timestamps
            )
            plots.add_figure(
                f"Object RME {object_id}",
                fig)

        return results_per_object

    def _process_pose_traj(self,plot_collection: evo_plot.PlotCollection, results: Dict):
        # est and ref object trajectories
        # this is used to draw the trajectories
        object_trajectories = {}
        object_trajectories_ref = {}
        object_trajectories_calibrated = {}

        fig_all_object_traj = plt.figure(figsize=(8,8))

        trajectory_helper = TrajectoryHelper()



        for object_id, object_traj, object_traj_ref in common_entries(self._object_poses_traj, self._object_poses_traj_ref):
            object_traj, object_traj_ref = tools.sync_and_align_trajectories(object_traj, object_traj_ref)
            data = (object_traj_ref, object_traj)

            object_traj.check()

            trajectory_helper.append(object_traj)

            # # TODO: align?
            object_trajectories[f"Object {object_id}"] = object_traj
            object_trajectories_ref[f"Ground Truth Object {object_id}"] = object_traj_ref

            logger.debug(f"Logging pose metrics for object {object_id}")

            ape_trans = metrics.APE(metrics.PoseRelation.translation_part)
            ape_rot = metrics.APE(metrics.PoseRelation.rotation_angle_deg)

            rpe_trans = metrics.RPE(metrics.PoseRelation.translation_part,
                            1.0, metrics.Unit.frames, 0.0, False)
            rpe_rot = metrics.RPE(metrics.PoseRelation.rotation_angle_deg,
                          1.0, metrics.Unit.frames, 1.0, False)

            ape_trans.process_data(data)
            ape_rot.process_data(data)

            rpe_trans.process_data(data)
            rpe_rot.process_data(data)

            results_per_object = {}
            results_per_object["ape_translation"] = ape_trans.get_all_statistics()
            results_per_object["ape_rotation"] = ape_rot.get_all_statistics()
            results_per_object["rpe_translation"] = rpe_trans.get_all_statistics()
            results_per_object["rpe_rotation"] = rpe_rot.get_all_statistics()


            calculate_omd_errors(object_traj, object_traj_ref, object_id)

            plot_collection.add_figure(
                    f"Object_Pose_RPE_translation_{object_id}",
                    plotting.plot_metric(rpe_trans, f"Object Pose RPE Translation: {object_id}", x_axis=object_traj.timestamps[1:])
                )

            plot_collection.add_figure(
                f"Object_Pose_APE_rotation_{object_id}",
                plotting.plot_metric(rpe_rot, f"Object Pose RPE Rotation: {object_id}", x_axis=object_traj.timestamps[1:])
            )


            # do reconstruction with relative pose
            reconstructed_object_traj = reconstruct_trajectory_from_relative(object_traj, object_traj_ref)
            reconstructed_data = (reconstructed_object_traj, object_traj_ref)

            object_trajectories_calibrated[f"Object {object_id}"] = reconstructed_object_traj

            rpe_trans_recon = metrics.RPE(metrics.PoseRelation.translation_part,
                            1.0, metrics.Unit.frames, 0.0, False)
            rpe_rot_recon = metrics.RPE(metrics.PoseRelation.rotation_angle_deg,
                          1.0, metrics.Unit.frames, 1.0, False)
            rpe_trans_recon.process_data(reconstructed_data)
            rpe_rot_recon.process_data(reconstructed_data)

            results_per_object["rpe_translation_reconstruction"] = rpe_trans_recon.get_all_statistics()
            results_per_object["rpe_rotation_reconstruction"] = rpe_rot_recon.get_all_statistics()

            # expect results to already have results["objects"][id]["poses"] prepared
            results["objects"][object_id]["poses"] = results_per_object

        # plot object poses
        plot_mode = evo_plot.PlotMode.xyz
        ax = fig_all_object_traj.add_subplot(111, projection="3d")
        ax = core.plotting.plot_object_trajectories(
            fig_all_object_traj,
            object_trajectories,
            object_trajectories_ref,
            plot_mode=plot_mode,
            plot_start_end_markers=True,
            plot_axis_est=False,
            plot_axis_ref=False,
            # axis_marker_scale=1.0,
            downscale=0.1)
        # ax.get_legend().remove()


        fig_all_object_traj.suptitle(r"Estimated \& Ground Truth Object Trajectories")
        ax = fig_all_object_traj.gca()
        # trajectory_helper.set_ax_limits(ax, plot_mode)
        fig_all_object_traj.tight_layout()

        # must happen after plot_object_trajectories becuase this is where we call the 'prepare axis'
        # evo_plot.draw_coordinate_axes(ax, object_trajectories[f"Object 2"], plot_mode=evo_plot.PlotMode.xyz, marker_scale=2.0)
        # evo_plot.draw_coordinate_axes(ax, object_trajectories_ref[f"Ground Truth Object 2"], plot_mode=evo_plot.PlotMode.xyz, marker_scale=2.0)

        # plot reconsructed (calibrated) object poses
        # fig_all_object_traj_calibrated = plt.figure(figsize=(8,8))
        # core.plotting.plot_object_trajectories(fig_all_object_traj_calibrated, object_trajectories_calibrated, object_trajectories_ref, plot_mode=evo_plot.PlotMode.xyz, plot_start_end_markers=True)
        # fig_all_object_traj_calibrated.suptitle("Obj Trajectories Calibrated")
        # ax = fig_all_object_traj_calibrated.gca()
        trajectory_helper.set_ax_limits(ax, evo_plot.PlotMode.xyz)

        plot_collection.add_figure(
            "Obj Trajectories", fig_all_object_traj
        )

        # plot_collection.add_figure(
        #     "Obj Trajectories Calibrated", fig_all_object_traj_calibrated
        # )



    @staticmethod
    def _construct_object_se3_trajectories(object_poses_log_file: csv.DictReader, convert_to_world_coordinates: bool) -> Tuple[ObjectTrajDict, ObjectTrajDict]:
        """
        Constructs dictionaries representing the object poses from the object_poses_log_file.
        The object_poses_log_file is expected to have a header with the form ["frame_id", "object_id", "x", "y", "z", "roll", "pitch", "yaw"].
        The function returnes two dictionaries with the poses per object in different forms.
        First return is an object pose dictionary in the form
        {
            object_id: { frame_id: T }
        }
        where T is a [4x4] homogenous matrix (np.ndarray).

        Second return is an dictionary of trajectory.PosePath3D (from the evo lib) per object:
        {
            object_id: trajectory.PosePath3D
        }

        Args:
            object_poses_log_file (csv.DictReader[str]): Input reader

        Raises:
            FileNotFoundError: _description_

        Returns:
            Tuple[ObjectPoseDict, ObjectTrajDict]: _description_
        """
        # must be floats if we want to align/sync
        timestamps = []

        # used to construct the pose trajectires with evo
        object_poses_tmp_dict = {}
        object_poses_ref_tmp_dict = {}

        for row in object_poses_log_file:
            frame_id = float(row["frame_id"])
            object_id = int(row["object_id"])

            if object_id not in object_poses_tmp_dict:
                object_poses_tmp_dict[object_id] = {"traj": [], "timestamps": []}

            if object_id not in object_poses_ref_tmp_dict:
                object_poses_ref_tmp_dict[object_id] = {"traj": [], "timestamps": []}

            T, T_ref = load_pose_from_row(row)

            object_poses_tmp_dict[object_id]["traj"].append(T)
            object_poses_tmp_dict[object_id]["timestamps"].append(frame_id)
            object_poses_ref_tmp_dict[object_id]["traj"].append(T_ref)
            object_poses_ref_tmp_dict[object_id]["timestamps"].append(frame_id)

        def sort_by_frame_order(object_T_dict: Dict):
            for object_id, data in object_T_dict.items():
                frames = data["timestamps"]
                traj = data["traj"]

                # # Sort frames and associated values
                sorted_indices = sorted(range(len(frames)), key=lambda i: frames[i])
                frames = [frames[i] for i in sorted_indices]
                traj = [traj[i] for i in sorted_indices]

                # Sort traj based on the order of frames
                # sorted_frames, sorted_traj = zip(*sorted(zip(frames, traj)))

                # # Update map data
                object_T_dict[object_id]["timestamps"] = frames
                object_T_dict[object_id]["traj"] = traj

            return object_T_dict

        # frameids may not be in order.
        object_poses_tmp_dict = sort_by_frame_order(object_poses_tmp_dict)
        object_poses_ref_tmp_dict = sort_by_frame_order(object_poses_ref_tmp_dict)



        # {object_id: trajectory.PosePath3D}
        object_poses_traj: ObjectTrajDict  = {}
        object_poses_traj_ref: ObjectTrajDict  = {}

        for object_id, est, ref in common_entries(object_poses_tmp_dict, object_poses_ref_tmp_dict):
            # ignore last one for evaluation

            poses_est = est["traj"][:-1]
            poses_ref = ref["traj"][:-1]

            timestamps = np.array(est["timestamps"][:-1])
            timestamps_ref = np.array(ref["timestamps"][:-1])
            # This will need the poses to be in order
            # assume est and ref are the same size
            if len(timestamps) < 3:
                continue

            # #hack for now to handle trajectories that jump
            # for index, (prev_t, curr_t) in enumerate(zip(timestamps[:-1], timestamps[1:])):
            #     if prev_t != curr_t - 1:
            #         break

            # print(f"{index} -> {len(timestamps)}")
            # timestamps = timestamps[:index-1]
            # timestamps_ref = timestamps_ref[:index-1]
            # poses_ref = poses_ref[:index-1]
            # poses_est = poses_est[:index-1]
            # print(len(poses_est))
            # if len(poses_est) < 2:
            #     continue


            if convert_to_world_coordinates:
                object_poses_traj[object_id] = transform_camera_trajectory_to_world(
                    trajectory.PoseTrajectory3D(poses_se3=np.array(poses_est), timestamps=timestamps))
                object_poses_traj_ref[object_id] = transform_camera_trajectory_to_world(
                    trajectory.PoseTrajectory3D(poses_se3=np.array(poses_ref), timestamps=timestamps_ref))
            else:
                object_poses_traj[object_id] =  trajectory.PoseTrajectory3D(poses_se3=np.array(poses_est), timestamps=timestamps)
                object_poses_traj_ref[object_id] = trajectory.PoseTrajectory3D(poses_se3=np.array(poses_ref), timestamps=timestamps_ref)



        return object_poses_traj, object_poses_traj_ref



class CameraPoseEvaluator(Evaluator):

    def __init__(self, camera_pose_log:str) -> None:
        self._camera_pose_log:str = camera_pose_log

        self._camera_pose_file = read_csv(self._camera_pose_log,
                                          ["frame_id",
                                           "tx", "ty", "tz", "qx", "qy", "qz", "qw",
                                            "gt_tx", "gt_ty", "gt_tz", "gt_qx", "gt_qy", "gt_qz", "gt_qw"])

        poses = []
        poses_ref = []

        timestamps = []

        for row in self._camera_pose_file:
            frame_id = float(row["frame_id"])
            timestamps.append(frame_id)

            T, T_ref = load_pose_from_row(row)

            poses.append(T)
            poses_ref.append(T_ref)

        self._camera_pose_traj = transform_camera_trajectory_to_world(
            trajectory.PoseTrajectory3D(poses_se3=poses, timestamps=timestamps))
        self._camera_pose_traj_ref = transform_camera_trajectory_to_world(
            trajectory.PoseTrajectory3D(poses_se3=poses_ref, timestamps=timestamps))

    @property
    def camera_pose_traj(self) -> trajectory.PoseTrajectory3D:
        return self._camera_pose_traj

    @property
    def camera_pose_traj_ref(self) -> trajectory.PoseTrajectory3D:
        return self._camera_pose_traj_ref


    def process(self, plot_collection: evo_plot.PlotCollection, results: Dict):
        fig_traj = plt.figure(figsize=(8,8))

        traj_est_vo = self.camera_pose_traj
        traj_ref_vo = self.camera_pose_traj_ref

        traj_est_vo, traj_ref_vo = tools.sync_and_align_trajectories(traj_est_vo, traj_ref_vo)

        # alignment_transformation = lie_algebra.sim3(
        #     *traj_est_vo.align(traj_ref_vo, False, False, n=-1))

        # used to draw trajectories for plot collection
        trajectories = {}
        trajectories["Estimated VO"] = traj_est_vo
        trajectories["Ground Truth VO"] = traj_ref_vo

        evo_plot.trajectories(fig_traj, trajectories, plot_mode=evo_plot.PlotMode.xyz, plot_start_end_markers=True)
        # evo_plot.draw_coordinate_axes(fig_traj.gca(), traj_est_vo, plot_mode=evo_plot.PlotMode.xyz)
        evo_plot.draw_correspondence_edges(fig_traj.gca(), traj_est_vo, traj_ref_vo, evo_plot.PlotMode.xyz)

        # ax_traj = evo_plot.prepare_axis(fig_traj, plot_mode)
        # evo_plot.traj(ax_traj, plot_mode, self._camera_pose_traj, color="")

        plot_collection.add_figure(
            "Camera Trajectory",
            fig_traj
        )

        data = (traj_ref_vo, traj_est_vo)

        ape_trans = metrics.APE(metrics.PoseRelation.translation_part)
        ape_rot = metrics.APE(metrics.PoseRelation.rotation_angle_deg)

        rpe_trans = metrics.RPE(metrics.PoseRelation.translation_part,
                        1.0, metrics.Unit.frames, 0.0, False)
        rpe_rot = metrics.RPE(metrics.PoseRelation.rotation_angle_deg,
                        1.0, metrics.Unit.frames, 1.0, False)

        ape_trans.process_data(data)
        ape_rot.process_data(data)
        rpe_trans.process_data(data)
        rpe_rot.process_data(data)


        plot_collection.add_figure(
                    "VO_APE_translation",
                    plotting.plot_metric(ape_trans, f"VO APE Translation", x_axis=traj_est_vo.timestamps)
                )

        plot_collection.add_figure(
                    "VO_APE_rotation",
                    plotting.plot_metric(ape_rot, f"VO APE Rotation", x_axis=traj_est_vo.timestamps)
                )

        plot_collection.add_figure(
                    "VO_RPE_translation",
                    plotting.plot_metric(rpe_trans, f"VO RPE Translation", x_axis=traj_est_vo.timestamps[1:])
                )

        plot_collection.add_figure(
                    "VO_RPE_rotation",
                    plotting.plot_metric(rpe_rot, f"VO RPE Rotation", x_axis=traj_est_vo.timestamps[1:])
                )

        # update results dict that will be saved to file
        results["vo"] = {}
        results["vo"]["ape_translation"] = ape_trans.get_all_statistics()
        results["vo"]["ape_rotation"] = ape_rot.get_all_statistics()
        results["vo"]["rpe_translation"] = rpe_trans.get_all_statistics()
        results["vo"]["rpe_rotation"] = rpe_rot.get_all_statistics()



class EgoObjectMotionEvaluator(Evaluator):

    def __init__(self, camera_eval: CameraPoseEvaluator, object_eval: MotionErrorEvaluator):
        self._camera_eval = camera_eval
        self._object_eval = object_eval

    def process(self, plot_collection: evo_plot.PlotCollection, results: Dict):
        fig_traj = plt.figure(figsize=(8,8))
        camera_traj = copy.deepcopy(self._camera_eval.camera_pose_traj)
        object_trajs = copy.deepcopy(self._object_eval.object_poses_traj)

        trajectory_helper = TrajectoryHelper()
        trajectory_helper.append(camera_traj)
        trajectory_helper.append(object_trajs)

        all_traj = object_trajs
        all_traj["Camera"] = camera_traj

        evo_plot.trajectories(fig_traj, all_traj, plot_mode=evo_plot.PlotMode.xyz)

        trajectory_helper.set_ax_limits(fig_traj.gca())

        plot_collection.add_figure(
            "Trajectories",
            fig_traj
        )

class MapPlotter3D(Evaluator):

    SEQUENTIAL_COLOUR_MAPS = ['Greens', 'OrRd', 'Purples', 'Blues', 'Oranges',
                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',  'Reds',
                      'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

    def __init__(self, map_points_csv_file_path: str, camera_eval: CameraPoseEvaluator, object_eval: MotionErrorEvaluator, **kwargs):
        self._camera_eval = camera_eval
        self._object_eval = object_eval
        self.kwargs = kwargs

        self._map_points_file = read_csv(
            map_points_csv_file_path,
            ["frame_id", "object_id", "tracklet_id", "x_world", "y_world", "z_world"]
        )

    def process(self, plot_collection: evo_plot.PlotCollection, results: Dict):
        print("Logging 3d points")
        self.plot_3d_map_points(plot_collection)

    def plot_3d_map_points(self, plot_collection: evo_plot.PlotCollection) -> Tuple[matplotlib.axes.Axes,matplotlib.figure.Figure]:
        map_fig = plt.figure(figsize=(8,14))
        # ax = evo_plot.prepare_axis(map_fig, evo_plot.PlotMode.xyz)
        ax = map_fig.add_subplot(111, projection="3d")
        ax.set_ylabel(r"Y(m)")
        ax.set_xlabel(r"X(m)")
        ax.set_zlabel(r"Z(m)")
        # ax = map_fig.add_subplot(111, projection="3d")

        plot_object_points = self.kwargs.get("plot_object_points", True)
        plot_static_points = self.kwargs.get("plot_static_points", True)
        plot_gt_objects = self.kwargs.get("plot_gt_objects", True)
        plot_gt_camera = self.kwargs.get("plot_gt_camera", True)
        downsample_static_cloud = self.kwargs.get("downsample_static_cloud", None)



        camera_traj = copy.deepcopy(self._camera_eval.camera_pose_traj)
        camera_traj_ref = copy.deepcopy(self._camera_eval.camera_pose_traj_ref)

        camera_ref_dict = None
        if plot_gt_camera:
            camera_ref_dict = {"Camera Ref": camera_traj_ref}

        core.plotting.plot_object_trajectories(map_fig,
                                       {"Camera":camera_traj},
                                       camera_ref_dict,
                                       plot_mode=evo_plot.PlotMode.xyz,
                                       colours=['blue'],
                                       plot_axis_est=True,
                                       plot_start_end_markers=False,
                                       axis_marker_scale=0.1,
                                       downscale=0.1,
                                       traj_zorder=30,
                                       traj_linewidth=3.0)


        import itertools

         # list for the object trajectory generator
        colour_list = []
        trajectory_helper = TrajectoryHelper()

        if self._object_eval is not None:

            object_trajs = copy.deepcopy(self._object_eval.object_poses_traj)


            # trajectory_helper.append(camera_traj)
            trajectory_helper.append(object_trajs)

            # do gross renamign
            all_traj = object_trajs
            all_gt_traj = copy.deepcopy(self._object_eval.object_poses_traj_ref)

            # map for the point cloud plot per object id
            colour_generator_map = {}

            plot_velocities = self.kwargs.get("plot_velocities", False)

            colour_map_names = itertools.cycle(MapPlotter3D.SEQUENTIAL_COLOUR_MAPS)
            for object_id, t in all_traj.items():
                id = int(object_id)
                # get colour map
                colour_map  = matplotlib.colormaps[next(colour_map_names)]

                trajectory_and_velocity_colour = colour_map(0.8)
                colour_list.append(trajectory_and_velocity_colour)
                colour_generator_map[id] = colour_map
                object_trajectory = self._object_eval.make_object_trajectory(object_id)

                if object_trajectory and plot_velocities:
                    core.plotting.plot_velocities(ax, object_trajectory, color=trajectory_and_velocity_colour)

            core.plotting.plot_object_trajectories(map_fig, all_traj,
                                       plot_mode=evo_plot.PlotMode.xyz,
                                       colours=colour_list,
                                       plot_axis_est=True,
                                       plot_start_end_markers=True,
                                       axis_marker_scale=0.1,
                                       traj_zorder=30,
                                       est_name_prefix="Object",
                                       traj_linewidth=3.0)
            if plot_gt_objects:
                core.plotting.plot_object_trajectories(map_fig, all_gt_traj,
                                            plot_mode=evo_plot.PlotMode.xyz,
                                            colours=colour_list,
                                            plot_axis_est=True,
                                            plot_start_end_markers=True,
                                            axis_marker_scale=0.1,
                                            traj_zorder=30,
                                            est_style="--",
                                            est_name_prefix="Object GT",
                                            traj_linewidth=3.0)


        static_points = []
        object_points = {}

        tracklet_set = set()

        for row in self._map_points_file:
            frame_id = float(row["frame_id"])
            object_id = int(row["object_id"])
            tracklet_id = int(row["tracklet_id"])


            if tracklet_id in tracklet_set:
                continue

            x_world = float(row["x_world"])
            y_world = float(row["y_world"])
            z_world = float(row["z_world"])

            # do lazy conversion from camera convention to world convention
            t_cam_convention = np.array([x_world, y_world, z_world, 1])
            transform =camera_coordinate_to_world()
            t_robot_convention = transform @ t_cam_convention

            if object_id == 0 and plot_static_points:
                static_points.append([t_robot_convention[0], t_robot_convention[1], t_robot_convention[2]])

                tracklet_set.add(tracklet_id)
            elif plot_object_points:
                # just draw last object
                # cont

                # # this might happen becuase we log ALL the points, even on objects we only see a few number of times
                if object_id not in object_trajs:
                    continue

                # # # since this takes AGES just skip every 10 frames
                # if int(frame_id) % 10 != 0:
                #     continue

                # object_trajectory = self._object_eval.make_object_trajectory(object_id)
                # #this worls for when the code output the frame id as the timestamp (which may change in future?)
                # object_trajectory_frames = object_trajs[object_id].timestamps

                # last_frame = object_trajectory_frames[-1]

                # print(f"frame {frame_id} last frame {last_frame}")

                # if frame_id < last_frame:
                #     k_H_last = lie_algebra.se3()

                #     # calculate motion that takes us from current frame to last frame
                #     for i in range(int(frame_id+1), last_frame):
                #         # motions need to be in robot convention!!
                #         _, motion = object_trajectory.get_motion_with_pose_current(i)
                #         assert motion is not None
                #         k_H_last = motion @ k_H_last


                #     # put points from frame K to last frame
                #     t_robot_convention = k_H_last @ t_robot_convention


                # if int(frame_id) == int(last_frame):
                    # get normalised timestamp in range 0-1
                    # normalised_frame_id = (frame_id - np.min(object_trajectory_frames))/(np.max(object_trajectory_frames) - np.min(object_trajectory_frames))
                    # print(frame_id)
                    # print(normalised_frame_id)
                    # time_dependant_colour = colour_generator_map[object_id](normalised_frame_id)

                if object_id not in object_points:
                    # x,y,z,colour
                    object_points[object_id] = [[], [], []]

                object_points[object_id][0].append(t_robot_convention[0])
                object_points[object_id][1].append(t_robot_convention[1])
                object_points[object_id][2].append(t_robot_convention[2])

                # print(f"Adding object point {t_robot_convention}: {object_id}")




        ax.view_init(azim=0, elev=90)
        ax.patch.set_facecolor('white')
        ax.axis('off')

        static_points = np.array(static_points)

        if downsample_static_cloud is not None:
            assert type(downsample_static_cloud) == float

            import open3d as o3d
            static_pc = o3d.geometry.PointCloud()
            static_pc.points = o3d.utility.Vector3dVector(static_points)

            logger.info(f"Downsampling static cloud with voxel size {downsample_static_cloud}")
            static_pc_down = static_pc.voxel_down_sample(voxel_size=downsample_static_cloud)

            static_points = np.asarray(static_pc_down.points)

        # static points
        # some of these params are after handtuning on particular datasets for pretty figures ;)
        ax.scatter(static_points[:,0], static_points[:,1], static_points[:,2], s=2.0, c='black',alpha=1.0, zorder=0, marker=".")
        for (_, data), object_colour in zip(object_points.items(), colour_list):
            ax.scatter(data[0], data[1], data[2], s=3.0, alpha=0.7, c=object_colour)


        trajectory_helper.set_ax_limits(map_fig.gca())
        map_fig.tight_layout()
        # plt.show()
        # ax.legend().set_visible(False)

        plot_collection.add_figure(self.kwargs.get("title", "Map"), map_fig)
        return ax, map_fig



class DatasetEvaluator:
    # args should be dictionary of keyword arguments. shoudl default be none?
    def __init__(self, output_folder_path:str, args = None) -> None:
        self._output_folder_path = output_folder_path
        self._args = args

    def run_analysis(self):
        logger.info("Running analysis using files at output path {}".format(self._output_folder_path))
        from .formatting_utils import LatexTableFormatter

        def run_data_file_analysis():
            possible_path_prefixes = self._search_for_datafiles()
            logger.info(f"Searching for datafiles using prefixes {possible_path_prefixes}")

            table_formatter = LatexTableFormatter()

            for prefixs in possible_path_prefixes:
                data_files = self.make_data_files(prefixs)
                data = self.run_and_save_single_analysis(data_files)

                if data is None:
                    continue

                plot_collection, results = data

                table_formatter.add_results(data_files.plot_collection_name, results)
                # right now just save metric plots per prefix
                plot_collection.export(
                    self._create_new_file_path(data_files.plot_collection_name + "_metrics.pdf"),
                    confirm_overwrite=False)

            table_formatter.save_pdf(self._create_new_file_path("result_tables"))

                # # plot_collection.show()
                # self._save_to_pdf(data_files.plot_collection_name, plot_collection, results)
                # # TEST
                # plot_collection.export(self._create_new_file_path(data_files.plot_collection_name + ".pdf"))

        def run_misc_analysis():
            plot_collection = evo_plot.PlotCollection("Statistics & mis")
            result_dict = {}

            misc_evaluator = self._check_and_cosntruct_generic_eval(
                MiscEvaluator,
                self._output_folder_path
            )
            misc_evaluator.process(plot_collection, result_dict)

            plot_collection.export(
                self._create_new_file_path("stats_misc.pdf"),
                confirm_overwrite=False)

        run_data_file_analysis()
        run_misc_analysis()


    def make_data_files(self, prefix:str) -> DataFiles:
        """
        Make a DataFiles object using the specified output folder path and the given prefix string

        Args:
            prefix (str): Datafile string prefix

        Returns:
            DataFiles: _description_
        """
        return DataFiles(prefix, self._output_folder_path)

    def create_motion_error_evaluator(self, data_files: DataFiles) -> MotionErrorEvaluator:
        object_pose_log_path = self.create_existing_file_path(data_files.object_pose_log)
        object_motion_log_path = self.create_existing_file_path(data_files.object_motion_log)

        return self._check_and_cosntruct_generic_eval(
            MotionErrorEvaluator,
            object_motion_log_path,
            object_pose_log_path
        )

    def create_camera_pose_evaluator(self, data_files: DataFiles) -> CameraPoseEvaluator:
        camera_pose_log_path = self.create_existing_file_path(data_files.camera_pose_log)
        return self._check_and_cosntruct_generic_eval(
            CameraPoseEvaluator,
            camera_pose_log_path,
        )

    def run_and_save_single_analysis(self, datafiles: DataFiles) -> Tuple[evo_plot.PlotCollection, Dict]:
        try:
            data = self._run_single_analysis(datafiles)
            result_dict = data[1]
            if len(result_dict) > 0:
                self._save_results_dict_json(datafiles.results_file_name, result_dict)
                logger.info(f"Successfully ran analysis for datafiles {datafiles} and saving results json")
            else:
                logger.info(f"Ran analysis for datafiles {datafiles} with no error but not saving results json")

            return data
        except Exception as e:
            import traceback
            logger.warning(f"Failed to run analysis for datafiles {datafiles}: error was \n{traceback.format_exc()}")
            return None

    def _run_single_analysis(self, datafiles: DataFiles) -> Tuple[evo_plot.PlotCollection, Dict]:
        prefix = datafiles.prefix

        #TODO: this does not work...?
        analysis_logger = logger.getChild(prefix)

        analysis_logger.info("Running analysis with datafile prefix: {}".format(prefix))
        map_points_log_path = self.create_existing_file_path(datafiles.map_point_log)

        evaluators = []
        motion_eval = self.create_motion_error_evaluator(datafiles)
        camera_pose_eval = self.create_camera_pose_evaluator(datafiles)

        if motion_eval:
            analysis_logger.info("Adding motion eval")
            evaluators.append(motion_eval)

        if camera_pose_eval:
            analysis_logger.info("Adding camera pose eval")
            evaluators.append(camera_pose_eval)

        if camera_pose_eval and motion_eval:
            analysis_logger.info("Adding EgoObjectMotionEvaluator")
            evaluators.append(EgoObjectMotionEvaluator(camera_pose_eval, motion_eval))

        #TODO: and note, we now do this in the plot_3d_map.py script
        # if map_points_log_path and camera_pose_eval and motion_eval:
        #     map_points_eval = self._check_and_cosntruct_generic_eval(
        #         MapPlotter3D,
        #         map_points_log_path,
        #         camera_pose_eval,
        #         motion_eval
        #     )
        #     if map_points_eval:
        #         analysis_logger.info("Adding Map points plotter")
        #         evaluators.append(map_points_eval)


        plot_collection_name = datafiles.plot_collection_name
        analysis_logger.info("Constructing plot collection {}".format(plot_collection_name))
        plot_collection = evo_plot.PlotCollection(plot_collection_name)

        result_dict = {}

        for evals  in evaluators:
            evals.process(plot_collection, result_dict)

        # plot_collection.show()
        return plot_collection, result_dict

    def _search_for_datafiles(self):
        from .filesystem_utils import search_for_results_prefixes
        return search_for_results_prefixes(self._output_folder_path)


    def _save_results_dict_json(self, file_name: str, results: Dict):
        output_file_name = self._create_new_file_path(file_name) + ".json"

        logger.info("Writing results json to file {}".format(output_file_name))

        import json

        with open(output_file_name, "w") as output_file:
            json.dump(results, output_file)

    def _create_new_file_path(self, file:str) -> str:
        return os.path.join(self._output_folder_path, file)

    def create_existing_file_path(self, file:str) -> Optional[str]:
        """
        Create a full file path at the output folder path using the input file name to an existing file path.
        Checks that a file exists at the path given

        Args:
            file (str): Input file name

        Returns:
            Optional[str]: Absolute file path or None if path does not exist
        """
        file_path = os.path.join(self._output_folder_path, file)

        if not os.path.exists(file_path):
            return None
        return file_path

    def _check_and_cosntruct_generic_eval(self, cls, *args):
        import inspect

        if not inspect.isclass(cls):
            logger.fatal("Argument cls ({}) shoudl be a class type and not an instance".format(cls))

        if not issubclass(cls, Evaluator):
            logger.fatal("Argument cls ({}) must derive from Evaluator!".format(cls.__name__))

        # for arg in args:
        #     if arg is None:
        #         logger.warning("Expected file path is none when constructing evaluator {} with args: {}".format(cls.__name__, args))
        #         return None
        #     assert isinstance(arg, str), "arg is not a string"


        logger.debug("Constructor evaluator: {}".format(cls.__name__))

        try:
            return cls(*args)
        except Exception as e:
            logger.error(f"Failed to load class {cls.__name__}: constructor failed with {str(e)}")
            return None

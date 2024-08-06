import csv
import os
import logging
from typing import Optional, List, Tuple, Dict, TypeAlias
from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from evo.core import lie_algebra, trajectory, metrics, transformations
import evo.tools.plot as evo_plot

from .tools import (
    so3_from_euler,
    common_entries,
    transform_camera_trajectory_to_world,
    TrajectoryHelper,
    load_pose_from_row,
    reconstruct_trajectory_from_relative,
    calculate_omd_errors,
    load_bson
)

import evaluation.tools as tools

from .result_types import (
    ObjectPoseDict,
    ObjectTrajDict,
    plot_metric,
    sync_and_align_trajectories
)

logger = logging.getLogger("dynosam_eval")
# ----> console info messages require these lines <----
# create console handler and set level to debug
_ch = logging.StreamHandler()
_ch.setLevel(logging.DEBUG)
_ch.setFormatter(logging.Formatter(
    fmt='[%(levelname)s][%(filename)s:%(lineno)d] %(name)s: %(message)s'))
# add ch to logger
logger.addHandler(_ch)




def read_csv(csv_file_path:str, expected_header: List[str]):
    assert csv_file_path is not None
    csvfile = open(csv_file_path)
    reader = csv.DictReader(csvfile)

    try:
        header = next(reader)
        keys = list(header.keys())
        if keys != expected_header:
            raise Exception(
                "Csv file headers were not valid when loading file at path: {}. "
                "Expected header was {} but actual keys were {}".format(
                    csv_file_path, expected_header, keys
                )
            )

        return reader
    except Exception as e:
        raise Exception(f"Failed to read csv file {csv_file_path}. Exception raised was {str(e)}")



class DataFiles:
    def __init__(self, prefix:str, output_folder_path: str, **kwargs) -> None:
        # These files will match the output logger file names from Logger.cc
        # as the prefix here is the prefix used in the logger
        self.prefix = prefix
        self.output_folder_path = output_folder_path
        self.results_file_name = kwargs.get("results_file_name", self.prefix + "_results")
        self.plot_collection_name = kwargs.get("plot_collection_name", self.prefix.capitalize())

    @property
    def object_pose_log(self):
        return self.prefix + "_object_pose_log.csv"

    @property
    def object_motion_log(self):
        return self.prefix + "_object_motion_log.csv"

    @property
    def camera_pose_log(self):
        return self.prefix + "_camera_pose_log.csv"

    @property
    def map_point_log(self):
        return self.prefix + "_map_points_log.csv"

    def __str__(self):
        return "DataFiles [\n\tprefix: {}\n\tresult file name: {}\n\tplot collection name: {}".format(
            self.prefix,
            self.results_file_name,
            self.plot_collection_name
        )

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


        # TODO: dont need dynosam to do the error metrics for us - let evo do it all, just record all the se3 things
        # TODO: need to make traj's a property trajectory by using the frame id as timestamp - this allows us to synchronize
        # {object_id: {frame_id : T}} -> T is a homogenous, and {object_id: trajectory.PosePath3D}
        self._object_poses_traj, self._object_poses_traj_ref = MotionErrorEvaluator._construct_object_se3_trajectories(self._object_pose_log_file, convert_to_world_coordinates=True)
        self._object_motions_traj, self._object_motions_traj_ref = MotionErrorEvaluator._construct_object_se3_trajectories(self._object_motion_log_file, convert_to_world_coordinates=False)

        # self._object_pose_error_dict: ObjectPoseErrorDict = MotionErrorEvaluator._construct_pose_error_dict( self._object_pose_error_log_file)



    @property
    def object_poses_traj(self) -> ObjectTrajDict:
        return self._object_poses_traj

    @property
    def object_poses_traj_ref(self) -> ObjectTrajDict:
        return self._object_poses_traj_ref


    def process(self, plot_collection: evo_plot.PlotCollection, results: Dict):
        # prepare results dict
        results["objects"] = {}
        for object_id, _, _ in common_entries(self._object_motions_traj, self._object_motions_traj_ref):
            results["objects"][object_id] = {}

        for object_id, _, _ in common_entries(self._object_poses_traj, self._object_poses_traj_ref):
            results["objects"][object_id] = {}

        self._process_motion_traj(plot_collection, results)
        self._process_pose_traj(plot_collection, results)


    def _process_motion_traj(self,plot_collection: evo_plot.PlotCollection, results: Dict):
        for object_id, object_traj, object_traj_ref in common_entries(self._object_motions_traj, self._object_motions_traj_ref):

            # no need to sync and align motions since they are all relative anyway
            # if anything we should put them into the gt object frame via a PCG(3) object change
            # to ensure the estimation is in the gt frame from which the original error was calculated!!
            data = (object_traj_ref, object_traj)

            # only interested in APE as this matches our error metrics
            ape_trans = metrics.APE(metrics.PoseRelation.translation_part)
            ape_rot = metrics.APE(metrics.PoseRelation.rotation_angle_deg)

            ape_trans.process_data(data)
            ape_rot.process_data(data)

            results_per_object = {}
            results_per_object["ape_translation"] = ape_trans.get_all_statistics()
            results_per_object["ape_rotation"] = ape_rot.get_all_statistics()

            # expect results to already have results["objects"][id]["motions"] prepared
            results["objects"][object_id]["motions"] = results_per_object

            plot_collection.add_figure(
                    f"Object_Motion_translation_{object_id}",
                    plot_metric(ape_trans, f"Object Motion Translation Error: {object_id}", x_axis=object_traj.timestamps)
                )

            plot_collection.add_figure(
                f"Object_Motion_rotation_{object_id}",
                plot_metric(ape_rot, f"Object Motion Rotation Error: {object_id}", x_axis=object_traj.timestamps)
            )

    def _process_pose_traj(self,plot_collection: evo_plot.PlotCollection, results: Dict):
        # est and ref object trajectories
        # this is used to draw the trajectories
        object_trajectories = {}
        object_trajectories_ref = {}
        object_trajectories_calibrated = {}

        fig_all_object_traj = plt.figure(figsize=(8,8))

        trajectory_helper = TrajectoryHelper()



        for object_id, object_traj, object_traj_ref in common_entries(self._object_poses_traj, self._object_poses_traj_ref):

            object_traj, object_traj_ref = sync_and_align_trajectories(object_traj, object_traj_ref)
            data = (object_traj_ref, object_traj)

            # add reference edges
            # TODO: doesnt seem to work?
            object_traj.check()

            # evo_plot.draw_correspondence_edges(fig_all_object_traj.gca(), object_traj, object_traj_ref, evo_plot.PlotMode.xyz)

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
                    plot_metric(rpe_trans, f"Object Pose RPE Translation: {object_id}", x_axis=object_traj.timestamps[1:])
                )

            plot_collection.add_figure(
                f"Object_Pose_APE_rotation_{object_id}",
                plot_metric(rpe_rot, f"Object Pose RPE Rotation: {object_id}", x_axis=object_traj.timestamps[1:])
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
        tools.plot_object_trajectories(fig_all_object_traj, object_trajectories, object_trajectories_ref, plot_mode=evo_plot.PlotMode.xyz, plot_start_end_markers=True)
        ax = fig_all_object_traj.gca()
        trajectory_helper.set_ax_limits(ax)

        # must happen after plot_object_trajectories becuase this is where we call the 'prepare axis'
        # evo_plot.draw_coordinate_axes(ax, object_trajectories[f"Object 2"], plot_mode=evo_plot.PlotMode.xyz, marker_scale=2.0)
        # evo_plot.draw_coordinate_axes(ax, object_trajectories_ref[f"Ground Truth Object 2"], plot_mode=evo_plot.PlotMode.xyz, marker_scale=2.0)


        # plot reconsructed (calibrated) object poses
        fig_all_object_traj_calibrated = plt.figure(figsize=(8,8))
        tools.plot_object_trajectories(fig_all_object_traj_calibrated, object_trajectories_calibrated, object_trajectories_ref, plot_mode=evo_plot.PlotMode.xyz, plot_start_end_markers=True)
        fig_all_object_traj_calibrated.suptitle("Obj Trajectories Calibrated")
        ax = fig_all_object_traj_calibrated.gca()
        trajectory_helper.set_ax_limits(ax)

        plot_collection.add_figure(
            "Obj Trajectories", fig_all_object_traj
        )

        plot_collection.add_figure(
            "Obj Trajectories Calibrated", fig_all_object_traj_calibrated
        )

    @staticmethod
    def _construct_object_se3_trajectories(object_poses_log_file: csv.DictReader, convert_to_world_coordinates: bool) -> Tuple[ObjectPoseDict, ObjectTrajDict]:
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

        # assume frames are ordered!!!
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
            if len(poses_est) < 4:
                continue

            if convert_to_world_coordinates:
                object_poses_traj[object_id] = transform_camera_trajectory_to_world(
                    trajectory.PoseTrajectory3D(poses_se3=poses_est, timestamps=timestamps))
                object_poses_traj_ref[object_id] = transform_camera_trajectory_to_world(
                    trajectory.PoseTrajectory3D(poses_se3=poses_ref, timestamps=timestamps_ref))
            else:
                object_poses_traj[object_id] =  trajectory.PoseTrajectory3D(poses_se3=poses_est, timestamps=timestamps)
                object_poses_traj_ref[object_id] = trajectory.PoseTrajectory3D(poses_se3=poses_ref, timestamps=timestamps_ref)

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

        traj_est_vo, traj_ref_vo = sync_and_align_trajectories(traj_est_vo, traj_ref_vo)

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
                    plot_metric(ape_trans, f"VO APE Translation", x_axis=traj_est_vo.timestamps)
                )

        plot_collection.add_figure(
                    "VO_APE_rotation",
                    plot_metric(ape_rot, f"VO APE Rotation", x_axis=traj_est_vo.timestamps)
                )

        plot_collection.add_figure(
                    "VO_RPE_translation",
                    plot_metric(rpe_trans, f"VO RPE Translation", x_axis=traj_est_vo.timestamps[1:])
                )

        plot_collection.add_figure(
                    "VO_RPE_rotation",
                    plot_metric(rpe_rot, f"VO RPE Rotation", x_axis=traj_est_vo.timestamps[1:])
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
        camera_traj = self._camera_eval.camera_pose_traj
        object_trajs = self._object_eval.object_poses_traj

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



class DatasetEvaluator:
    def __init__(self, output_folder_path:str, args) -> None:
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
                data_files = DataFiles(prefixs, self._output_folder_path)
                data = self.run_and_save_single_analysis(data_files)

                if data is None:
                    continue

                plot_collection, results = data

                table_formatter.add_results(data_files.plot_collection_name, results)
                # right now just save metric plots per prefix
                plot_collection.export(
                    self._create_new_file_path(data_files.plot_collection_name + "_metrics.pdf"),
                    confirm_overwrite=False)

                # plot_collection.show()

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
        object_pose_log_path = self._create_existing_file_path(datafiles.object_pose_log)
        object_motion_log_path = self._create_existing_file_path(datafiles.object_motion_log)
        camera_pose_log_path = self._create_existing_file_path(datafiles.camera_pose_log)

        evaluators = []
        motion_eval = self._check_and_cosntruct_generic_eval(
            MotionErrorEvaluator,
            object_motion_log_path,
            object_pose_log_path
        )

        camera_pose_eval = self._check_and_cosntruct_generic_eval(
            CameraPoseEvaluator,
            camera_pose_log_path,
        )

        if motion_eval:
            analysis_logger.info("Adding motion eval")
            evaluators.append(motion_eval)

        if camera_pose_eval:
            analysis_logger.info("Adding camera pose eval")
            evaluators.append(camera_pose_eval)

        if camera_pose_eval and motion_eval:
            analysis_logger.info("Adding EgoObjectMotionEvaluator")
            evaluators.append(EgoObjectMotionEvaluator(camera_pose_eval, motion_eval))

        plot_collection_name = datafiles.plot_collection_name
        analysis_logger.info("Constructing plot collection {}".format(plot_collection_name))
        plot_collection = evo_plot.PlotCollection(plot_collection_name)

        result_dict = {}

        for evals  in evaluators:
            evals.process(plot_collection, result_dict)

        # plot_collection.show()
        return plot_collection, result_dict

    def _search_for_datafiles(self):
        from pathlib import Path
        output_folder_path = Path(self._output_folder_path)

        path_prefix_set = set()

        dummy_data_files = DataFiles("","")

        def get_datafile_properties(data_files: DataFiles):
            # get functions relating to file paths of expected output logs
            # note: we dont look at ALL the possible properties of DataFiles, just the main ones
            return [
                data_files.camera_pose_log,
                data_files.object_motion_log,
                data_files.object_pose_log
            ]


        # these strings will return the names of the output log files, the suffix of which
        # we want to find in the output_folder_path
        data_files = get_datafile_properties(dummy_data_files)

        for p in output_folder_path.iterdir():
            # p is absolute path
            if p.is_file():
                found_file_name = p.name
                # expect file name of loggers to be suffixed with the output of data_file_property_functons
                for expected_log_suffx_name in data_files:
                    # found file name should be in the form prefix_suffix
                    # e.g. rgbd_backend_object_pose_log.csv
                    # where rgbd_backend is the prefix_we want to find
                    # and _object_pose_log is a known suffux to a log file
                    if found_file_name.endswith(expected_log_suffx_name):
                        possible_prefix = found_file_name.removesuffix(expected_log_suffx_name)
                        path_prefix_set.add(possible_prefix)
        return list(path_prefix_set)


    def _save_results_dict_json(self, file_name: str, results: Dict):
        output_file_name = self._create_new_file_path(file_name) + ".json"

        logger.info("Writing results json to file {}".format(output_file_name))

        import json

        with open(output_file_name, "w") as output_file:
            json.dump(results, output_file)

    def _create_new_file_path(self, file:str) -> str:
        return os.path.join(self._output_folder_path, file)

    def _create_existing_file_path(self, file:str) -> Optional[str]:
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
        return cls(*args)

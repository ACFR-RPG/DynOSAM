from evo.core.metrics import PE, RPE, APE, PoseRelation, MetricsException
import evo.core.lie_algebra as lie
import evo.core.trajectory as traj
import evo.core.sync as sync

import numpy as np
import copy
import typing

from enum import Enum, unique

TrajPair = typing.Tuple[traj.PoseTrajectory3D, traj.PoseTrajectory3D]


def reduce_to_largest_strictly_ascending_range(traj: traj.PoseTrajectory3D):
    timestamps = list(traj.timestamps)
    if not timestamps:
        return None, None  # Handle empty list case

    max_start, max_end = 0, 0
    start, end = 0, 0

    for i in range(1, len(timestamps)):
        if timestamps[i] > timestamps[i - 1]:
            end = i
        else:
            # Check if the current range is the largest so far
            if (end - start) > (max_end - max_start):
                max_start, max_end = start, end
            # Reset start and end for the next potential range
            start, end = i, i

    # Final check in case the longest range is at the end of the list
    if (end - start) > (max_end - max_start):
        max_start, max_end = start, end

    reduced_traj = copy.deepcopy(traj)
    reduced_traj.reduce_to_time_range(max_start, max_end)
    return reduced_traj


@unique
class MetricType(Enum):
    ame = "ame"
    rme = "rme"
    rpe = "rpe"


class AME(APE):
    def __init__(self,
                 pose_relation: PoseRelation = PoseRelation.translation_part):
        super().__init__(pose_relation)

# In the paper this is defined as Motion Error (ME)
class RME(APE):
    def __init__(self,
                 pose_relation: PoseRelation = PoseRelation.translation_part):
        super().__init__(pose_relation)
        # timestamp data after processing (this will be k to N when the size of the timestamps matchs the size of the error array)
        self.timestamps = np.array([])

    def process_data(self, data: TrajPair) -> None:
        """
        Calcualtes RME  on a motion trajectory and associated ground truth pose trajectory
        :param data: tuple (traj_pose_ref, traj_motion_est) with:
        traj_pose_ref: reference evo.trajectory.PosePath or derived and is the Pose associated with the trajectory
        traj_motion_est: estimated evo.trajectory.PosePath or derived and is the Motion associated with the trajectory
        """
        if len(data) != 2:
            raise MetricsException(
                "please provide data tuple as: (traj_pose_ref, traj_motion_est)")
        traj_pose_k_1_ref, traj_pose_k_ref, traj_motion_est = RME.sync_object_motion_and_pose(data)

        def traj_as_tuple(traj: traj.PoseTrajectory3D):
            timestamps = traj.timestamps
            poses = traj.poses_se3
            return zip(timestamps, poses)



        # we have designed the sync_object_motion_and_pose to return the pose data in the from k-1 to N and the motion data k to N
        object_motion_L = []
        object_motion_L_gt = []
        for pose_tuple_k_1, pose_tuple_k, motion_tuple_k in zip(traj_as_tuple(traj_pose_k_1_ref), traj_as_tuple(traj_pose_k_ref), traj_as_tuple(traj_motion_est)):

            pose_timestamp_k_1, object_pose_k_1 = pose_tuple_k_1
            pose_timestamp_k, object_pose_k = pose_tuple_k
            motion_timestamp_k, object_motion_k = motion_tuple_k

            assert motion_timestamp_k == pose_timestamp_k
            assert pose_timestamp_k_1 == motion_timestamp_k - 1

            object_motion_L.append(lie.se3_inverse(object_pose_k) @ object_motion_k @ object_pose_k_1)
            object_motion_L_gt.append(lie.se3())

        # update internal timestamps
        # -1 on timestamps so that the size matches the size of the motion error (ie. k...N)
        self.timestamps = np.array(traj_motion_est.timestamps)

        object_traj_in_L = traj.PoseTrajectory3D(poses_se3=np.array(object_motion_L), timestamps=self.timestamps)
        object_traj_in_L_ref = traj.PoseTrajectory3D(poses_se3=np.array(object_motion_L_gt), timestamps=self.timestamps)

        processed_data = (object_traj_in_L_ref, object_traj_in_L)

        super().process_data(processed_data)

        assert self.error.shape == self.timestamps.shape, ( self.error.shape, self.timestamps.shape)


    @staticmethod
    def sync_object_motion_and_pose(data: TrajPair, max_diff = 0.01):
        # sync so that we get object motions from i to N
        # and object poses from i-1 to N (inclusive)
        traj_pose_ref, traj_motion_est = data

        traj_pose_ref_sync, traj_motion_est_sync = sync.associate_trajectories(traj_pose_ref, traj_motion_est,max_diff=max_diff)

        pose_timestamps = traj_pose_ref_sync.timestamps
        motion_timestamps = traj_motion_est_sync.timestamps

        poses = traj_pose_ref_sync.poses_se3
        motions = traj_motion_est_sync.poses_se3

        traj_pose_ref_tuple = list(zip(pose_timestamps, poses))
        traj_motion_est_tuple = list(zip(motion_timestamps, motions))

        assert len(traj_pose_ref_tuple) == len(traj_motion_est_tuple)

        traj_pose_t_k_indices = set()
        traj_pose_t_k_1_indices = set() # this is also the motion times
        # assumes timestamps are actually FRAMES!!!!!
        for index, (motion_tuple, pose_tuple) in enumerate(zip(traj_motion_est_tuple, traj_pose_ref_tuple)):
            if index == 0:
                continue
            else:
                prev_index = index - 1
                assert prev_index >= 0
                prev_pose_frame = traj_pose_ref_tuple[prev_index][0]
                curr_pose_frame = pose_tuple[0]
                curr_motion_frame = motion_tuple[0]

                assert curr_pose_frame == curr_motion_frame

                # print(f"prev_pose_frame {prev_pose_frame} curr_motion_frame {curr_motion_frame} ")

                # is previous and in current frame id
                if prev_pose_frame == curr_motion_frame - 1:
                    traj_pose_t_k_indices.add(index)
                    traj_pose_t_k_1_indices.add(prev_index)

        # remove duplicates based on frameid
        traj_pose_t_k_indices = list(traj_pose_t_k_indices)
        traj_pose_t_k_1_indices = list(traj_pose_t_k_1_indices)

        # reduce to the acutal id's we want
        traj_pose_ref_sync_k = copy.deepcopy(traj_pose_ref_sync)
        traj_pose_ref_sync_k.reduce_to_ids(traj_pose_t_k_indices)

        traj_pose_ref_sync_k_1 = copy.deepcopy(traj_pose_ref_sync)
        traj_pose_ref_sync_k_1.reduce_to_ids(traj_pose_t_k_1_indices)

        traj_motion_est_sync = copy.deepcopy(traj_motion_est_sync)
        traj_motion_est_sync.reduce_to_ids(traj_pose_t_k_indices)

        assert traj_pose_ref_sync_k_1.num_poses == traj_pose_ref_sync_k.num_poses == traj_motion_est_sync.num_poses


        return (traj_pose_ref_sync_k_1, traj_pose_ref_sync_k, traj_motion_est_sync)

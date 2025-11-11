from __future__ import annotations


import evo.core
import evo.core.geometry
import evo.core.metrics
import evo.core.transformations
import numpy as np
from matplotlib.figure import Figure
from evo.core.lie_algebra import se3
from matplotlib.axes import Axes
import gtsam

from dynosam_utils.evaluation.formatting_utils import *
from dynosam_utils.evaluation.tools import ObjectMotionTrajectory
import dynosam_utils.evaluation.core.metrics as dynosam_metrics


# apparently this ordereed needed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



import evo.tools.plot as evo_plot
import evo.core.trajectory as evo_trajectory
from evo.core.metrics import PE

import evo

from typing import Tuple, Optional, List, Any, Iterable

from copy import deepcopy
import typing
import math

# Nic Barbara
def startup_plotting(font_size=14, line_width=1.5, output_dpi=600, tex_backend=True):
    """Edited from https://github.com/nackjaylor/formatting_tips-tricks/
    """
    # plt.rcdefaults()
    if tex_backend:
        try:
            plt.rcParams.update({
                    "text.usetex": True,
                    "font.family": "serif",
                    "font.serif": ["Computer Modern Roman"],
                    "text.latex.unicode": True
                    })
        except:
            print("WARNING: LaTeX backend not configured properly. Not using.")
            plt.rcParams.update({"font.family": "serif",
                    "font.serif": ["Computer Modern Roman"],
                        })

    # Default settings
    plt.rcParams.update({
        "lines.linewidth": line_width,

        "axes.grid" : True,
        "axes.grid.which": "major",
        "axes.linewidth": 0.5,
        "axes.prop_cycle": cycler("color", prop_cycle()),

        "errorbar.capsize": 2.5,

        "grid.linewidth": 0.25,
        "grid.alpha": 0.5,

        "legend.framealpha": 0.7,
        "legend.edgecolor": [1,1,1],

        "savefig.dpi": output_dpi,
        "savefig.format": 'pdf'
    })

    # Change default font sizes.
    plt.rc('font', size=font_size)
    plt.rc('axes', titlesize=font_size)
    plt.rc('axes', labelsize=font_size)
    plt.rc('xtick', labelsize=0.8*font_size)
    plt.rc('ytick', labelsize=0.8*font_size)
    plt.rc('legend', fontsize=0.8*font_size)


def plot_metric(metric: PE, plot_title="", figsize=(8,8), fig = None, x_axis=None):
    """ Adds a metric plot to a plot collection.

        Args:
            plot_collection: a PlotCollection containing plots.
            metric: an evo.core.metric object with statistics and information.
            plot_title: a string representing the title of the plot.
            figsize: a 2-tuple representing the figure size.

        Returns:
            A plt figure.
    """
    if not fig:
        fig = plt.figure(figsize=figsize)
    stats = metric.get_all_statistics()

    ax = fig.gca()

    # remove from stats
    del stats["rmse"]
    del stats["sse"]

    from evo.tools import plot
    plot.error_array(ax, metric.error,
                        x_array=x_axis,
                        statistics=stats,
                        title=plot_title,
                        xlabel="Frame Index [-]",
                        ylabel=plot_title + " " + metric.unit.value)

    return fig


def plot_traj(ax: Axes, plot_mode: evo_plot.PlotMode, traj: evo_trajectory.PosePath3D,
         style: str = '-', color='black', label: str = "", alpha: float = 1.0,
         plot_start_end_markers: bool = False, **kwargs) -> None:
    """
    plot a path/trajectory based on xyz coordinates into an axis.
    Modification of evo.plot_traj, allowing kwargs to be parsed to ax.plot.

    :param ax: the matplotlib axis
    :param plot_mode: PlotMode
    :param traj: trajectory.PosePath3D or trajectory.PoseTrajectory3D object
    :param style: matplotlib line style
    :param color: matplotlib color
    :param label: label (for legend)
    :param alpha: alpha value for transparency
    :param plot_start_end_markers: Mark the start and end of a trajectory
                                   with a symbol.
    :param **kwargs: kwargs parsed to ax.plot
    """
    from evo.tools.settings import SETTINGS
    x_idx, y_idx, z_idx = evo_plot.plot_mode_to_idx(plot_mode)

    x = traj.positions_xyz[:, x_idx]
    y = traj.positions_xyz[:, y_idx]
    if plot_mode == evo_plot.PlotMode.xyz:
        z = traj.positions_xyz[:, z_idx]
        ax.plot(x, y, z, style, color=color, label=label, alpha=alpha, **kwargs)
    else:
        ax.plot(x, y, style, color=color, label=label, alpha=alpha, **kwargs)
    if SETTINGS.plot_xyz_realistic:
        evo_plot.set_aspect_equal(ax)
    if label and SETTINGS.plot_show_legend:
        ax.legend(frameon=True)
    if plot_start_end_markers:
        evo_plot.add_start_end_markers(ax, plot_mode, traj, start_color=color,
                              end_color=color, alpha=alpha)



class ObjectTrajectoryPlotter(object):

    def __init__(self) -> None:
        # internal colours to use. Some complex logic here but is updated everytime plot is used
        # will eventually be a iterable of colours which can be in any valid plt colour form
        self._colours_itr: Iterable[Any] = None
        # the original colour list, over which the _colours_itr operates. Used to reset the colours
        self._colours = None

    def plot(self,
             fig: Figure,
            obj_trajectories: typing.Dict[str, evo_trajectory.PosePath3D],
            obj_trajectories_ref: Optional[typing.Dict[str, evo_trajectory.PosePath3D]] = None,
            plot_mode=evo_plot.PlotMode.xy,
            title: str = "",
            subplot_arg: int = 111,
            plot_start_end_markers: bool = False,
            length_unit: evo_plot.Unit = evo_plot.Unit.meters,
            **kwargs) -> Axes:

        if obj_trajectories_ref is not None and len(obj_trajectories) != len(obj_trajectories_ref):
            raise evo_plot.PlotException(f"Expected trajectories and ref trajectories to have the same length {len(obj_trajectories)} != {len(obj_trajectories_ref)}")

        if len(obj_trajectories) == 0:
            return ax

        # get all top level params
        plot_axis_est = kwargs.pop("plot_axis_est", False)
        plot_axis_ref = kwargs.pop("plot_axis_ref", False)
        # downscale as a percentage - how many poses to use when plotting anything over the top of the tajectories
        downscale = kwargs.pop("downscale", 0.1)
        axis_marker_scale = kwargs.pop("axis_marker_scale", 0.1)

        ax = None
        if len(fig.axes) == 0:
            ax = evo_plot.prepare_axis(fig, plot_mode, subplot_arg, length_unit)
        else:
            # use existing axis TODO: check 3d?
            ax = fig.gca()
        if title:
            ax.set_title(title)

        self._set_colours(obj_trajectories, kwargs.get("colours"))

        self._draw_trajectory(
            ax,
            obj_trajectories,
            plot_mode,
            plot_start_end_markers,
            style=kwargs.pop("est_style", '-'),
            shift_color=kwargs.pop("shift_est_colour", None),
            name_prefix=kwargs.pop("est_name_prefix", ""),
            **kwargs)
        if plot_axis_est:
            ObjectTrajectoryPlotter.plot_coordinate_axis(
                ax,
                obj_trajectories,
                plot_mode,
                downscale,
                axis_marker_scale
            )

        self._reset_colours()

        if obj_trajectories_ref is not None:
            self._draw_trajectory(
                ax,
                obj_trajectories_ref,
                plot_mode,
                plot_start_end_markers,
                style=kwargs.pop("ref_style", '--'),
                alpha=0.8,
                shift_color=kwargs.pop("shift_ref_colour", None),
                name_prefix=kwargs.pop("ref_name_prefix", ""),
                **kwargs)

            if plot_axis_ref:
                ObjectTrajectoryPlotter.plot_coordinate_axis(
                    ax,
                    obj_trajectories_ref,
                    plot_mode,
                    downscale,
                    axis_marker_scale
                )
        return ax


    @staticmethod
    def reduce_trajectory(downscale_percentage: float, traj:evo_trajectory.PosePath3D):
        assert downscale_percentage >= 0.0 and downscale_percentage <= 1.0

        reduced_trajectory = deepcopy(traj)

        reduced_pose_num = downscale_percentage * reduced_trajectory.num_poses
        # round UP to the nearest int (so we at least get 1 pose
        reduced_pose_num = int(math.ceil(reduced_pose_num))
        if reduced_pose_num >= 2:
            reduced_trajectory.downsample(reduced_pose_num)
        return reduced_trajectory

    @staticmethod
    def plot_coordinate_axis(
            ax: Axes,
            trajectories: typing.Dict[str, evo_trajectory.PosePath3D],
            plot_mode: evo_plot.PlotMode,
            downscale_percentage:float,
            axis_marker_scale:float):
        """
        Plot coordinate axes on a map of object trajectories.
        This is down by downscaling the numner of poses over which the coordiante axes will
        be drawn for visability.

        Args:
            trajectories (typing.Dict[str, evo_trajectory.PosePath3D]): _description_
        """
        reduced_trajectories = deepcopy(trajectories)
        for _, obj_traj in reduced_trajectories.items():
            obj_traj = ObjectTrajectoryPlotter.reduce_trajectory(downscale_percentage, obj_traj)
            evo_plot.draw_coordinate_axes(
                ax, obj_traj, plot_mode,marker_scale=axis_marker_scale)

    def _draw_trajectory(self,
                         ax: Axes,
                         traj: typing.Any,
                         plot_mode: evo_plot.PlotMode,
                         plot_start_end_markers: bool,
                         **kwargs):
        if isinstance(traj, evo_trajectory.PosePath3D):
            self._draw_impl(ax, traj, plot_mode, plot_start_end_markers, **kwargs)
        elif isinstance(traj, dict):
            for name, t in traj.items():
                if "name_prefix" in kwargs:
                    name = kwargs.get("name_prefix") + " " + str(name)

                kwargs.update({"name":name})
                self._draw_impl(ax,t, plot_mode, plot_start_end_markers,**kwargs)
        else:
            for t in traj:
                self._draw_impl(ax, t, plot_mode, plot_start_end_markers, **kwargs)

    def _draw_impl(self,
                   ax: Axes,
                   traj: typing.Any,
                   plot_mode: evo_plot.PlotMode,
                   plot_start_end_markers: bool,
                   **kwargs):
        color = self._get_next_colour()

        name = kwargs.get("name", "")
        style = kwargs.get("style", '-')
        alpha = kwargs.get("alpha", 1.0)
        shift_color = kwargs.get("shift_colour", False)
        z_order = kwargs.get("traj_zorder", 1)
        linewidth = kwargs.get("traj_linewidth",plt.rcParams["lines.linewidth"])


        if shift_color:
            shift_color = float(shift_color)
            import colorsys
            hsv_color = colorsys.rgb_to_hsv(color[0], color[1], color[1])
            # shift slightly
            h = hsv_color[0] + shift_color
            color = colorsys.hsv_to_rgb(h, hsv_color[1], hsv_color[2])

        if evo_plot.SETTINGS.plot_usetex:
            name = name.replace("_", "\\_")

        plot_traj(ax, plot_mode, traj, style, color, name,
             plot_start_end_markers=plot_start_end_markers,
             alpha=alpha,
             zorder=z_order,
             linewidth=linewidth)

    def _set_colours(self,
                     obj_trajectories: typing.Dict[str, evo_trajectory.PosePath3D],
                     provided_colours: Optional[List[Any]] = None):
        import collections
        import matplotlib.cm as cm

        # use provided colour map
        if provided_colours is not None:
            # should be list of colours, one for each obj traj
            if len(provided_colours) != len(obj_trajectories):
                raise evo_plot.PlotException(f"Expected trajectories and provided colours to have the same length {len(obj_trajectories)} != {len(provided_colours)}")
            #set internal colours
            self._colours = provided_colours

        # use cmap as described by evo plot settings
        elif evo_plot.SETTINGS.plot_multi_cmap.lower() != "none" and isinstance(
                obj_trajectories, collections.abc.Iterable):
            cmap = getattr(cm, evo_plot.SETTINGS.plot_multi_cmap)
            self._colours = cmap(np.linspace(0, 1, len(obj_trajectories)))
        else:
            import seaborn as sns
            self._colours= sns.color_palette()

        self._reset_colours()

    def _reset_colours(self):
        """
        Updates (resets) the colour iterator based on the current colours
        """
        import itertools
        assert self._colours is not None
        self._colours_itr = itertools.cycle(self._colours)


    def _get_next_colour(self) -> Any:
        return  next(self._colours_itr)



# modification of evo.trajectories
# so much going on in this function - refactor and comment!!!!
def plot_object_trajectories(
        fig: Figure,
        obj_trajectories: typing.Dict[str, evo_trajectory.PosePath3D],
        obj_trajectories_ref: Optional[typing.Dict[str, evo_trajectory.PosePath3D]] = None,
        plot_mode=evo_plot.PlotMode.xy,
        title: str = "",
        subplot_arg: int = 111,
        plot_start_end_markers: bool = False,
        length_unit: evo_plot.Unit = evo_plot.Unit.meters,
        **kwargs) -> Axes:

    return ObjectTrajectoryPlotter().plot(
        fig,
        obj_trajectories,
        obj_trajectories_ref,
        plot_mode,
        title,
        subplot_arg,
        plot_start_end_markers,
        length_unit,
        **kwargs)


def plot_ame_error(fig: Figure ,motion_est: evo_trajectory.PoseTrajectory3D, motion_ref: evo_trajectory.PoseTrajectory3D, label:str):
    assert motion_est.num_poses == motion_ref.num_poses

    ape_E = dynosam_metrics.AME(dynosam_metrics.PoseRelation.full_transformation)
    data = (motion_ref, motion_est)
    ape_E.process_data(data)

    plot_per_frame_error(fig, ape_E, label)

# needs to be RPE so we can check for pose
def plot_per_frame_error(fig: Figure, errors: dynosam_metrics.RPE , label:str, frames = None):

    if len(errors.E) == 0:
        print(f"Cannot plot per frame error as metric {errors} is empty!")
        return

    # check result is full transformation
    assert errors.E[0].shape == (4,4), ("Error must be a full transformation!")

    rot_error = []
    trans_error = []
    for E in errors.E:
        E_se3 = gtsam.Pose3(E)
        E_trans = E_se3.translation()
        E_rot = E_se3.rotation()
        rx, ry, rz =  np.degrees(E_rot.rpy())

        trans_error.append(E_trans)
        rot_error.append(np.array([rx, ry, rz]))

    rot_error = np.array(rot_error)
    trans_error = np.array(trans_error)

    rot_error_axes = fig.add_subplot(211)
    trans_error_axes = fig.add_subplot(212)

    rot_error_axes.set_title(f"Rotation Error ({label})", fontweight="bold")
    trans_error_axes.set_title(f"Translation Error ({label})", fontweight="bold")


    set_clean_background(rot_error_axes)
    set_clean_background(trans_error_axes)
    trans_error_axes.margins(0.001)
    rot_error_axes.margins(0.001)

    if frames is not None:
        assert len(frames) == len(errors.E)
    else:
        frames = np.arange(1, len(errors.E) + 1)


    rot_error_axes.plot(frames, rot_error[:,0], label="rx", color=get_nice_red())
    rot_error_axes.plot(frames, rot_error[:,1], label="ry", color=get_nice_green())
    rot_error_axes.plot(frames, rot_error[:,2], label="rz", color=get_nice_blue())

    trans_error_axes.plot(frames, trans_error[:,0], label="tx", color=get_nice_red())
    trans_error_axes.plot(frames, trans_error[:,1], label="ty", color=get_nice_green())
    trans_error_axes.plot(frames, trans_error[:,2], label="tz", color=get_nice_blue())

    trans_error_axes.set_xlabel("Frame Index [-]")
    rot_error_axes.set_xlabel("Frame Index [-]")

    trans_error_axes.set_ylabel("$E_t$(m)", fontsize=19)
    rot_error_axes.set_ylabel("$E_r$(\N{degree sign})", fontsize=19)

    rot_error_axes.legend(loc='upper left')
    trans_error_axes.legend(loc='upper left')

    fig.tight_layout(pad=0.5)


# fix from here https://github.com/matplotlib/matplotlib/issues/21688
# TODO: clean up
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)


#TODO: plotmode?
# TODO: outdoor vs indoor presets
# or just the ObjectMotionTrajectory
def plot_velocities(
        ax: Axes,
        object_trajectory: ObjectMotionTrajectory,
        color = 'r'):

    def draw_arrow(ax, xs, ys, zs, color):
        arrow = Arrow3D(xs, ys, zs, arrowstyle='-|>', color=color, mutation_scale=15, lw=3)
        # arrow = Arrow3D(xs, ys, zs, arrowstyle='-|>', color=color, mutation_scale=8, lw=1)
        ax.add_artist(arrow)

    for (pose_k_1, e_H_k_world) in object_trajectory.get_motion_with_pose_previous_iterator(skip=8):
        if pose_k_1 is None or e_H_k_world is None:
            continue
        I =  evo.core.transformations.identity_matrix()
        R_motion =  evo.core.transformations.identity_matrix()
        # ensure homogenous
        R_motion[0:3, 0:3] = e_H_k_world[0:3, 0:3]

        t_motion = evo.core.transformations.translation_from_matrix(e_H_k_world)
        t_pose = evo.core.transformations.translation_from_matrix(pose_k_1)

        # make homogenous
        t_motion = np.insert(t_motion, 3, 1)
        t_pose = np.insert(t_pose, 3, 1)

        # # implement ^o_Ad_B = (I - ^o_AR_B)^o_ot_A + ^o_At_B from Chirikjian
        # velocity = (I - R_motion) @ t_pose + t_motion
        # from VDO-SLAM
        velocity = t_motion - (I - R_motion) @ t_pose

        start = t_pose
        end = t_pose +  20.0 *velocity
        # end = t_pose +  3.0 *velocity
        draw_arrow(ax, (start[0], end[0]), (start[1], end[1]), (start[2], end[2]), color)

        # arrow_length_ratio
        # ax.quiver(t_pose[0], t_pose[1], t_pose[2], velocity[0], velocity[1], velocity[2], length=5.0, normalize=False, color="red", pivot='tail')






def draw_camera_frustum(ax: Axes,
                        transformation_matrix: np.ndarray,
                        far=0.6, fov=50,
                        aspect_ratio=0.9,
                        near=0.0,
                        frustrum_colour = get_nice_blue(),
                        convention="camera",
                        draw_planes: bool = False):
    if convention not in ["camera","world"]:
        raise Exception(f"Unknown convention {convention}. Must either be camera or world")

    # Field of view (fov) should be in degrees
    fov = np.radians(fov)  # Convert to radians

    # Calculate height and width of near and far planes
    h_near = 2 * np.tan(fov / 2) * near
    w_near = h_near * aspect_ratio
    h_far = 2 * np.tan(fov / 2) * far
    w_far = h_far * aspect_ratio

    # Frustum vertices in camera coordinates (default in camera frame)
    frustum_points = np.array([
        [0, 0, 0],  # Camera origin
        [-w_near / 2, -h_near / 2, -near],  # Near plane bottom-left
        [w_near / 2, -h_near / 2, -near],   # Near plane bottom-right
        [w_near / 2, h_near / 2, -near],    # Near plane top-right
        [-w_near / 2, h_near / 2, -near],   # Near plane top-left
        [-w_far / 2, -h_far / 2, -far],     # Far plane bottom-left
        [w_far / 2, -h_far / 2, -far],      # Far plane bottom-right
        [w_far / 2, h_far / 2, -far],       # Far plane top-right
        [-w_far / 2, h_far / 2, -far]       # Far plane top-left
    ])

    # Transform to the world convention if required
    if convention == 'world':
        # Rotate the points to match world convention (x forward, z up, y right)
        rotation_matrix = np.array([
            [0, 0, 1, 0],  # X-axis (forward) becomes Z-axis
            [0, 1, 0, 0],  # Y-axis remains Y-axis
            [-1, 0, 0, 0], # Z-axis (up) becomes -X-axis (to point along X in world)
            [0, 0, 0, 1]   # Homogeneous coordinate
        ])
        frustum_points_h = np.hstack([frustum_points, np.ones((frustum_points.shape[0], 1))])
        frustum_points = (rotation_matrix @ frustum_points_h.T).T[:, :3]

    # Apply the transformation matrix to the frustum points
    frustum_points_h = np.hstack([frustum_points, np.ones((frustum_points.shape[0], 1))])
    transformed_points = (transformation_matrix @ frustum_points_h.T).T[:, :3]

    # Define the frustum edges (connecting lines between vertices)
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),  # Camera origin to near plane corners
        (1, 2), (2, 3), (3, 4), (4, 1),  # Near plane edges
        (5, 6), (6, 7), (7, 8), (8, 5),  # Far plane edges
        (1, 5), (2, 6), (3, 7), (4, 8)   # Near plane to far plane edges
    ]

    # Plot the frustum edges
    for edge in edges:
        points = transformed_points[np.array(edge)]
        ax.plot(points[:, 0], points[:, 1], points[:, 2], '-', color=frustrum_colour)

    # Optionally, draw the near and far planes as polygons
    if draw_planes:
        near_plane = Poly3DCollection([transformed_points[1:5]], color='cyan', alpha=0.3)
        far_plane = Poly3DCollection([transformed_points[5:]], color='magenta', alpha=0.3)
        ax.add_collection3d(near_plane)
        ax.add_collection3d(far_plane)



def draw_camera_frustrums(ax: Axes, trajectory: evo_trajectory.PosePath3D, downsample: float = 1.0, **kwargs):
    traj = ObjectTrajectoryPlotter.reduce_trajectory(downsample, trajectory)
    for poses in traj.poses_se3:
        draw_camera_frustum(ax, poses, **kwargs)

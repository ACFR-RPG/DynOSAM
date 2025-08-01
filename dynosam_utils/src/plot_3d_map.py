import dynosam_utils.evaluation.evaluation_lib as eval

from evo.core import lie_algebra, trajectory, metrics, transformations
import evo.tools.plot as evo_plot
import evo.core.units as evo_units
import numpy as np

import matplotlib.pyplot as plt

plt.rcdefaults()



def make_plot(results_folder_path, prefix, plot_collection: evo_plot.PlotCollection=None):
    dataset_eval = eval.DatasetEvaluator(results_folder_path)
    data_files = dataset_eval.make_data_files(prefix)

    map_points_log_path = dataset_eval.create_existing_file_path(data_files.map_point_log)
    if map_points_log_path is None:
        print("Cannot find map points file")
        return

    camera_pose_eval = dataset_eval.create_camera_pose_evaluator(data_files)
    motion_eval = dataset_eval.create_motion_error_evaluator(data_files)

    plotter = eval.MapPlotter3D(map_points_log_path, camera_pose_eval, motion_eval, title=prefix,
                                plot_object_points=True,
                                plot_gt_objects=True,
                                plot_gt_camera=True,
                                downsample_static_cloud=0.05)

    if plot_collection is None:
        plot_collection = evo_plot.PlotCollection("Map")

    ax, fig = plotter.plot_3d_map_points(plot_collection)

    ax.view_init(azim=-113, elev=38, roll=1)
    ax.grid(False)
    fig.tight_layout()
    # ax.get_legend().remove()

    ax.set_rasterization_zorder(10)



    return plot_collection

    # MapPlotter3D,
    #             map_points_log_path,
    #             camera_pose_eval,
    #             motion_eval


# make_plot("/root/results/DynoSAM/acfr_1_moving_small", "rgbd_motion_world_backend")
# make_plot("/root/results/Dynosam_tro2024/kitti_0000", "rgbd_motion_world_backend")
# make_plot("/root/results/Dynosam_tro2024/carla_l2", "rgbd_motion_world_backend")
# make_plot("/root/results/Dynosam_tro2024/omd_swinging_4_unconstrained_sliding", "rgbd_motion_world_backend")
# make_plot("/root/results/DynoSAM/test_carla_l1/", "rgbd_motion_world_backend")
# make_plot("/root/results/DynoSAM/test_kitti_main/", "rgbd_motion_world_backend")
# make_plot("/root/results/DynoSAM/test_kitti_vo_0004/", "rgbd_motion_world_backend")
# make_plot("/root/results/DynoSAM/omd_vo_test/", "rgbd_motion_world_backend")
# make_plot("/root/results/DynoSAM/test_kitti_main/", "rgbd_motion_world_backend")

# make_plot("/root/results/Dynosam_tro2024/kitti_0000/", "rgbd_motion_world_backend")

# make_plot("/root/results/misc/", "object_centric_backend")
# make_plot("/root/results/misc/", "object_centric_LM_opt_backend")

plot_collection = evo_plot.PlotCollection("Map")
# make_plot("/root/results/TRO2025/omd_swinging_4_unconstrained_sliding", "rgbd_motion_world_backend", plot_collection)


make_plot("/root/results/misc/", "hybrid_backend", plot_collection)
make_plot("/root/results/misc/", "hybrid_batch_opt_backend", plot_collection)

make_plot("/root/results/misc/", "hybrid_smart_structureless_backend", plot_collection)
make_plot("/root/results/misc/", "hybrid_smart_structureless_batch_opt_backend", plot_collection)


# make_plot("/root/results/misc/", "hybrid_all_robust_backend", plot_collection)
# make_plot("/root/results/misc/", "hybrid_all_robust_opt_backend", plot_collection)

# make_plot("/root/results/misc/", "hybrid_not_robust_backend", plot_collection)
# make_plot("/root/results/misc/", "hybrid_not_robust_opt_backend", plot_collection)

# make_plot("/root/results/misc/", "hybrid_static_robust_backend", plot_collection)
# make_plot("/root/results/misc/", "hybrid_static_robust_opt_backend", plot_collection)

# make_plot("/root/results/misc/", "hybrid_smart_structureless_not_robust_backend", plot_collection)
# make_plot("/root/results/misc/", "hybrid_smart_structureless_not_robust_batch_opt_backend", plot_collection)

# make_plot("/root/results/misc/", "hybrid_smart_structureless_static_robust_backend", plot_collection)
# make_plot("/root/results/misc/", "hybrid_smart_structureless_static_robust_batch_opt_backend", plot_collection)

# make_plot("/root/results/misc/", "rgbd_motion_world_backend", plot_collection)

# make_plot("/root/results/misc/test_oc/", "object_centric_backend", plot_collection)
# make_plot("/root/results/misc/test_oc/", "object_centric_isam_opt_backend", plot_collection)

# # make_plot("/root/results/misc/test_oc/", "object_centric_structureless_decoupled_backend", plot_collection)
# make_plot("/root/results/misc/test_oc/", "object_centric_structureless_decoupled_isam_opt_backend", plot_collection)

# # # make_plot("/root/results/misc/test_oc/", "object_centric_decoupled_backend", plot_collection)
# # make_plot("/root/results/misc/test_oc/", "object_centric_decoupled_isam_opt_backend", plot_collection)

# # # make_plot("/root/results/misc/test_oc/", "object_centric_structureless_backend", plot_collection)
# # make_plot("/root/results/misc/test_oc/", "object_centric_structureless_isam_opt_backend", plot_collection)

# make_plot("/root/results/misc/test_oc/", "object_centric_smart_structureless_backend", plot_collection)
# make_plot("/root/results/misc/test_oc/", "object_centric_smart_structureless_batch_opt_backend", plot_collection)

# make_plot("/root/results/misc/test_parallel_oc/", "parallel_object_centric_backend", plot_collection)

# make_plot("/root/results/misc/test/", "object_centric_batch_opt_backend", plot_collection)
# make_plot("/root/results/misc/test/", "object_centric_backend", plot_collection)

# make_plot("/root/results/misc/test/", "object_centric_smart_structureless_backend", plot_collection)
# make_plot("/root/results/misc/test/", "object_centric_smart_structureless_batch_opt_backend", plot_collection)

# make_plot("/root/results/TRO2025/test_kitti_main", "frontend", plot_collection)
# make_plot("/root/results/TRO2025/test_kitti_main", "rgbd_motion_world_backend", plot_collection)
# make_plot("/root/results/Dynosam_ecmr2024/kitti_0000", "object_centric_inc_backend", plot_collection)
# make_plot("/root/results/Dynosam_ecmr2024/kitti_0000", "frontend", plot_collection)




plot_collection.show()

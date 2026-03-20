from dynosam_utils.evaluation.runner import run, run_dynosam_rgbd_from_rosbag
import os
import sys

# runs new incremental backend (parallel-hybrid)
parallel_hybrid = 3
full_hybrid=2
# runs world centric backend as batch (now called wcme)
motion_world_backend_type = 0

kf_hybrid=8

test_hybrid_smf = 7 # TESTING_HYBRID_SMF

output_path = "/root/results/frontend_filtering/"

def run_sequnce(path, name, data_loader_num, backend_type, *args, **kwargs):
    run_as_frontend = kwargs.get("run_as_frontend", True)
    # use the dyno_sam_experiments_launch file
    run_as_experiment = kwargs.get("run_as_experiment", False)
    run_analysis = kwargs.get("run_analysis", False)

    parsed_args = {
        "dataset_path": path,
        "output_path": output_path,
        "name": name,
        "run_pipeline": True,
        "run_analysis": run_analysis,
    }

    additional_args = [
        "--data_provider_type={}".format(data_loader_num),
        "--v=30"
    ]

    parsed_args["launch_file"] = "dyno_sam_launch.py"

    if run_as_frontend:
        additional_args.extend([
            "--use_backend=0",
            "--save_frontend_json=true"
        ])
    else:
        additional_args.extend([
            "--backend_updater_enum={}".format(backend_type),
            "--use_backend=1"
        ])
        if run_as_experiment:
            parsed_args["launch_file"] = "dyno_sam_experiments_launch.py"

    if len(args) > 0:
        additional_args.extend(list(args))

    # print(additional_args)
    run(parsed_args, additional_args)


def run_analysis(name):
    parsed_args = {
        "dataset_path": "",
        "output_path": output_path,
        "name": name,
        "run_pipeline": False,
        "run_analysis": True,
    }
    parsed_args["launch_file"] = "dyno_sam_launch.py"
    run(parsed_args, [])

kitti_dataset = 0
virtual_kitti_dataset = 1
cluster_dataset = 2
omd_dataset = 3
aria=4
tartan_air = 5
viode = 6

def run_online_sequence(name, *args):
    rosbag = "/root/data/craig_integration/realsense/realsense_2025-02-07-14-57-13_0/"
    run_dynosam_rgbd_from_rosbag(
        rosbag,
        output_path,
        name,
        list(args),
        run_pipeline=True
    )

def prep_dataset(path, name, data_loader_num, *args):
    backend_type = parallel_hybrid
    run_as_frontend=True
    run_sequnce(
        path,
        name,
        data_loader_num,
        backend_type,
        *args,
        run_as_frontend=run_as_frontend)

# from saved data
def run_saved_sequence(path, name, data_loader_num, *args, **kwargs):
    backend_type = kwargs.get("backend_type", parallel_hybrid)
    kwargs_dict = dict(kwargs)
    kwargs_dict["run_as_frontend"] = False
    args_list = list(args)
    # args_list.append("--init_object_pose_from_gt=true")
    run_sequnce(
        path,
        name,
        data_loader_num,
        backend_type,
        *args_list,
        **kwargs_dict)


# kitti stuff
def prep_kitti_sequence(path, name, *args):
    args_list = list(args)
    args_list.append("--shrink_row=25")
    args_list.append("--shrink_col=50")
    # args_list.append("--use_propogate_mask=true")
    prep_dataset(path, name, kitti_dataset, *args_list)

def run_kitti_sequence(path, name, *args, **kwargs):
    run_saved_sequence(path, name, kitti_dataset, *args, **kwargs)
    # run_analysis(name)

# cluster
def prep_cluster_sequence(path, name, *args, **kwargs):
    prep_dataset(path, name, cluster_dataset, *args, **kwargs)

def run_cluster_sequence(path, name, *args, **kwargs):
    run_saved_sequence(path, name, cluster_dataset, *args, **kwargs)

# omd
def prep_omd_sequence(path, name, *args, **kwargs):
    args_list = list(args)
    args_list.append("--shrink_row=0")
    args_list.append("--shrink_col=0")
    prep_dataset(path, name, omd_dataset, *args_list, **kwargs)

def run_omd_sequence(path, name, *args, **kwargs):
    run_saved_sequence(path, name, omd_dataset, *args, **kwargs)


def run_experiment_sequences(dataset_path, dataset_name, dataset_loader, *args):

    def append_args_list(*specific_args):
        args_list = list(args)
        args_list.extend(list(specific_args))
        return args_list
    # run fukk hybrid in (full)batch mode to get results!!
    run_sequnce(dataset_path, dataset_name, dataset_loader, kf_hybrid,  *append_args_list(), run_as_frontend=False, run_as_experiment=False, run_analysis=False)


def run_viodes():

#     run_experiment_sequences("/root/data/VIODE/city_day/mid", "viode_city_day_mid", viode, "--v=100")
    # run_experiment_sequences("/root/data/VIODE/city_day/high","viode_city_day_high", viode, "--ending_frame=1110")
    run_experiment_sequences("/root/data/VIODE/city_day/high","test_viode", viode,"--starting_frame=0", "--ending_frame=1110", "--v=10",  "--use_backend=true")
# # zero_elements_ratio
#     run_experiment_sequences("/root/data/VIODE/city_night/mid", "viode_city_night_mid", viode)
    # run_experiment_sequences("/root/data/VIODE/city_night/high", "viode_city_night_high", viode)

    # run_experiment_sequences("/root/data/VIODE/parking_lot/mid", "parking_lot_night_mid", viode)
    # run_experiment_sequences("/root/data/VIODE/parking_lot/high", "parking_lot_night_high", viode)

def run_omd():
    run_experiment_sequences("/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/","test_omd", omd_dataset, "--ending_frame=300", "--use_backend=true", "--v=40", "--hybrid_motion_solver=1")


def run_tartan_air():
    # run_experiment_sequences("/root/data/TartanAir_shibuya/RoadCrossing03", "tas_rc3", tartan_air) #max_object_depth: 10.0
    # run_experiment_sequences("/root/data/TartanAir_shibuya/RoadCrossing04", "test_tartan", tartan_air, "--use_backend=false", "--use_object_motion_filtering=true")
    # run_experiment_sequences("/root/data/TartanAir_shibuya/RoadCrossing05", "test_tartan", tartan_air, "--use_backend=true")
    run_experiment_sequences("/root/data/TartanAir_shibuya/RoadCrossing06", "tas_rc6_FS", tartan_air,"--use_backend=true","--hybrid_motion_solver=2")
    # run_experiment_sequences("/root/data/TartanAir_shibuya/RoadCrossing07", "tas_rc7", tartan_air, "--starting_frame=5", "--ending_frame=65")
    # run_analysis("tas_rc7")

    # run_experiment_sequences("/root/data/TartanAir_shibuya/Standing01", "tas_s1", tartan_air)
    # run_experiment_sequences("/root/data/TartanAir_shibuya/Standing02", "tas_s2", tartan_air)

def run_cluster():
    # run_experiment_sequences("/root/data/cluster_slam/CARLA-L2/", "cluster_l2_static_only", cluster_dataset)
    run_experiment_sequences("/root/data/cluster_slam/CARLA-L1/", "carla_test_MO", cluster_dataset,  "--use_backend=true","--hybrid_motion_solver=4")
    # run_experiment_sequences("/root/data/cluster_slam/CARLA-S2/", "cluster_s2_static_only", cluster_dataset)
    # run_experiment_sequences("/root/data/cluster_slam/CARLA-S1/", "cluster_s1_static_only", cluster_dataset)

def run_kitti():
    # run_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0000/", "test_kitti", kitti_dataset, "--shrink_row=25", "--shrink_col=50", "--v=30", "--use_backend=true", "--use_propogate_mask=true")
    # run_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0001/", "kitti_0001_static_only", kitti_dataset, "--shrink_row=25", "--shrink_col=50")
    # run_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0002/", "kitti_0002_static_only", kitti_dataset, "--shrink_row=25", "--shrink_col=50")
    # run_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0003/", "test", kitti_dataset, "--shrink_row=25", "--shrink_col=50",  "--use_backend=true", "--use_object_motion_filtering=true")
    # run_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0005/", "kitti_0005_static_only", kitti_dataset, "--shrink_row=25", "--shrink_col=50")
    # run_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0006/", "kitti_0006_static_only", kitti_dataset, "--shrink_row=25", "--shrink_col=50")
    # run_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0018/", "kitti_0018_static_only", kitti_dataset, "--shrink_row=25", "--shrink_col=50")
    # run_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0020/", "test_kitti", kitti_dataset,"--v=10", "--shrink_row=25", "--shrink_col=50",  "--use_backend=true")
    run_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0004/", "test_kitti", kitti_dataset, "--shrink_row=25", "--shrink_col=50", "--use_backend=true", "--v=30")


def run_aria():
    run_experiment_sequences("/root/data/zed/acfr_2_moving_small", "test_small_acfr", aria, "--use_backend=true",  "--hybrid_motion_solver=2")

def run_hybrid_solver_comparison_omd():
    # run_experiment_sequences("/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/","omd_FS", omd_dataset, "--ending_frame=500", "--hybrid_motion_solver=2")
    # run_experiment_sequences("/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/","omd_SS", omd_dataset, "--ending_frame=500", "--hybrid_motion_solver=1")
    # run_experiment_sequences("/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/","omd_EIF", omd_dataset, "--ending_frame=500", "--hybrid_motion_solver=0")
    # run_experiment_sequences("/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/","omd_PnP", omd_dataset, "--ending_frame=500", "--hybrid_motion_solver=3")
    run_experiment_sequences("/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/","omd_MO_test2", omd_dataset, "--ending_frame=300", "--hybrid_motion_solver=4", "--use_backend=true")
    # run_experiment_sequences("/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/","omd_FS_test1", omd_dataset, "--ending_frame=300", "--hybrid_motion_solver=2", "--use_backend=true")

    # run_analysis("omd_FS")
    # run_analysis("omd_SS")
    # run_analysis("omd_EIF")
    # run_analysis("omd_PnP")

def run_hybrid_solver_comparison_kitti():
    # run_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0004/","kitti04_FS", kitti_dataset, "--ending_frame=150", "--hybrid_motion_solver=1")
    # run_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0004/","kitti04_EIF", kitti_dataset, "--ending_frame=150", "--hybrid_motion_solver=0")
    # run_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0004/","kitti04_PnP", kitti_dataset, "--ending_frame=150", "--hybrid_motion_solver=2")

    # run_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0000/","kitti00_MO_backend", kitti_dataset, "--ending_frame=70", "--hybrid_motion_solver=4")
    # run_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0000/","kitti00_SS", kitti_dataset, "--ending_frame=70", "--hybrid_motion_solver=1")
    # run_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0000/","kitti00_FS", kitti_dataset, "--ending_frame=70", "--hybrid_motion_solver=2")
    # run_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0000/","kitti00_EIF", kitti_dataset, "--ending_frame=70", "--hybrid_motion_solver=0")
    # run_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0000/","kitti00_PnP", kitti_dataset, "--ending_frame=70", "--hybrid_motion_solver=3")

    # run_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0004/","kitti04_MO_noise_added_mL", kitti_dataset, "--ending_frame=100", "--hybrid_motion_solver=4")
    # run_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0004/","kitti04_FS_noise_added_mL", kitti_dataset, "--ending_frame=100", "--hybrid_motion_solver=2")

    # run_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0004/","kitti04_MO", kitti_dataset, "--ending_frame=100", "--hybrid_motion_solver=4")
    # run_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0020/","kitti20_MO", kitti_dataset, "--hybrid_motion_solver=4")
    # run_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0018/","kitti18_MO", kitti_dataset, "--hybrid_motion_solver=4")
    run_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0000/","kitti00_MO", kitti_dataset, "--hybrid_motion_solver=4")





    # run_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0004/","kitti04_SS", kitti_dataset, "--ending_frame=100", "--hybrid_motion_solver=1")
    # run_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0004/","kitti04_FS", kitti_dataset, "--ending_frame=100", "--hybrid_motion_solver=2")
    # run_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0004/","kitti04_EIF", kitti_dataset, "--ending_frame=100", "--hybrid_motion_solver=0")
    # run_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0004/","kitti04_PnP", kitti_dataset, "--ending_frame=100", "--hybrid_motion_solver=3")

    # run_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0000/","kitti00_SS", kitti_dataset, "--ending_frame=70", "--hybrid_motion_solver=1")
    # run_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0000/","kitti00_FS", kitti_dataset, "--ending_frame=70", "--hybrid_motion_solver=2")
    # run_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0000/","kitti00_EIF", kitti_dataset, "--ending_frame=70", "--hybrid_motion_solver=0")
    # run_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0000/","kitti00_PnP", kitti_dataset, "--ending_frame=70", "--hybrid_motion_solver=3")



    # run_analysis("kitti04_FS")
    # run_analysis("kitti04_EIF")
    # run_analysis("kitti04_PnP")

    # run_analysis("kitti20_SS")
    # run_analysis("kitti20_FS")
    # run_analysis("kitti20_EIF")
    # run_analysis("kitti20_PnP")
    # run_analysis("kitti00_EIF")
    # run_analysis("kitti00_PnP")

def run_uts_tech_lab_solver_comparison_test():
    run_online_sequence("tech_lab_1_MO_test1", "--ending_frame=300", "--hybrid_motion_solver=4")
    # run_online_sequence("tech_lab_1_SS", "--ending_frame=300", "--hybrid_motion_solver=1")
    # run_online_sequence("tech_lab_1_FS", "--ending_frame=300", "--hybrid_motion_solver=2")
    # run_online_sequence("tech_lab_1_FS_test1", "--ending_frame=300", "--hybrid_motion_solver=2")
    # run_online_sequence("tech_lab_1_EIF","--ending_frame=300", "--hybrid_motion_solver=0")
    # run_online_sequence("tech_lab_1_PnP","--ending_frame=300", "--hybrid_motion_solver=3")



if __name__ == '__main__':
    # run_hybrid_solver_comparison_omd()
    # run_hybrid_solver_comparison_kitti()
    # run_uts_tech_lab_solver_comparison_test()
    # run_tartan_air()
    # run_kitti()
    # run_viodes()
    run_cluster()
    # run_tartan_air()
    # run_aria()
    # run_omd()
    # run_online_sequence("test_online", "--hybrid_motion_solver=1")

    # run_analysis("test_kitti")

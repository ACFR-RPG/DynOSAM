from dynosam_utils.evaluation.runner import run
import os
import sys

# runs new incremental backend (parallel-hybrid)
parallel_hybrid = 3
full_hybrid=2
# runs world centric backend as batch (now called wcme)
motion_world_backend_type = 0

test_hybrid_smf = 7 # TESTING_HYBRID_SMF

output_path = "/root/results/Dynosam_ecmr2024/"

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
        "--v=20"
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


def run_ecmr_experiment_sequences(dataset_path, dataset_name, dataset_loader, *args):

    def append_args_list(*specific_args):
        args_list = list(args)
        args_list.extend(list(specific_args))
        return args_list
    # run fukk hybrid in (full)batch mode to get results!!
    # run_sequnce(dataset_path, dataset_name, dataset_loader, parallel_hybrid,  *append_args_list(), run_as_frontend=False, run_as_experiment=False, run_analysis=False)
    # run_sequnce(dataset_path, dataset_name, dataset_loader, full_hybrid, *append_args_list("--optimization_mode=0"), run_as_frontend=False, run_as_experiment=False, run_analysis=False)
    # run_sequnce(dataset_path, dataset_name, dataset_loader, motion_world_backend_type, *append_args_list("--optimization_mode=0"), run_as_frontend=False, run_as_experiment=False, run_analysis=True)

    run_sequnce(dataset_path, dataset_name, dataset_loader, test_hybrid_smf, *append_args_list("--optimization_mode=2", "--use_robust_kernals=false"), run_as_frontend=False, run_as_experiment=False, run_analysis=False)

    # pass
    # run the two batches again but with increemntal mode and additional suffix so we can individual logs!!!
    # run_sequnce(dataset_path, dataset_name, dataset_loader, full_hybrid, *append_args_list("--optimization_mode=2", "--regular_backend_relinearize_skip=10","--updater_suffix=inc"), run_as_frontend=False, run_as_experiment=False, run_analysis=False)
    # run_sequnce(dataset_path, dataset_name, dataset_loader, motion_world_backend_type, *append_args_list("--optimization_mode=2", "--regular_backend_relinearize_skip=10","--updater_suffix=inc"), run_as_frontend=False, run_as_experiment=False, run_analysis=True)

    # run_sequnce(dataset_path, dataset_name, dataset_loader, full_hybrid, *append_args_list("--optimization_mode=2", "--regular_backend_relinearize_skip=1","--updater_suffix=inc_relin1"), run_as_frontend=False, run_as_experiment=False, run_analysis=True)
    # run_sequnce(dataset_path, dataset_name, dataset_loader, motion_world_backend_type, *append_args_list("--optimization_mode=2", "--regular_backend_relinearize_skip=1","--updater_suffix=inc_relin1"), run_as_frontend=False, run_as_experiment=False, run_analysis=True)

def run_viodes():

#     run_ecmr_experiment_sequences("/root/data/VIODE/city_day/mid", "viode_city_day_mid", viode, "--v=100")
    # run_ecmr_experiment_sequences("/root/data/VIODE/city_day/high","viode_city_day_high", viode, "--ending_frame=1110")
    run_ecmr_experiment_sequences("/root/data/VIODE/city_day/high","test_viode", viode, "--starting_frame=500","--ending_frame=1110")
# # zero_elements_ratio
#     run_ecmr_experiment_sequences("/root/data/VIODE/city_night/mid", "viode_city_night_mid", viode)
    # run_ecmr_experiment_sequences("/root/data/VIODE/city_night/high", "viode_city_night_high", viode)

    # run_ecmr_experiment_sequences("/root/data/VIODE/parking_lot/mid", "parking_lot_night_mid", viode)
    # run_ecmr_experiment_sequences("/root/data/VIODE/parking_lot/high", "parking_lot_night_high", viode)

def run_omd():
    run_ecmr_experiment_sequences("/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/","test", omd_dataset, "--ending_frame=300")


def run_tartan_air():
    # run_ecmr_experiment_sequences("/root/data/TartanAir_shibuya/RoadCrossing03", "tas_rc3", tartan_air) #max_object_depth: 10.0
    # run_ecmr_experiment_sequences("/root/data/TartanAir_shibuya/RoadCrossing04", "test_tartan", tartan_air, "--v=100")
    run_ecmr_experiment_sequences("/root/data/TartanAir_shibuya/RoadCrossing05", "test_tartan", tartan_air)
    # run_ecmr_experiment_sequences("/root/data/TartanAir_shibuya/RoadCrossing06", "tas_rc6", tartan_air)
    # run_ecmr_experiment_sequences("/root/data/TartanAir_shibuya/RoadCrossing07", "tas_rc7", tartan_air, "--starting_frame=5", "--ending_frame=65")
    # run_analysis("tas_rc7")

    # run_ecmr_experiment_sequences("/root/data/TartanAir_shibuya/Standing01", "tas_s1", tartan_air)
    # run_ecmr_experiment_sequences("/root/data/TartanAir_shibuya/Standing02", "tas_s2", tartan_air)

def run_cluster():
    # run_ecmr_experiment_sequences("/root/data/cluster_slam/CARLA-L2/", "cluster_l2", cluster_dataset)
    run_ecmr_experiment_sequences("/root/data/cluster_slam/CARLA-L1/", "cluster_l1", cluster_dataset)
    # run_ecmr_experiment_sequences("/root/data/cluster_slam/CARLA-S2/", "cluster_s2", cluster_dataset)
    # run_ecmr_experiment_sequences("/root/data/cluster_slam/CARLA-S1/", "cluster_s1", cluster_dataset)

def run_kitti():
    # run_ecmr_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0000/", "test", kitti_dataset, "--shrink_row=25", "--shrink_col=300")
    # run_ecmr_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0001/", "kitti_0001", kitti_dataset, "--shrink_row=25", "--shrink_col=50")
    # run_ecmr_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0002/", "kitti_0002", kitti_dataset, "--shrink_row=25", "--shrink_col=50")
    run_ecmr_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0003/", "test", kitti_dataset, "--shrink_row=25", "--shrink_col=50", "--save_per_frame_dynamic_cloud=false")
    # run_ecmr_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0005/", "kitti_0005", kitti_dataset, "--shrink_row=25", "--shrink_col=50")
    # run_ecmr_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0006/", "kitti_0006", kitti_dataset, "--shrink_row=25", "--shrink_col=50")
    # run_ecmr_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0018/", "test", kitti_dataset, "--shrink_row=25", "--shrink_col=50")
    # run_ecmr_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0020/", "kitti_0020", kitti_dataset, "--shrink_row=25", "--shrink_col=50")

    # run_ecmr_experiment_sequences("/root/data/vdo_slam/kitti/kitti/0004/", "test_smf", kitti_dataset, "--shrink_row=25", "--shrink_col=50")


def run_aria():
    run_ecmr_experiment_sequences("/root/data/zed/acfr_2_moving_small", "test_small", aria)


if __name__ == '__main__':
    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0004/",
    #     "kitti_0004",
    #     "--ending_frame=150"
    # )

    # run_viodes()
    # run_tartan_air()
    # run_cluster()
    # run_omd()
    # run_aria()
    run_kitti()
    # run_analysis("kitti_0004_test")
    # run_analysis("kitti_0020")
    sys.exit(0)


    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0000/",
    #     "kitti_0000"
    # )
    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0000/",
    #     "kitti_0000",
    #     "--updater_suffix=inc",
    #     backend_type = object_centric_batch
    # )

    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0000/",
    #     "kitti_0000"
    # )
    # pass

    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0018/",
    #     "kitti_0018",
    #     "--updater_suffix=inc",
    #     backend_type = object_centric_batch
    # )

    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0018/",
    #     "kitti_0018",
    #     "--updater_suffix=inc",
    #     backend_type = motion_world_backend_type
    # )

    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0020/",
    #     "kitti_0020",
    #     "--updater_suffix=inc",
    #     backend_type = object_centric_batch
    # )

    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0020/",
    #     "kitti_0020",
    #     "--updater_suffix=inc",
    #     backend_type = motion_world_backend_type
    # )

    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0000/",
    #     "kitti_0000",
    #     "--updater_suffix=beta_01",
    #     "--relinearize_threshold=0.1"
    # )
    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0000/",
    #     "kitti_0000",
    #     "--updater_suffix=beta_1",
    #     "--relinearize_threshold=1.0"
    # )
    # run_analysis("kitti_0000")

    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0003/",
    #     "kitti_0003"
    # )

    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0005/",
    #     "kitti_0005"
    # )

    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0003/",
    #     "kitti_0003"
    # )
    # run_analysis("kitti_0003")

    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0004/",
    #     "kitti_0004",
    #     backend_type = motion_world_backend_type
    # )
    # run_analysis("kitti_0004")

    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0006/",
    #     "kitti_0006",
    #     backend_type = parallel_hybrid
    # )

    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0005/",
    #     "kitti_0005",
    #     backend_type = motion_world_backend_type
    # )
    # run_analysis("kitti_0006")
    # run_analysis("kitti_0005")

    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0006/",
    #     "kitti_0006"
    # )

    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0006/",
    #     "kitti_0006"
    # )
    # run_analysis("kitti_0006")


    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0018/",
    #     "kitti_0018"
    # )

    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0018/",
    #     "kitti_0018",
    #     backend_type=object_centric_batch
    # )
    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0018/",
    #     "kitti_0018",
    #     backend_type=motion_world_backend_type
    # )
    # run_analysis("kitti_0018")

    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0020/",
    #     "kitti_0020",
    #     "--ending_frame=550"
    # )
    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0020/",
    #     "kitti_0020",
    #     "--ending_frame=500",
    #     backend_type=motion_world_backend_type
    # )
    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0020/",
    #     "kitti_0020",
    #     "--ending_frame=500",
    #     backend_type=object_centric_batch
    # )
    # run_analysis("kitti_0020")

    # prep_omd_sequence(
    #     "/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/",
    #     "omd_swinging_4_unconstrained_long",
    #     "--ending_frame=318")

    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0000/",
    #     "kitti_0000",
    #     backend_type=parallel_hybrid,
    #     # backend_type=motion_world_backend_type
    # )
    # run_analysis("kitti_0000")

    # run_omd_sequence(
    #     "/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/",
    #     "omd_swinging_4_unconstrained_long",
    #     "--updater_suffix=beta_01",
    #     "--relinearize_threshold=0.1"
    # )
    # run_omd_sequence(
    #     "/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/",
    #     "omd_swinging_4_unconstrained_long",
    #     "--updater_suffix=beta_1",
    #     "--relinearize_threshold=1.0"
    # )
    # run_omd_sequence(
    #     "/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/",
    #     "omd_swinging_4_unconstrained_long",
    #     "--updater_suffix=inc",
    #     backend_type=motion_world_backend_type
    # )
    # run_omd_sequence(
    #     "/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/",
    #     "omd_swinging_4_unconstrained_long",
    #     "--updater_suffix=inc",
    #     backend_type=object_centric_batch
    # )
    # run_analysis("omd_swinging_4_unconstrained_long")

    prep_cluster_sequence(
        "/root/data/cluster_slam/CARLA-L2/",
        "cluster_l2_new",
        "--starting_frame=33",
        "--ending_frame=385",
        "--init_object_pose_from_gt=true"
    )

    prep_cluster_sequence(
        "/root/data/cluster_slam/CARLA-L1/",
        "cluster_l1_new",
        "--starting_frame=33",
        "--ending_frame=600",
        "--init_object_pose_from_gt=true"
    )

    prep_cluster_sequence(
        "/root/data/cluster_slam/CARLA-S1/",
        "cluster_s1_new",
        "--init_object_pose_from_gt=true"
    )

    prep_cluster_sequence(
        "/root/data/cluster_slam/CARLA-S2/",
        "cluster_s2_new",
        "--init_object_pose_from_gt=true"
    )

    # run_cluster_sequence(
    #     "/root/data/cluster_slam/CARLA-L2/",
    #     "cluster_l2",
    #     "--init_object_pose_from_gt=true",
    #     "--num_dynamic_optimize=4",
    #     backend_type=parallel_hybrid

    # )
    # run_analysis("cluster_l2")

    # prep_cluster_sequence(
    #     "/root/data/cluster_slam/CARLA-L1/",
    #     "cluster_l1",
    #     "--starting_frame=33",
    #     "--ending_frame=385",
    #     "--init_object_pose_from_gt=true"
    # )

    # run_cluster_sequence(
    #     "/root/data/cluster_slam/CARLA-L1/",
    #     "cluster_l1",
    #     "--init_object_pose_from_gt=false",
    #     "--num_dynamic_optimize=4"
    # )

    # run_cluster_sequence(
    #     "/root/data/cluster_slam/CARLA-L1/",
    #     "cluster_l1",
    #     "--init_object_pose_from_gt=false",
    #     "--num_dynamic_optimize=4",
    #     # backend_type=object_centric_batch

    # )
    # run_analysis("cluster_l1")

    # run_omd_sequence(
    #     "/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/",
    #     "omd_swinging_4_unconstrained",
    #     "--relinearize_thereshold=0.1"
    #     # backend_type=motion_world_backend_type
    # )

    # run_analysis("omd_swinging_4_unconstrained")

    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0004/",
    #     "kitti_0004",
    #     # backend_type=object_centric_batch
    # )
    # run_analysis("kitti_0004")

    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0003/",
    #     "kitti_0003",
    #     backend_type=object_centric_batch
    # )
    # run_analysis("kitti_0003")

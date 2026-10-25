from dynosam_utils.evaluation.runner import run

output_path = "/root/results/Dynosam2/"

kitti_dataset = 0
virtual_kitti_dataset = 1
cluster_dataset = 2
omd_dataset = 3
aria=4
tartan_air = 5
viode = 6
dynoepts=7

def run_sequnce(path, name, data_loader_num, *args, **kwargs):
    parsed_args = {
        "dataset_path": path,
        "output_path": output_path,
        "name": name,
        "run_pipeline": True,
        "run_analysis": False,
    }

    additional_args = [
        "--data_provider_type={}".format(data_loader_num),
        "--v=30"
    ]


    parsed_args["launch_file"] = "dynosam2_launch.py"
    parsed_args["package"] = "dynosam2_ros"

    if len(args) > 0:
        additional_args.extend(list(args))

    # print(additional_args)
    run(parsed_args, additional_args)

def run_omd():
    run_sequnce("/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/","test_omd", omd_dataset)

def run_kitti():
    run_sequnce("/root/data/vdo_slam/kitti/kitti/0020/", "test_kitti", kitti_dataset)


if __name__ == '__main__':
    run_kitti()

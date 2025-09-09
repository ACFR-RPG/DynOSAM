#!/usr/bin/python3

def parser():
    import argparse
    basic_desc = "TODO:"
    parser = argparse.ArgumentParser(add_help=True, description="{}".format(basic_desc))


    parser.add_argument("-f", "--folder", type=str, help="Path to folder container dynosam results folder")
    parser.add_argument("-n", "--name", type=str, help="Name of dynosam results folder (added poster to folder)")
    parser.add_argument("-p", "--path", type=str, help="Absolute path to dynosam results folder (ie. folder/name)")

    args = parser.parse_args()

    # Validation logic:
    if args.path is None:
        if args.name is None or args.folder is None:
            parser.error("Either --path OR both --name and --folder arguments be provided")
        else:
            return {"name": args.name, "output_path": args.folder}
    else:
        def split_path_and_name(s: str):
            from pathlib import Path
            p = Path(s)
            return str(p.parent), p.name
        # expect to be something in the form /output_path/name
        output_path, name = split_path_and_name(args.path)
        return {"name": name, "output_path": output_path}

if __name__ == '__main__':
    from dynosam_utils.evaluation.runner import run

    args = parser()

    parsed_args = {
        "dataset_path": "",
        "output_path": args["output_path"],
        "name": args["name"],
        "run_pipeline": False,
        "run_analysis": True,
    }
    parsed_args["launch_file"] = "dyno_sam_launch.py"
    run(parsed_args, [])

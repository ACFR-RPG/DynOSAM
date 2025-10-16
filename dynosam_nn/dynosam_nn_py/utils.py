import ament_index_python
import os

def get_weights_folder() -> str:
    return os.path.join(ament_index_python.packages.get_package_share_path("dynosam_nn"), "weights")

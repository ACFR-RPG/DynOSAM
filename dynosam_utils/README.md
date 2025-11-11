

## ROS bag record
```
ros2 bag record -e "dynosam\/(.*)"
```


requires a Latex compiler for the analysis
- pip3 install pylatex (for the python package)
- sudo apt-get install texlive-latex-base texlive-fonts-extra texlive-science(for the compiler)


- python3 eval_launch.py --dataset_path=/root/data/vdo_slam/kitti/kitti/0004 --name kitti_0004 --run_analysis


## Errors
If matplot lib errors e.g.
```
raise RuntimeError(
RuntimeError: latex was not able to process the following string:
b'lp'...)

```
Possibly missing rendering packages: sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super
Maybe also
`
sudo apt-get install texlive-science
`

Need scipy version 1.14.1
need gtsam


need `pip install pymongo` for bson

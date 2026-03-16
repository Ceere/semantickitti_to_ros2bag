# semantickitti_to_ros2bag
Convert semanticKitti dataset to ros2 bag

# Host requirement
- Ros jazzy
- Ubuntu 24.04
- python 3.12+

# Here is example
1. Download semanticKITTI dataset from https://semantic-kitti.org/dataset.html

2. run the below command
```
python3 semantickitti2rosbag.py --sequence_dir sequences/00 --bag_out final_bag_00
```

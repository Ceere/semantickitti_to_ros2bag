import os
import sys
import argparse
import numpy as np
import rclpy
from rclpy.serialization import serialize_message
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
import rosbag2_py

# SemanticKITTI official color mapping (BGR to RGB format for RViz)
LEARNING_MAP_COLOR = {
    0 : [0, 0, 0], 1 : [0, 0, 255], 10: [245, 150, 100], 11: [245, 230, 100],
    13: [250, 80, 100], 15: [150, 60, 30], 16: [255, 0, 0], 18: [180, 30, 80],
    20: [255, 0, 0], 30: [30, 30, 255], 31: [200, 40, 255], 32: [90, 30, 150],
    40: [255, 0, 255], 44: [255, 150, 255], 48: [75, 0, 75], 49: [75, 0, 175],
    50: [0, 200, 255], 51: [50, 120, 255], 52: [0, 150, 255], 70: [170, 255, 150],
    71: [0, 175, 0], 72: [0, 60, 135], 80: [80, 240, 150], 81: [150, 240, 255],
    99: [0, 0, 0]
}

def matrix_to_quaternion(m):
    """Convert a 3x3 rotation matrix to a quaternion [x, y, z, w]."""
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m[2, 1] - m[1, 2]) / S
        y = (m[0, 2] - m[2, 0]) / S
        z = (m[1, 0] - m[0, 1]) / S
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / S
        x = 0.25 * S
        y = (m[0, 1] + m[1, 0]) / S
        z = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / S
        x = (m[0, 1] + m[1, 0]) / S
        y = 0.25 * S
        z = (m[1, 2] + m[2, 1]) / S
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / S
        x = (m[0, 2] + m[2, 0]) / S
        y = (m[1, 2] + m[2, 1]) / S
        z = 0.25 * S
    return [x, y, z, w]

def parse_calib(calib_path):
    """Parse calib.txt to extract the Velodyne-to-Camera transformation matrix Tr."""
    with open(calib_path, 'r') as f:
        for line in f:
            if line.startswith('Tr:'):
                values = list(map(float, line.strip().split()[1:]))
                Tr = np.array(values).reshape(3, 4)
                Tr = np.vstack((Tr, [0, 0, 0, 1]))  # Pad to 4x4 homogeneous matrix
                return Tr
    return np.eye(4)

def create_colored_cloud(header, points, labels):
    """Generate a PointCloud2 message containing x, y, z, intensity, and RGB colors."""
    cloud_msg = PointCloud2()
    cloud_msg.header = header
    cloud_msg.height = 1
    cloud_msg.width = points.shape[0]
    cloud_msg.is_dense = False
    cloud_msg.is_bigendian = False

    # Define PointCloud2 fields (adding rgb)
    cloud_msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=16, datatype=PointField.UINT32, count=1),
    ]
    cloud_msg.point_step = 20
    cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width

    # Extract semantic class from the lower 16 bits of the label
    semantic_labels = labels[:, 0] & 0xFFFF
    rgb_packed = np.zeros(points.shape[0], dtype=np.uint32)
    
    # Map semantic IDs to packed RGB uint32 values
    for semantic_id, color in LEARNING_MAP_COLOR.items():
        mask = (semantic_labels == semantic_id)
        r, g, b = color
        packed_val = (r << 16) | (g << 8) | b
        rgb_packed[mask] = packed_val

    # Construct a structured array to easily convert to bytes
    structured_array = np.zeros(points.shape[0], dtype=[
        ('x', np.float32), ('y', np.float32), ('z', np.float32), 
        ('intensity', np.float32), ('rgb', np.uint32)
    ])
    structured_array['x'] = points[:, 0]
    structured_array['y'] = points[:, 1]
    structured_array['z'] = points[:, 2]
    structured_array['intensity'] = points[:, 3]
    structured_array['rgb'] = rgb_packed

    cloud_msg.data = structured_array.tobytes()
    return cloud_msg

def main():
    # ---------- Argument Parsing ----------
    parser = argparse.ArgumentParser(description="Convert SemanticKITTI dataset to ROS 2 Bag.")
    parser.add_argument("--sequence_dir", type=str, required=True, help="Path to the sequence directory (e.g., sequences/00)")
    parser.add_argument("--bag_out", type=str, default="semantickitti_bag", help="Output bag folder name")
    parser.add_argument("--cloud_topic", type=str, default="/velodyne_points", help="Topic name for point cloud")
    parser.add_argument("--tf_topic", type=str, default="/tf", help="Topic name for TF data")
    args = parser.parse_args()

    sequence_dir = args.sequence_dir
    bag_out_path = args.bag_out
    cloud_topic = args.cloud_topic
    tf_topic = args.tf_topic
    # --------------------------------------

    velodyne_dir = os.path.join(sequence_dir, "velodyne")
    labels_dir = os.path.join(sequence_dir, "labels")
    times_file = os.path.join(sequence_dir, "times.txt")
    poses_file = os.path.join(sequence_dir, "poses.txt")
    calib_file = os.path.join(sequence_dir, "calib.txt")

    if not os.path.exists(sequence_dir):
        print(f"[Error] Directory not found: {sequence_dir}")
        sys.exit(1)

    if os.path.exists(bag_out_path):
        print(f"[Error] Output bag directory '{bag_out_path}' already exists. Please choose a different name or delete it.")
        sys.exit(1)

    # 1. Read timestamps
    with open(times_file, 'r') as f:
        times = [float(line.strip()) for line in f.readlines()]

    # 2. Read Ground Truth Poses (Camera 0 poses in world coordinates)
    poses = []
    with open(poses_file, 'r') as f:
        for line in f:
            P_i = np.array(list(map(float, line.strip().split()))).reshape(3, 4)
            P_i = np.vstack((P_i, [0, 0, 0, 1]))
            poses.append(P_i)

    # 3. Read Calibration (Velodyne to Camera external parameters)
    Tr = parse_calib(calib_file)

    bin_files = sorted(os.listdir(velodyne_dir))
    label_files = sorted(os.listdir(labels_dir))

    # Initialize ROS 2 Bag Writer
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py._storage.StorageOptions(uri=bag_out_path, storage_id='mcap'),
        rosbag2_py._storage.ConverterOptions('', '')
    )

    # Create PointCloud Topic (id=0 required for Jazzy)
    writer.create_topic(rosbag2_py._storage.TopicMetadata(
        id=0, name=cloud_topic, type='sensor_msgs/msg/PointCloud2', serialization_format='cdr'))
    
    # Create TF Topic (id=1 required for Jazzy)
    writer.create_topic(rosbag2_py._storage.TopicMetadata(
        id=1, name=tf_topic, type='tf2_msgs/msg/TFMessage', serialization_format='cdr'))

    print(f"Starting conversion. Total frames: {len(bin_files)}...")

    for i, (bin_f, label_f, t, P_i) in enumerate(zip(bin_files, label_files, times, poses)):
        # --- A. Process Timestamps ---
        sec = int(t)
        nanosec = int((t - sec) * 1e9)
        timestamp_ns = int(t * 1e9)
        
        # --- B. Calculate and Write TF (Trajectory) ---
        # 1. Coordinate transformation matrix: KITTI Camera World -> ROS ENU World
        T_ros_cam = np.array([
            [ 0,  0,  1,  0],
            [-1,  0,  0,  0],
            [ 0, -1,  0,  0],
            [ 0,  0,  0,  1]
        ], dtype=np.float64)

        # 2. Real pose of Velodyne in ROS World Coordinates: T_world_velo = T_ros_cam * P_i * Tr
        T_world_velo = T_ros_cam @ P_i @ Tr
        
        t_msg = TransformStamped()
        t_msg.header.stamp.sec = sec
        t_msg.header.stamp.nanosec = nanosec
        t_msg.header.frame_id = "map"      # Global coordinate frame
        t_msg.child_frame_id = "velodyne"  # Sensor coordinate frame

        # Translation
        t_msg.transform.translation.x = float(T_world_velo[0, 3])
        t_msg.transform.translation.y = float(T_world_velo[1, 3])
        t_msg.transform.translation.z = float(T_world_velo[2, 3])
        
        # Rotation (Matrix to Quaternion)
        q = matrix_to_quaternion(T_world_velo[0:3, 0:3])
        t_msg.transform.rotation.x = float(q[0])
        t_msg.transform.rotation.y = float(q[1])
        t_msg.transform.rotation.z = float(q[2])
        t_msg.transform.rotation.w = float(q[3])

        tf_msg = TFMessage()
        tf_msg.transforms = [t_msg]
        writer.write(tf_topic, serialize_message(tf_msg), timestamp_ns)

        # --- C. Process and Write PointCloud ---
        bin_path = os.path.join(velodyne_dir, bin_f)
        label_path = os.path.join(labels_dir, label_f)

        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        labels = np.fromfile(label_path, dtype=np.uint32).reshape(-1, 1)

        header = Header()
        header.stamp.sec = sec
        header.stamp.nanosec = nanosec
        header.frame_id = "velodyne"

        cloud_msg = create_colored_cloud(header, points, labels)
        writer.write(cloud_topic, serialize_message(cloud_msg), timestamp_ns)

        # Monitor progress and altitude (Z-axis) to ensure correct coordinate mapping
        if i % 100 == 0:
            print(f"Processed {i}/{len(bin_files)} frames | Altitude(Z): {t_msg.transform.translation.z:.2f}m, Distance(X): {t_msg.transform.translation.x:.2f}m")

    print(f"Success! The ROS 2 bag with semantic point clouds and TF has been saved to: {bag_out_path}")

if __name__ == '__main__':
    main()

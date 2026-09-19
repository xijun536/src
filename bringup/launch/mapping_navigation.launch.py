
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import ReplaceString


def generate_launch_description():
    bringup_share = get_package_share_directory("bringup")
    livox_share = get_package_share_directory("livox_ros_driver2")
    fast_lio_share = get_package_share_directory("fast_lio")
    terrain_share = get_package_share_directory("terrain_analysis")
    nav_share = get_package_share_directory("rm_navigation")
    serial_bridge_share = get_package_share_directory("serial_bridge")
    nav2_bt_share = get_package_share_directory("nav2_bt_navigator")

    use_sim_time = LaunchConfiguration("use_sim_time")
    nav_rviz = LaunchConfiguration("nav_rviz")
    serial_config_file = LaunchConfiguration("serial_config_file")

    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time", default_value="False",
        description="Use a simulation clock (normally False on the real robot).",
    )
    declare_nav_rviz = DeclareLaunchArgument(
        "nav_rviz", default_value="True",
        description="Launch Nav2 RViz, which displays the live /map and Nav2 Goal tool.",
    )
    declare_serial_config_file = DeclareLaunchArgument(
      "serial_config_file",
      default_value=os.path.join(
          serial_bridge_share,
          "config",
          "bridge.yaml",
      ),
      description="Serial bridge parameter file",
    )


    #publish /livox/lidar and /livox/imu
    livox_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(livox_share, "launch_ROS2", "msg_MID360_launch.py")
        )
    )

    #publish static TF (livox_frame and base_frame)
    #had fixed in msg_MID360 config
    base_to_livox = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_livox_tf",# ros node name
        output="log",
        arguments=[
                    "--x", "0",
                    "--y", "0",
                    "--z", "0",
                    "--roll", "0",
                    "--pitch", "0",
                    "--yaw", "0",
                    "--frame-id", "base_frame",
                    "--child-frame-id", "livox_frame",
                  ]
    )
    
    fast_lio = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(fast_lio_share, "launch", "mapping.launch.py")
        ),
        launch_arguments={
          "use_sim_time": use_sim_time,
          "config_file": "mid360_re.yaml",
          "rviz": "False",
        }.items(),
    )

    #terrian 
    # receive(/odometry and /cloud_registerd from fastlio) and analyse terrian
    # publish /terrain_map
    terrain_analysis = Node(
        package="terrain_analysis",
        executable="terrainAnalysis",
        name="terrain_analysis",
        output="screen",
        parameters=[
            os.path.join(terrain_share, "config", "terrain_analysis_plus.yaml")
        ],
    )

    #receive pointcloud_to_laser_scan
    #publish  2d /scan
    pointcloud_to_laserscan = Node(
        package="pointcloud_tools",
        executable="pointcloud_to_laserscan_node",
        name="pointcloud_to_laserscan",
        output="screen",
        remappings=[("cloud_in", "/terrain_map"), ("scan", "/scan")],
        parameters=[{
            "target_frame": "base_frame",
            "transform_tolerance": 0.1,
            "queue_size": 20,
            "min_height": -0.25,
            "max_height": 1.6,
            "angle_min": -3.14159,
            "angle_max": 3.14159,
            "angle_increment": 0.0043,
            # Metadata for the real 10 Hz input.  slam_toolbox throttling is
            # configured separately in slam_toolbox_single_lidar.yaml.
            "scan_time": 0.1,
            "range_min": 0.35,
            "range_max": 10.0,
            "use_inf": True,
            "inf_epsilon": 1.0,
        }],
    )

    # Publishes the live OccupancyGrid /map and map -> odom_init TF.
    slam_toolbox = Node(
        package="slam_toolbox",
        executable="sync_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        parameters=[
            os.path.join(
                bringup_share, "config", "slam_toolbox.yaml"
            ),
            {"use_sim_time": use_sim_time},
        ],
    )

    # static map->odom_init regard as overlapping at first
    map_to_odom_init = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="online_map_to_odom_init_tf",
        output="log",
        arguments=
        [
            "--x", "0", 
            "--y", "0", 
            "--z", "0",
            "--roll", "0", 
            "--pitch", "0", 
            "--yaw", "0",
            "--frame-id", "map", 
            "--child-frame-id", "odom_init",
        ],
    )

    # This server lets the user save the live /map without starting a
    # map_server (map_server would incorrectly try to publish a prebuilt map).
    map_saver = Node(
        package="nav2_map_server",
        executable="map_saver_server",
        name="map_saver",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "save_map_timeout": 5.0,
            "free_thresh_default": 0.25,
            "occupied_thresh_default": 0.65,
            "map_subscribe_transient_local": True,
        }],
    )

    # manage the life of map_saver 
    map_saver_lifecycle = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_online_map_saver",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "autostart": True,
            "node_names": ["map_saver"],
        }],
    )

    nav2_params = ReplaceString(
      source_file=os.path.join(bringup_share, "config", "nav2_params.yaml"),
      replacements={
          "/home/eatwhat/nav_ws/src/bringup/behavior_tree/"
          "navigate_to_point_w_recovery.xml": os.path.join(
              nav2_bt_share,
              "behavior_trees",
              "navigate_to_pose_w_replanning_and_recovery.xml",
          ),
          "/home/eatwhat/nav_ws/src/bringup/behavior_tree/"
          "navigate_through_pose_w_recovery.xml": os.path.join(
              nav2_bt_share,
              "behavior_trees",
              "navigate_through_poses_w_replanning_and_recovery.xml",
          ),
      },
    )

    # Do not start map_server or alignment in this launch.  slam_toolbox is
    # the sole publisher of /map and the static node above is the sole
    # publisher of map -> odom_init.
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav_share, "launch", "bringup_rm_navigation.py")
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "map": "",
            "params_file": nav2_params,
            "nav_rviz": nav_rviz,
        }.items(),
    )

    serial_bridge_node = Node(
      package="serial_bridge",
      executable="serial_bridge",
      name="serial_driver",
      output="screen",
      emulate_tty=True,
      parameters=[serial_config_file],
    )


    return LaunchDescription([
        declare_use_sim_time,
        declare_nav_rviz,
        declare_serial_config_file,
        livox_driver,
        base_to_livox,
        fast_lio,
        terrain_analysis,
        pointcloud_to_laserscan,
        slam_toolbox,
        map_to_odom_init,
        map_saver,
        map_saver_lifecycle,
        navigation,
        serial_bridge_node,
    ])

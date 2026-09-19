import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    serial_bridge_share = get_package_share_directory("serial_bridge")  #找到serial_bridge包的目录

    config_file = LaunchConfiguration("config_file")   #在启动的时候可以通过命令行输入的配置文件当作配置文件

    declare_config_file = DeclareLaunchArgument(
        "config_file",
        default_value=os.path.join(
            serial_bridge_share,
            "config",
            "bridge.yaml",
        ),
        description="Serial bridge parameter file",
    )

    serial_bridge_node = Node(
        package="serial_bridge",
        executable="serial_bridge",
        name="serial_driver",
        output="screen",
        emulate_tty=True,
        parameters=[config_file],
    )

    return LaunchDescription([  #一键启动函数，先读取参数文件，后启动节点
        declare_config_file,
        serial_bridge_node,
    ])
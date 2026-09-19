import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _launch_setup(context):
    arguments = ["--config", LaunchConfiguration("config_file")]
    if _is_true(LaunchConfiguration("no_display").perform(context)):
        arguments.append("--no-display")
    return [
        Node(
            package="d435i_vision",
            executable="d435i_vision",
            output="screen",
            emulate_tty=True,
            arguments=arguments,
        )
    ]


def generate_launch_description():
    share = get_package_share_directory("d435i_vision")
    return LaunchDescription(
        [
            DeclareLaunchArgument("config_file",
                default_value=os.path.join(share, "config", "config.yaml"),
                description="D435i vision YAML configuration",
            ),
            DeclareLaunchArgument("no_display",
                default_value="false",
                description="Disable all OpenCV windows",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )

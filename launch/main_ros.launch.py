from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='persistent_tracker',
            executable='main_ros',
            name='persistent_tracker',
            output='screen',
        )
    ])

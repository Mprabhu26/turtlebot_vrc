import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_path          = get_package_share_directory("turtlebot_vrc")
    gazebo_ros        = get_package_share_directory("gazebo_ros")
    turtlebot3_gazebo = get_package_share_directory("turtlebot3_gazebo")
    world_file        = os.path.join(pkg_path, "worlds", "hospital_vrc.world")

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros, "launch", "gazebo.launch.py")
        ),
        launch_arguments={"world": world_file, "verbose": "false"}.items(),
    )

    spawn = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-entity", "turtlebot3",
            "-file", os.path.join(turtlebot3_gazebo, "models",
                                  "turtlebot3_burger", "model.sdf"),
            "-x", "0", "-y", "0", "-z", "0.01",
        ],
        output="screen",
    )

    return LaunchDescription([
        SetEnvironmentVariable("TURTLEBOT3_MODEL", "burger"),
        gazebo,
        TimerAction(period=20.0, actions=[spawn]),
    ])

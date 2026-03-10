import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_path     = get_package_share_directory('turtlebot_vrc')
    gazebo_ros   = get_package_share_directory('gazebo_ros')
    turtlebot3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    world_file   = os.path.join(pkg_path, 'worlds', 'hospital.world')

    return LaunchDescription([
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'burger'),

        # Launch Gazebo with hospital world
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_ros, 'launch', 'gazebo.launch.py')
            ),
            launch_arguments={'world': world_file, 'verbose': 'false'}.items(),
        ),

        # Spawn TurtleBot3 at reception (starting position)
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', 'turtlebot3',
                '-file', os.path.join(turtlebot3_gazebo, 'models',
                                      'turtlebot3_burger', 'model.sdf'),
                '-x', '-8', '-y', '-3', '-z', '0.01',
            ],
            output='screen',
        ),
    ])

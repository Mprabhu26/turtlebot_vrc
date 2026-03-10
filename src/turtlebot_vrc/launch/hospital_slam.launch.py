import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg          = get_package_share_directory('turtlebot_vrc')
    gazebo_ros   = get_package_share_directory('gazebo_ros')
    world_file   = os.path.join(pkg, 'worlds', 'hospital.world')
    urdf_file    = '/opt/ros/humble/share/turtlebot3_description/urdf/turtlebot3_burger.urdf'
    robot_desc   = open(urdf_file).read()

    return LaunchDescription([
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'burger'),
        SetEnvironmentVariable('LDS_MODEL', 'LDS-01'),
        SetEnvironmentVariable('GAZEBO_MODEL_PATH',
            '/opt/ros/humble/share/turtlebot3_gazebo/models'),

        # 1. Gazebo
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_ros, 'launch', 'gazebo.launch.py')
            ),
            launch_arguments={
                'world':   world_file,
                'verbose': 'false',
            }.items(),
        ),

        # 2. Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{
                'use_sim_time':       True,
                'robot_description':  robot_desc,
            }],
        ),

        # 3. Spawn robot after 5s
        TimerAction(period=5.0, actions=[
            Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                arguments=[
                    '-entity', 'turtlebot3_burger',
                    '-file', '/opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf',
                    '-x', '-8.0', '-y', '-3.0', '-z', '0.01',
                ],
                output='screen',
            ),
        ]),

        # 4. SLAM after 8s
        TimerAction(period=8.0, actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory('slam_toolbox'),
                        'launch', 'online_async_launch.py'
                    )
                ),
                launch_arguments={'use_sim_time': 'true'}.items(),
            ),
        ]),

        # 5. Auto explorer after 12s
        TimerAction(period=12.0, actions=[
            Node(
                package='turtlebot_vrc',
                executable='map_explorer',
                output='screen',
                parameters=[{'use_sim_time': True}],
            ),
        ]),
    ])

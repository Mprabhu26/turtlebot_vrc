from setuptools import find_packages, setup

package_name = 'turtlebot_vrc'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name,
            ['package.xml', 'commands.yaml']),
        ('share/' + package_name + '/worlds',
            ['worlds/hospital.world']),
        ('share/' + package_name + '/launch',
            ['launch/hospital.launch.py', 'launch/hospital_slam.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mprabhu',
    maintainer_email='mprabhu@todo.todo',
    description='VRC-7 Voice Recognition Control for TurtleBot3',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'voice_control = turtlebot_vrc.voice_control:main',
            'map_explorer = turtlebot_vrc.map_explorer:main',
        ],
    },
)

#!/bin/bash
cd ~/turtlebot_vrc_ws/
source install/setup.bash
export TURTLEBOT3_MODEL=burger

echo "🚀 Starting Gazebo Hospital World..."
ros2 launch turtlebot_vrc hospital.launch.py &
GAZEBO_PID=$!

echo "⏳ Waiting 8 seconds for Gazebo to load..."
sleep 8

echo "🎤 Starting Voice Control with MICROPHONE..."
echo "Speak commands like: 'go to pharmacy' or 'go from reception to ward'"
ros2 run turtlebot_vrc voice_control

# Cleanup
kill $GAZEBO_PID

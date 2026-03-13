#!/bin/bash
source /opt/ros/humble/setup.bash
cd ~/turtlebot_vrc_ws/
source install/setup.bash
export TURTLEBOT3_MODEL=burger

# Prompt for GROQ API key if not already set
if [ -z "$GROQ_API_KEY" ]; then
    read -p "Enter your GROQ API key: " GROQ_API_KEY
    export GROQ_API_KEY
fi

echo "Starting Gazebo Hospital World..."
ros2 launch turtlebot_vrc hospital.launch.py &
GAZEBO_PID=$!

echo "Waiting 8 seconds for Gazebo to load..."
sleep 8

echo "Starting Voice Control..."
echo "Say commands like: 'go to pharmacy' / 'go to ICU' / 'go to ward' / 'go to reception'"
ros2 run turtlebot_vrc voice_control

# Cleanup on exit
kill $GAZEBO_PID 2>/dev/null

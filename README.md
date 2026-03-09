# SO101 Arm - Pick and Place

## Author:

Venkata Madhav Tadavarthi (vmadhav@umd.edu)
Interview Assignment for Fireloop AI

## Folders:

- Isaac sim related files such as the usda, scripts can be found here: [Isaac Sim](./isaac-usd/)
- ROS2 Workspace (Perception, Planning Behavior Tree) can be found here: [so-arm](./so-arm/)
- More details provided in the `README.md` in their respective folders.

## Documentation:

- Demo Video: [click here]()
- Architecure Documentation: [click here]()

## Usage

Clone the repository using:

```bash
git clone https://github.com/Madhav2133/so101arm_fireloop.git
```

## Isaac Sim

- Open the USDA file with your Isaac Sim to view the environment

## Launch Instructions

- Go to workspace:

```bash
cd so-arm/so101_ws
colcon build
```

- Source the workspace:

```
source install/setup.bash
```

- Run the so101_arm bringup launch file:

- - Make sure to start the simulation in isaac sim before this command.

```
ros2 launch so101_bringup bringup_moveit.launch.py
```

- In another terminal, source the workspace and run the bt_node:

```
ros2 run so101_state_machine bt_node
```

You should be able to see the gripper open and the arm moving towards the cup.

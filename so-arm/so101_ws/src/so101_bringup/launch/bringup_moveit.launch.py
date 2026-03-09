import os

from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from moveit_configs_utils.substitutions import Xacro

from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def _as_bool(s: str) -> bool:
    return str(s).lower() in ["true", "1", "yes", "y", "on"]


def _moveit_params(moveit_config):
    if hasattr(moveit_config, "to_dict"):
        return moveit_config.to_dict()

    params = {}
    for attr in [
        "robot_description",
        "robot_description_semantic",
        "robot_description_kinematics",
        "planning_pipelines",
        "trajectory_execution",
    ]:
        if hasattr(moveit_config, attr):
            v = getattr(moveit_config, attr)
            if isinstance(v, dict):
                params.update(v)
            else:
                try:
                    params.update(dict(v))
                except Exception:
                    pass

    if hasattr(moveit_config, "planning_scene_monitor_parameters"):
        v = getattr(moveit_config, "planning_scene_monitor_parameters")
        if isinstance(v, dict):
            params.update(v)
        else:
            try:
                params.update(dict(v))
            except Exception:
                pass

    return params


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument("moveit_config_pkg", default_value="so101_moveit_config"),
        DeclareLaunchArgument("robot_name", default_value="so101_new_calib"),
        DeclareLaunchArgument("rviz_config", default_value="config/moveit.rviz"),
        DeclareLaunchArgument("ros2_controllers_file", default_value="config/ros2_controllers.yaml"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument(
            "controller_names",
            default_value="joint_state_broadcaster arm_controller gripper_controller",
            description="Space-separated ros2_control controller names to spawn.",
        ),
        # default to false; Isaac provides real hw interface
        DeclareLaunchArgument(
            "use_fake_hardware",
            default_value="false",
            description="Use mock ros2_control hardware instead of real hardware. Set true for standalone testing.",
        ),
        # TopicBasedSystem bridging /isaac_joint_states + /isaac_joint_command 
        DeclareLaunchArgument(
            "use_isaac",
            default_value="true",
            description="Use Isaac Sim as hardware backend via topic_based_ros2_control/TopicBasedSystem.",
        ),
        # world->robot static TF 
        DeclareLaunchArgument(
            "robot_base_link",
            default_value="so101_base_link",
            description="Name of the robot base link frame for the world->robot static TF. "
                        "Check your URDF and update this if needed.",
        ),
        # world frame origin offset (x y z roll pitch yaw)
        DeclareLaunchArgument("world_x", default_value="0.0"),
        DeclareLaunchArgument("world_y", default_value="0.0"),
        DeclareLaunchArgument("world_z", default_value="0.0"),
        DeclareLaunchArgument("world_roll",  default_value="0.0"),
        DeclareLaunchArgument("world_pitch", default_value="0.0"),
        DeclareLaunchArgument("world_yaw",   default_value="0.0"),
    ]

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=_launch_setup)])


def _launch_setup(context, *args, **kwargs):
    moveit_config_pkg = LaunchConfiguration("moveit_config_pkg").perform(context)
    robot_name        = LaunchConfiguration("robot_name").perform(context)
    ros2_controllers_rel = LaunchConfiguration("ros2_controllers_file").perform(context)
    use_sim_time      = _as_bool(LaunchConfiguration("use_sim_time").perform(context))
    use_isaac = _as_bool(LaunchConfiguration("use_isaac").perform(context))
    robot_base_link   = LaunchConfiguration("robot_base_link").perform(context)

    controller_names_str = LaunchConfiguration("controller_names").perform(context).strip()
    controller_names = controller_names_str.split() if controller_names_str else []

    moveit_share    = get_package_share_directory(moveit_config_pkg)
    bringup_pkg_path = get_package_share_directory('so101_bringup')
    rviz_config_path = os.path.join(bringup_pkg_path, 'rviz', 'so101.rviz')
    ros2_controllers_path = os.path.join(moveit_share, ros2_controllers_rel)

    moveit_config = MoveItConfigsBuilder(robot_name, package_name=moveit_config_pkg).to_moveit_configs()
    moveit_common_params = _moveit_params(moveit_config)

    robot_description_content = ParameterValue(
        Command([
            "xacro ",
            PathJoinSubstitution([
                FindPackageShare("so101_moveit_config"),
                "config",
                "so101_new_calib.urdf.xacro",
            ]),
            " use_fake_hardware:=", LaunchConfiguration("use_fake_hardware"),
            " use_isaac:=",         LaunchConfiguration("use_isaac"),
        ]),
        value_type=str,
    )

    robot_description = {"robot_description": robot_description_content}

    # Robot State Publisher 
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[robot_description, {"use_sim_time": use_sim_time}],
    )

    # Removed joint_state_publisher 
    # Isaac Sim publishes joint states through joint_state_broadcaster.

    # Static TF: world → robot base link 
    world_to_robot_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher_world_to_robot",
        output="log",
        arguments=[
            LaunchConfiguration("world_x"),
            LaunchConfiguration("world_y"),
            LaunchConfiguration("world_z"),
            LaunchConfiguration("world_roll"),
            LaunchConfiguration("world_pitch"),
            LaunchConfiguration("world_yaw"),
            "world",
            robot_base_link,
        ],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # ros2_control node 
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        output="screen",
        parameters=[
            ros2_controllers_path,
            {"use_sim_time": use_sim_time},
        ],
        remappings=[
            ("/controller_manager/robot_description", "/robot_description"),
        ],
    )

    # Controller spawners 
    spawn_jsb = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen",
    )

    spawners = []
    for c in controller_names:
        if c == "joint_state_broadcaster":
            continue
        spawners.append(
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[c, "--controller-manager", "/controller_manager"],
                output="screen",
            )
        )

    # MoveGroup 
    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_common_params, {"use_sim_time": use_sim_time}],
        arguments=["--ros-args", "--log-level", "info"],
    )

    # RViz2 
    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_path],
        parameters=[moveit_common_params, {"use_sim_time": use_sim_time}],
    )

    # Static TF: world → sim_camera (overhead fixed camera)
    camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher_camera",
        output="log",
        arguments=["0.0", "0.0", "1.5", "3.1416", "0.0", "0.0",
                   "base_link", "sim_camera"],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # Cup detector 
    cup_detector = Node(
        package="so101_state_machine",
        executable="cup_detector",
        name="cup_detector",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    return [
        world_to_robot_tf,       
        camera_tf,               
        robot_state_publisher,
        ros2_control_node,       
        spawn_jsb,
        *spawners,
        move_group,
        rviz2,
        cup_detector,
    ]

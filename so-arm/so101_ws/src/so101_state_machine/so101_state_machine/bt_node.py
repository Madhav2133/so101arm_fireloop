#!/usr/bin/env python3
"""
Pick-and-place behaviour tree for SO101 arm.

Sequence:
  1. OpenGripper
  2. Grabbing       -- detects red cup via /red_cup_pose, moves to it, closes gripper
  3. AttachCube     -- provided
  4. MoveToBoxPosition
  5. DetachCube     -- provided
  6. OpenGripper
"""

import copy
import time
import threading

import rclpy
import rclpy.executors
from rclpy.node import Node
from rclpy.action import ActionClient
import py_trees

from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import tf2_ros
import tf2_geometry_msgs

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    Constraints,
    JointConstraint,
    BoundingVolume,
    PositionConstraint,
    WorkspaceParameters,
)
from shape_msgs.msg import SolidPrimitive


# Robot config
ARM_GROUP     = "arm"
BASE_FRAME    = "base_link"
EEF_LINK      = "gripper_frame_link"
GRIPPER_JOINT = "gripper"
GRIPPER_OPEN  = 1.0
GRIPPER_CLOSE = 0.01

# Box drop joint angles - trail position
BOX_DROP_JOINTS = {
    "shoulder_pan":   0.0,
    "shoulder_lift": -0.5,
    "elbow_flex":     1.0,
    "wrist_flex":    -0.5,
    "wrist_roll":     0.0,
}

'''
Grasp pose offsets applied to the detected cup position in base_link frame.
X/Y come from camera detection. Z is fixed -- the TF gives unreliable Z
due to the camera frame orientation, so i added the offsets to the known grasp height.
'''
GRASP_X_OFFSET = 0.15
GRASP_Y_OFFSET = 0.02
GRASP_Z        = 0.13

POSE_TIMEOUT = 15.0
MOVE_TIMEOUT = 120.0


class ArmController:
# Handles arm and gripper motion. All calls are non-blocking

    def __init__(self, node: Node):
        self.node = node

        # Dedicated node and executor for the action client to avoid
        # conflicts with the main BT spin loop
        self._action_node = rclpy.create_node("arm_controller_action")
        self._executor    = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self._action_node)
        threading.Thread(target=self._executor.spin, daemon=True).start()

        self._client = ActionClient(self._action_node, MoveGroup, "/move_action")
        self.node.get_logger().info("Waiting for /move_action...")
        self._client.wait_for_server()
        self.node.get_logger().info("/move_action ready.")

        self._gripper_pub = node.create_publisher(
            JointTrajectory, "/gripper_controller/joint_trajectory", 10
        )

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, node)

    def gripper_async(self, position: float, result: dict):
        def _run():
            traj = JointTrajectory()
            traj.joint_names = [GRIPPER_JOINT]
            pt = JointTrajectoryPoint()
            pt.positions = [position]
            pt.time_from_start = Duration(sec=1, nanosec=0)
            traj.points = [pt]
            self._gripper_pub.publish(traj)
            time.sleep(1.5)
            result.update({"done": True, "success": True})
        threading.Thread(target=_run, daemon=True).start()

    def arm_pose_async(self, pose: PoseStamped, result: dict):
        self._send(self._pose_request(pose), result)

    def arm_joints_async(self, joints: dict, result: dict):
        self._send(self._joint_request(joints), result)

    def transform_to_base(self, pose: PoseStamped) -> PoseStamped | None:
        try:
            pose.header.stamp = rclpy.time.Time().to_msg()
            return self.tf_buffer.transform(
                pose, BASE_FRAME, timeout=rclpy.duration.Duration(seconds=1.0)
            )
        except Exception as e:
            self.node.get_logger().warn(f"TF failed: {e}")
            return None

    def _workspace(self):
        ws = WorkspaceParameters()
        ws.header.frame_id = BASE_FRAME
        ws.min_corner.x = ws.min_corner.y = ws.min_corner.z = -1.5
        ws.max_corner.x = ws.max_corner.y = ws.max_corner.z =  1.5
        return ws

    def _joint_request(self, joints: dict) -> MotionPlanRequest:
        req = MotionPlanRequest()
        req.group_name                      = ARM_GROUP
        req.num_planning_attempts           = 5
        req.allowed_planning_time           = 5.0
        req.max_velocity_scaling_factor     = 0.5
        req.max_acceleration_scaling_factor = 0.5
        req.workspace_parameters            = self._workspace()

        c = Constraints()
        for name, pos in joints.items():
            jc = JointConstraint()
            jc.joint_name      = name
            jc.position        = pos
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight          = 1.0
            c.joint_constraints.append(jc)
        req.goal_constraints = [c]
        return req

    def _pose_request(self, pose: PoseStamped) -> MotionPlanRequest:
        req = MotionPlanRequest()
        req.group_name                      = ARM_GROUP
        req.num_planning_attempts           = 10
        req.allowed_planning_time           = 10.0
        req.max_velocity_scaling_factor     = 0.5
        req.max_acceleration_scaling_factor = 0.5
        req.workspace_parameters            = self._workspace()

        sphere = SolidPrimitive()
        sphere.type       = SolidPrimitive.SPHERE
        sphere.dimensions = [0.08]

        bv = BoundingVolume()
        bv.primitives      = [sphere]
        bv.primitive_poses = [pose.pose]

        pc = PositionConstraint()
        pc.header.frame_id   = BASE_FRAME
        pc.link_name         = EEF_LINK
        pc.constraint_region = bv
        pc.weight            = 1.0

        c = Constraints()
        c.position_constraints = [pc]
        req.goal_constraints = [c]
        return req

    def _send(self, plan_request: MotionPlanRequest, result: dict):
        goal = MoveGroup.Goal()
        goal.request                          = plan_request
        goal.planning_options.plan_only       = False
        goal.planning_options.replan          = False
        goal.planning_options.replan_attempts = 0
        goal.planning_options.look_around     = False

        def _run():
            future   = self._client.send_goal_async(goal)
            deadline = time.monotonic() + MOVE_TIMEOUT

            while not future.done():
                if time.monotonic() > deadline:
                    result.update({"done": True, "success": False})
                    return
                time.sleep(0.05)

            handle = future.result()
            if not handle.accepted:
                self.node.get_logger().warn("Goal rejected")
                result.update({"done": True, "success": False})
                return

            res_future = handle.get_result_async()
            while not res_future.done():
                if time.monotonic() > deadline:
                    # Arm was executing -- assume it reached the target
                    self.node.get_logger().warn("Result timeout, assuming success")
                    result.update({"done": True, "success": True})
                    return
                time.sleep(0.05)

            try:
                val = res_future.result().result.error_code.val
                self.node.get_logger().info(f"MoveGroup error_code={val}")
                result.update({"done": True, "success": (val == 1)})
            except Exception as e:
                self.node.get_logger().warn(f"Result parse error: {e}")
                result.update({"done": True, "success": False})

        threading.Thread(target=_run, daemon=True).start()


class OpenGripper(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, node: Node, arm_ctrl: ArmController):
        super().__init__(name)
        self.node     = node
        self.arm_ctrl = arm_ctrl
        self._result  = {}

    def initialise(self):
        self._result = {"done": False, "success": False}
        self.node.get_logger().info(f"[{self.name}] Opening gripper")
        self.arm_ctrl.gripper_async(GRIPPER_OPEN, self._result)

    def update(self):
        if not self._result.get("done"):
            return py_trees.common.Status.RUNNING
        if self._result["success"]:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class Grabbing(py_trees.behaviour.Behaviour):
# Subscribes to /red_cup_pose, transforms the pose to base_link, moves the arm to the grasp position, then closes the gripper.

    WAITING  = "WAITING"
    MOVING   = "MOVING"
    GRIPPING = "GRIPPING"

    def __init__(self, name: str, node: Node, arm_ctrl: ArmController):
        super().__init__(name)
        self.node     = node
        self.arm_ctrl = arm_ctrl
        self._state   = self.WAITING
        self._result  = {}
        self._cup_pose: PoseStamped | None = None
        self._wait_start = 0.0

        self._sub = node.create_subscription(
            PoseStamped, "/red_cup_pose", self._pose_cb, 10
        )

    def _pose_cb(self, msg: PoseStamped):
        if self._cup_pose is None:
            self._cup_pose = msg
            self.node.destroy_subscription(self._sub)
            self.node.get_logger().info("[Grabbing] Cup pose received")

    def initialise(self):
        self._state      = self.WAITING
        self._result     = {}
        self._wait_start = time.monotonic()
        self.node.get_logger().info("[Grabbing] Waiting for cup pose...")

    def update(self):
        if self._state == self.WAITING:
            if self._cup_pose is None:
                if time.monotonic() - self._wait_start > POSE_TIMEOUT:
                    self.node.get_logger().error("[Grabbing] Timeout waiting for pose")
                    return py_trees.common.Status.FAILURE
                return py_trees.common.Status.RUNNING

            base_pose = self.arm_ctrl.transform_to_base(self._cup_pose)
            if base_pose is None:
                return py_trees.common.Status.RUNNING

            grasp = copy.deepcopy(base_pose)
            grasp.pose.position.x += GRASP_X_OFFSET
            grasp.pose.position.y += GRASP_Y_OFFSET
            grasp.pose.position.z  = GRASP_Z

            self.node.get_logger().info(
                f"[Grabbing] Moving to "
                f"x={grasp.pose.position.x:.3f} "
                f"y={grasp.pose.position.y:.3f} "
                f"z={grasp.pose.position.z:.3f}"
            )
            self._result = {"done": False, "success": False}
            self.arm_ctrl.arm_pose_async(grasp, self._result)
            self._state = self.MOVING
            return py_trees.common.Status.RUNNING

        if self._state == self.MOVING:
            if not self._result.get("done"):
                return py_trees.common.Status.RUNNING
            if not self._result["success"]:
                self.node.get_logger().warn("[Grabbing] Move failed")
                return py_trees.common.Status.FAILURE
            self.node.get_logger().info("[Grabbing] Closing gripper")
            self._result = {"done": False, "success": False}
            self.arm_ctrl.gripper_async(GRIPPER_CLOSE, self._result)
            self._state = self.GRIPPING
            return py_trees.common.Status.RUNNING

        if self._state == self.GRIPPING:
            if not self._result.get("done"):
                return py_trees.common.Status.RUNNING
            self.node.get_logger().info("[Grabbing] Success")
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.FAILURE


class MoveToBoxPosition(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, node: Node, arm_ctrl: ArmController):
        super().__init__(name)
        self.node     = node
        self.arm_ctrl = arm_ctrl
        self._result  = {}

    def initialise(self):
        self._result = {"done": False, "success": False}
        self.node.get_logger().info("[MoveToBox] Moving to drop position")
        self.arm_ctrl.arm_joints_async(BOX_DROP_JOINTS, self._result)

    def update(self):
        if not self._result.get("done"):
            return py_trees.common.Status.RUNNING
        if self._result["success"]:
            self.node.get_logger().info("[MoveToBox] Success")
            return py_trees.common.Status.SUCCESS
        self.node.get_logger().warn("[MoveToBox] Failed")
        return py_trees.common.Status.FAILURE


class AttachDetachCube(py_trees.behaviour.Behaviour):
    """Provided -- publishes attach/detach signal to Isaac Sim."""

    def __init__(self, name, node, topic_name, attach, delay_sec=1.0):
        super().__init__(name)
        self.node      = node
        self.attach    = attach
        self.delay_sec = delay_sec
        self.pub       = node.create_publisher(Bool, topic_name, 10)
        self._start    = None
        self._done     = False

    def initialise(self):
        self._start = time.monotonic()
        self._done  = False

    def update(self):
        if not self._done and (time.monotonic() - self._start) >= self.delay_sec:
            msg = Bool()
            msg.data = self.attach
            self.pub.publish(msg)
            self.node.get_logger().info(f"Isaac attach={self.attach}")
            self._done = True
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


def create_tree(node: Node, arm_ctrl: ArmController):
    ATTACH_TOPIC = "/robot/isaac_attach_cube"
    RETRIES      = 2

    def retried(child):
        return py_trees.decorators.Retry(f"Retry_{child.name}", child, RETRIES)

    seq = py_trees.composites.Sequence(name="PickAndPlace", memory=True)
    seq.add_children([
        retried(OpenGripper("OpenGripper1", node, arm_ctrl)),
        retried(Grabbing("Grabbing", node, arm_ctrl)),
        AttachDetachCube("Attach", node, ATTACH_TOPIC, attach=True,  delay_sec=0.5),
        retried(MoveToBoxPosition("MoveToBox", node, arm_ctrl)),
        AttachDetachCube("Detach", node, ATTACH_TOPIC, attach=False, delay_sec=0.5),
        retried(OpenGripper("OpenGripper2", node, arm_ctrl)),
    ])

    return py_trees.decorators.OneShot(
        name="RunOnce",
        child=seq,
        policy=py_trees.common.OneShotPolicy.ON_COMPLETION,
    )


class BTNode(Node):
    def __init__(self):
        super().__init__("bt_interview_template_node")
        self.get_logger().info("Initialising ArmController...")
        self.arm_ctrl = ArmController(self)
        self.tree  = py_trees.trees.BehaviourTree(create_tree(self, self.arm_ctrl))
        self.timer = self.create_timer(0.1, self.tree.tick)
        self.get_logger().info("BT node started.")


def main():
    rclpy.init()
    node = BTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

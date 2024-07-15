class AssembledRobot(object):
    def __init__(self, airbot_player, dt, default_joints):
        self.robot = airbot_player
        self.arm_joints_num = 6
        self.all_joints_num = 7
        self.dt = dt
        self.default_joints = default_joints
        self.default_velocities = [1.0] * self.all_joints_num
        self.end_effector_open = 1
        self.end_effector_close = 0

    def get_current_joint_positions(self):
        return self.robot.get_current_joint_q() + [self.robot.get_current_end()]

    def get_current_joint_velocities(self):
        return self.robot.get_current_joint_v() + [self.robot.get_current_end_v()]
    
    def get_current_joint_efforts(self):
        return self.robot.get_current_joint_t() + [self.robot.get_current_end_t()]

    def set_joint_position_target(self, qpos, qvel=None, blocking=False):  # TODO: add blocking
        if qvel is None:
            qvel = self.default_velocities
        self.robot.set_target_joint_q(qpos[:self.arm_joints_num], blocking, qvel[0])
        if len(qpos) == self.all_joints_num:
            # 若不默认归一化，则需要对末端进行归一化操作
            self.robot.set_target_end(qpos[self.arm_joints_num])
    
    def set_joint_velocity_target(self, qvel, blocking=False):
        self.robot.set_target_joint_v(qvel[:self.arm_joints_num])
        if len(qvel) == self.all_joints_num:
            self.robot.set_target_end_v(qvel[self.arm_joints_num])
    
    def set_joint_effort_target(self, qeffort, blocking=False):
        self.robot.set_target_joint_t(qeffort[:self.arm_joints_num])
        if len(qeffort) == self.all_joints_num:
            self.robot.set_target_end_t(qeffort[self.arm_joints_num])

    def set_end_effector_value(self, value):
        # 若不默认归一化，则需要对末端进行归一化操作
        self.robot.set_target_end(value)


class AssembledFakeRobot(object):
    real_camera = False
    def __init__(self, dt, default_joints):
        self.robot = "fake robot"
        self.arm_joints_num = 6
        self.all_joints_num = 7
        self.dt = dt
        self.default_joints = default_joints
        self.end_effector_open = 1
        self.end_effector_close = 0
        assert len(default_joints) == self.all_joints_num

    def get_current_joint_positions(self):
        return self.default_joints

    def get_current_joint_velocities(self):
        return self.default_joints
    
    def get_current_joint_efforts(self):
        return self.default_joints

    def set_joint_position_target(self, qpos, qvel=None, blocking=False):  # TODO: add blocking
        print(f"Setting joint position target to {qpos}")
    
    def set_joint_velocity_target(self, qvel, blocking=False):
        print(f"Setting joint velocity target to {qvel}")
    
    def set_joint_effort_target(self, qeffort, blocking=False):
        print(f"Setting joint effort target to {qeffort}")

    def set_end_effector_value(self, value):
        print(f"Setting end effector value to {value}")


try:
    import rospy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray
    import numpy as np
    from threading import Thread

    from robot_tools.datar import get_values_by_names
except ImportError as e:
    print(f"Error: {e}")




  
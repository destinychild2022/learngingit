import gymnasium as gym
import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from gymnasium import spaces

import time
import numpy as np
import collections
import dm_env
import torch
from typing import List
from einops import rearrange
from custom_robot import AssembledRobot, AssembledFakeRobot

from gym_aloha.constants import (
    ACTIONS,
    ASSETS_DIR,
    DT,
    JOINTS,
)
from gym_aloha.tasks.sim import BOX_POSE, InsertionTask, TransferCubeTask
from gym_aloha.tasks.sim_end_effector import (
    InsertionEndEffectorTask,
    TransferCubeEndEffectorTask,
)
from gym_aloha.utils import sample_box_pose, sample_insertion_pose


class AlohaEnv(gym.Env):
    # TODO(aliberts): add "human" render_mode
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        task,
        obs_type="pixels",
        render_mode="rgb_array",
        observation_width=640,
        observation_height=480,
        visualization_width=640,
        visualization_height=480,
    ):
        super().__init__()
        self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height

        self.base_action=None
        self.get_obs=True
        self.sleep_time=0

        self.fake=False
        self.sleep_time=0
    

        self._env = self._make_env_task(self.task)

        if self.obs_type == "state":
            raise NotImplementedError()
            self.observation_space = spaces.Box(
                low=np.array([0] * len(JOINTS)),  # ???
                high=np.array([255] * len(JOINTS)),  # ???
                dtype=np.float64,
            )
        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "top": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    )
                }
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                            "top": spaces.Box(
                                low=0,
                                high=255,
                                shape=(self.observation_height, self.observation_width, 3),
                                dtype=np.uint8,
                            )
                        }
                    ),
                    "agent_pos": spaces.Box(
                        low=-1000.0,
                        high=1000.0,
                        shape=(len(JOINTS),),
                        dtype=np.float64,
                    ),
                }
            )

        self.action_space = spaces.Box(low=-1, high=1, shape=(len(ACTIONS),), dtype=np.float32)

    def render(self):
        return self._render(visualize=True)

    def _render(self, visualize=False):
        assert self.render_mode == "rgb_array"
        width, height = (
            (self.visualization_width, self.visualization_height)
            if visualize
            else (self.observation_width, self.observation_height)
        )
        # if mode in ["visualize", "human"]:
        #     height, width = self.visualize_height, self.visualize_width
        # elif mode == "rgb_array":
        #     height, width = self.observation_height, self.observation_width
        # else:
        #     raise ValueError(mode)
        # TODO(rcadene): render and visualizer several cameras (e.g. angle, front_close)
        image = self._env.physics.render(height=height, width=width, camera_id="top")
        return image

    def _make_env_task(self, task_name):
        # time limit is controlled by StepCounter in env factory
        time_limit = float("inf")

        if "transfer_cube" in task_name:
            xml_path = ASSETS_DIR / "bimanual_viperx_transfer_cube.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = TransferCubeTask()
        elif "insertion" in task_name:
            xml_path = ASSETS_DIR / "bimanual_viperx_insertion.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = InsertionTask()
        elif "end_effector_transfer_cube" in task_name:
            raise NotImplementedError()
            xml_path = ASSETS_DIR / "bimanual_viperx_end_effector_transfer_cube.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = TransferCubeEndEffectorTask()
        elif "end_effector_insertion" in task_name:
            raise NotImplementedError()
            xml_path = ASSETS_DIR / "bimanual_viperx_end_effector_insertion.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = InsertionEndEffectorTask()
        else:
            raise NotImplementedError(task_name)

        env = control.Environment(
            physics, task, time_limit, control_timestep=DT, n_sub_steps=None, flat_observation=False
        )
        return env

    def _format_raw_obs(self, raw_obs):
        if self.obs_type == "state":
            raise NotImplementedError()
        elif self.obs_type == "pixels":
            obs = {"top": raw_obs["images"]["top"].copy()}
        elif self.obs_type == "pixels_agent_pos":
            obs = {
                "pixels": {"top": raw_obs["images"]["top"].copy()},
                "agent_pos": raw_obs["qpos"],
            }
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # TODO(rcadene): how to seed the env?
        if seed is not None:
            self._env.task.random.seed(seed)
            self._env.task._random = np.random.RandomState(seed)

        # TODO(rcadene): do not use global variable for this
        if "transfer_cube" in self.task:
            BOX_POSE[0] = sample_box_pose(seed)  # used in sim reset
        elif "insertion" in self.task:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose(seed))  # used in sim reset
        else:
            raise ValueError(self.task)

        raw_obs = self._env.reset()

        observation = self._format_raw_obs(raw_obs.observation)

        info = {"is_success": False}
        return observation, info

    def step(self, action):
        assert action.ndim == 1
        # TODO(rcadene): add info["is_success"] and info["success"] ?

        _, reward, _, raw_obs = self._env.step(action)

        # TODO(rcadene): add an enum
        terminated = is_success = reward == 4

        info = {"is_success": is_success}

        observation = self._format_raw_obs(raw_obs)

        truncated = False
        return observation, reward, terminated, truncated, info

    

    
    def set_reset_position(self, reset_position):
        self.reset_position = reset_position

    def setup_robots(self):
        pass

    def setup_base(self):
        pass

    def get_qpos(self, normalize_gripper=False):
        """7 dof: 6 arm joints + 1 gripper joint"""
        qpos = []
        for airbot in self.airbot_players:
            qpos.append(airbot.get_current_joint_positions())
        return np.hstack(qpos)

    def get_qvel(self):
        qvel = []
        for airbot in self.airbot_players:
            qvel.append(airbot.get_current_joint_velocities())
        return np.hstack(qvel)

    def get_effort(self):
        effort = []
        for airbot in self.airbot_players:
            effort.append(airbot.get_current_joint_efforts())
        return np.hstack(effort)

    def get_images(self):
        return self.image_recorder.get_images()

    def get_base_vel(self):
        raise NotImplementedError
        vel, right_vel = 0.1, 0.1
        right_vel = -right_vel  # right wheel is inverted
        base_linear_vel = (vel + right_vel) * self.wheel_r / 2
        base_angular_vel = (right_vel - vel) * self.wheel_r / self.base_r

        return np.array([base_linear_vel, base_angular_vel])

    def get_tracer_vel(self):
        raise NotImplementedError
        linear_vel, angular_vel = 0.1, 0.1
        return np.array([linear_vel, angular_vel])

    def set_gripper_pose(self, gripper_desired_pos_normalized):
        for airbot in self.airbot_players:
            airbot.set_end_effector_value(gripper_desired_pos_normalized)

    def _reset_joints(self):
        if self.reset_position is None:
            reset_position = [robot.default_joints[:6] for robot in self.airbot_players]
        else:
            reset_position = [self.reset_position[:6]] * self.robot_num
        move_arms(self.airbot_players, reset_position, move_time=1)

    def _reset_gripper(self):
        """Set to position mode and do position resets: first open then close."""
        if self.reset_position is None:
            # move_grippers(self.airbot_players, self.eefs_open / 2, move_time=0.5)
            move_grippers(self.airbot_players, self.eefs_open, move_time=1)
        else:
            move_grippers(
                self.airbot_players,
                [self.reset_position[6]] * self.robot_num,
                move_time=1,
            )

    def get_observation(self, get_tracer_vel=False):
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos()
        # obs['qvel'] = self.get_qvel()
        # obs['effort'] = self.get_effort()
        obs["images"] = self.get_images()
        if self.use_base:
            obs["base_vel"] = self.get_base_vel()
        if get_tracer_vel:
            obs["tracer_vel"] = self.get_tracer_vel()
        return obs

    def get_reward(self):
        return 0

    def reset(self, seed=None, options=None)->dict:
        if not self.fake:
            self._reset_joints()
            self._reset_gripper()
            time.sleep(self.sleep_time)
        obs = self.get_observation()
        info = {"is_success": False}
        return obs, info
    
    def step(self, action, get_tracer_vel=False)->dict:
        action = action
        for index, robot in enumerate(self.airbot_players):
            jn = robot.all_joints_num
            robot.set_joint_position_target(
                action[jn * index : jn * (index + 1)], blocking=False
            )
        time.sleep(self.sleep_time)
        if self.base_action is not None:
            raise NotImplementedError
            # linear_vel_limit = 1.5
            # angular_vel_limit = 1.5
            # base_action_linear = np.clip(base_action[0], -linear_vel_limit, linear_vel_limit)
            # base_action_angular = np.clip(base_action[1], -angular_vel_limit, angular_vel_limit)
            base_action_linear, base_action_angular = base_action
            self.tracer.SetMotionCommand(
                linear_vel=base_action_linear, angular_vel=base_action_angular
            )
        # time.sleep(DT)
        if self.get_obs:
            obs = self.get_observation(get_tracer_vel)
        else:
            obs = None
        truncated = False
        reward=self.get_reward()
        terminated = is_success = reward
        info = {"is_success": is_success}
        return obs, reward, terminated, truncated, info

        



    def close(self):
        pass



def get_arm_joint_positions(bot: AssembledRobot):
    return bot.get_current_joint_positions()[:6]


def get_arm_gripper_positions(bot: AssembledRobot):
    return bot.get_current_joint_positions()[6]

def move_arms(bot_list: List[AssembledRobot], target_pose_list, move_time=1):
    DT = max([bot.dt for bot in bot_list])
    num_steps = int(move_time / DT)
    curr_pose_list = [get_arm_joint_positions(bot) for bot in bot_list]
    # 进行关节插值，保证平稳运动
    traj_list = [
        np.linspace(curr_pose, target_pose, num_steps)
        for curr_pose, target_pose in zip(curr_pose_list, target_pose_list)
    ]
    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            # blocking为False用于多台臂可以同时移动
            bot.set_joint_position_target(traj_list[bot_id][t], blocking=False)
        time.sleep(DT)


def move_grippers(bot_list: List[AssembledRobot], target_pose_list, move_time):
    DT = max([bot.dt for bot in bot_list])
    num_steps = int(move_time / DT)
    curr_pose_list = [get_arm_gripper_positions(bot) for bot in bot_list]
    traj_list = [
        np.linspace(curr_pose, target_pose, num_steps)
        for curr_pose, target_pose in zip(curr_pose_list, target_pose_list)
    ]
    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            bot.set_end_effector_value(traj_list[bot_id][t])
        time.sleep(DT)






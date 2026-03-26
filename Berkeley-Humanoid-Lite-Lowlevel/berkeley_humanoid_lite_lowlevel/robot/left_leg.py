# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

import time

from omegaconf import OmegaConf
import numpy as np

import berkeley_humanoid_lite_lowlevel.recoil as recoil


class State:
    INVALID = 0
    IDLE = 1
    RL_INIT = 2
    RL_RUNNING = 3


def linear_interpolate(start: np.ndarray, end: np.ndarray, percentage: float) -> np.ndarray:
    percentage = min(max(percentage, 0.0), 1.0)
    target = start * (1.0 - percentage) + end * percentage
    return target


class HumanoidLeft:
    def __init__(self):
        # 왼쪽 다리 CAN만 사용
        self.left_leg_transport = recoil.Bus("can0")

        # 왼쪽 다리 6관절
        self.joints = [
            (self.left_leg_transport, 1, "left_hip_roll_joint"),
            (self.left_leg_transport, 3, "left_hip_yaw_joint"),
            (self.left_leg_transport, 5, "left_hip_pitch_joint"),
            (self.left_leg_transport, 7, "left_knee_pitch_joint"),
            (self.left_leg_transport, 11, "left_ankle_pitch_joint"),
            (self.left_leg_transport, 13, "left_ankle_roll_joint"),
        ]

        # IMU 사용 안 함
        self.imu = None

        # 게임패드 사용 안 함
        self.command_controller = None

        self.state = State.IDLE
        self.next_state = State.IDLE

        # 왼쪽 다리 6관절 초기 자세
        self.rl_init_positions = np.array([
            0.0, 0.0, -0.2,
            0.4,
            -0.3, 0.0,
        ], dtype=np.float32)

        # 관절 방향 보정
        self.joint_axis_directions = np.array([
            -1, 1, -1,
            -1,
            -1, 1,
        ], dtype=np.float32)

        # 오프셋 초기값
        self.position_offsets = np.array([
            0.0, 0.0, 0.0,
            0.0,
            0.0, 0.0,
        ], dtype=np.float32)

        # obs = quat(4) + gyro(3) + joint_pos(6) + joint_vel(6) + mode(1) + cmd_vel(3)
        self.n_lowlevel_states = 4 + 3 + 6 + 6 + 1 + 3
        self.lowlevel_states = np.zeros(self.n_lowlevel_states, dtype=np.float32)

        self.joint_velocity_target = np.zeros(len(self.joints), dtype=np.float32)
        self.joint_position_target = np.zeros(len(self.joints), dtype=np.float32)
        self.joint_position_measured = np.zeros(len(self.joints), dtype=np.float32)
        self.joint_velocity_measured = np.zeros(len(self.joints), dtype=np.float32)

        self.init_percentage = 0.0
        self.starting_positions = np.zeros_like(self.joint_position_target, dtype=np.float32)

        # calibration_left.yaml 우선, 없으면 calibration.yaml 시도
        loaded = False
        for config_path, key in [
            ("calibration_left.yaml", "position_offsets_left"),
            ("calibration.yaml", "position_offsets"),
        ]:
            try:
                with open(config_path, "r") as f:
                    config = OmegaConf.load(f)

                position_offsets = np.array(config.get(key, None), dtype=np.float32)

                if position_offsets.shape[0] == len(self.joints):
                    self.position_offsets[:] = position_offsets
                    print(f"Loaded calibration from {config_path} ({key})")
                    loaded = True
                    break
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"Warning: failed to load {config_path}: {e}")

        if not loaded:
            print("Warning: calibration file not found or size mismatch. Using zero offsets.")

    def enter_damping(self):
        self.joint_kp = np.zeros((len(self.joints),), dtype=np.float32)
        self.joint_kd = np.zeros((len(self.joints),), dtype=np.float32)
        self.torque_limit = np.zeros((len(self.joints),), dtype=np.float32)

        self.joint_kp[:] = 20
        self.joint_kd[:] = 2
        self.torque_limit[:] = 4

        for i, entry in enumerate(self.joints):
            bus, device_id, joint_name = entry

            print(f"Initializing joint {joint_name}:")
            print(f"  kp: {self.joint_kp[i]}, kd: {self.joint_kd[i]}, torque limit: {self.torque_limit[i]}")

            bus.set_mode(device_id, recoil.Mode.IDLE)
            time.sleep(0.001)
            bus.write_position_kp(device_id, self.joint_kp[i])
            time.sleep(0.001)
            bus.write_position_kd(device_id, self.joint_kd[i])
            time.sleep(0.001)
            bus.write_torque_limit(device_id, self.torque_limit[i])
            time.sleep(0.001)
            bus.feed(device_id)
            bus.set_mode(device_id, recoil.Mode.DAMPING)

        print("Motors enabled")

    def stop(self):
        if self.imu is not None:
            self.imu.stop()

        if self.command_controller is not None:
            self.command_controller.stop()

        for entry in self.joints:
            bus, device_id, _ = entry
            bus.set_mode(device_id, recoil.Mode.DAMPING)

        print("Entered damping mode. Press Ctrl+C again to exit.\n")

        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("Exiting damping mode.")

        for entry in self.joints:
            bus, device_id, _ = entry
            bus.set_mode(device_id, recoil.Mode.IDLE)

        self.left_leg_transport.stop()

    def get_observations(self) -> np.ndarray:
        imu_quaternion = self.lowlevel_states[0:4]
        imu_angular_velocity = self.lowlevel_states[4:7]
        joint_positions = self.lowlevel_states[7:13]
        joint_velocities = self.lowlevel_states[13:19]
        mode = self.lowlevel_states[19:20]
        velocity_commands = self.lowlevel_states[20:23]

        # IMU 없으므로 기본값
        if self.imu is not None:
            imu_quaternion[:] = self.imu.quaternion[:]
            imu_angular_velocity[:] = np.deg2rad(self.imu.angular_velocity[:])
        else:
            imu_quaternion[:] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            imu_angular_velocity[:] = 0.0

        joint_positions[:] = self.joint_position_measured[:]
        joint_velocities[:] = self.joint_velocity_measured[:]

        # 게임패드 없으므로 기본값
        if self.command_controller is not None:
            mode[0] = self.command_controller.commands["mode_switch"]
            velocity_commands[0] = self.command_controller.commands["velocity_x"]
            velocity_commands[1] = self.command_controller.commands["velocity_y"]
            velocity_commands[2] = self.command_controller.commands["velocity_yaw"]
            self.next_state = self.command_controller.commands["mode_switch"]
        else:
            mode[0] = 0.0
            velocity_commands[:] = 0.0
            self.next_state = State.IDLE

        return self.lowlevel_states

    def update_joint(self, joint_id):
        position_target = (
            self.joint_position_target[joint_id] + self.position_offsets[joint_id]
        ) * self.joint_axis_directions[joint_id]

        bus, device_id, _ = self.joints[joint_id]
        bus.transmit_pdo_2(
            device_id,
            position_target=position_target,
            velocity_target=0.0
        )

        position_measured, velocity_measured = bus.receive_pdo_2(device_id)

        if position_measured is not None:
            self.joint_position_measured[joint_id] = (
                position_measured * self.joint_axis_directions[joint_id]
            ) - self.position_offsets[joint_id]

        if velocity_measured is not None:
            self.joint_velocity_measured[joint_id] = (
                velocity_measured * self.joint_axis_directions[joint_id]
            )

    def update_joints(self):
        for joint_id in range(len(self.joints)):
            self.update_joint(joint_id)

    def reset(self):
        obs = self.get_observations()
        return obs

    def step(self, actions: np.ndarray):
        match self.state:
            case State.IDLE:
                self.joint_position_target[:] = self.joint_position_measured[:]

                if self.next_state == State.RL_INIT:
                    print("Switching to RL initialization mode")
                    self.state = self.next_state

                    for entry in self.joints:
                        bus, device_id, _ = entry
                        bus.feed(device_id)
                        bus.set_mode(device_id, recoil.Mode.POSITION)

                    self.starting_positions = self.joint_position_target.copy()
                    self.init_percentage = 0.0

            case State.RL_INIT:
                print(f"init: {self.init_percentage:.2f}")
                if self.init_percentage < 1.0:
                    self.init_percentage += 1 / 100.0
                    self.init_percentage = min(self.init_percentage, 1.0)

                    self.joint_position_target = linear_interpolate(
                        self.starting_positions,
                        self.rl_init_positions,
                        self.init_percentage,
                    )
                else:
                    if self.next_state == State.RL_RUNNING:
                        print("Switching to RL running mode")
                        self.state = self.next_state

                    if self.next_state == State.IDLE:
                        print("Switching to idle mode")
                        self.state = self.next_state

                        for entry in self.joints:
                            bus, device_id, _ = entry
                            bus.set_mode(device_id, recoil.Mode.DAMPING)

            case State.RL_RUNNING:
                for i in range(len(self.joints)):
                    self.joint_position_target[i] = actions[i]

                if self.next_state == State.IDLE:
                    print("Switching to idle mode")
                    self.state = self.next_state

                    for entry in self.joints:
                        bus, device_id, _ = entry
                        bus.set_mode(device_id, recoil.Mode.DAMPING)

        self.update_joints()
        obs = self.get_observations()
        return obs

    def check_connection(self):
        for entry in self.joints:
            bus, device_id, joint_name = entry
            print(f"Pinging {joint_name} ... ", end="\t")
            result = bus.ping(device_id)
            if result:
                print("OK")
            else:
                print("ERROR")
            time.sleep(0.1)
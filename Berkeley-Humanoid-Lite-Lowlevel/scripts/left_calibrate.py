"""
left_calibrate.py

왼쪽 다리 6개 관절만 encoder offset을 캘리브레이션하는 스크립트
IMU 없음 / 게임패드 없음 / Ctrl+C 종료
"""

import time
import numpy as np
import yaml

from berkeley_humanoid_lite_lowlevel.robot.left_leg import HumanoidLeft


robot = HumanoidLeft()

left_joints = robot.joints[:6]

joint_axis_directions = np.array([
    -1, +1, -1,
    -1,
    -1, +1
])

ideal_values = np.array([
    np.deg2rad(-(10)),
    np.deg2rad(+(33.75)),
    np.deg2rad(+(56.25)),
    np.deg2rad(+(0)),
    np.deg2rad(-(45)),
    np.deg2rad(-(15)),
])

print("initial readings (left leg only):")
limit_readings = np.array(
    [joint[0].read_position_measured(joint[1]) for joint in left_joints]
) * joint_axis_directions
print([f"{reading:.2f}" for reading in limit_readings])

print("각 관절을 한쪽 limit까지 손으로 움직이세요.")
print("끝나면 Ctrl+C 를 눌러 저장합니다.")

try:
    while True:
        joint_readings = np.array(
            [joint[0].read_position_measured(joint[1]) for joint in left_joints]
        ) * joint_axis_directions

        limit_readings[0] = min(limit_readings[0], joint_readings[0])
        limit_readings[1] = max(limit_readings[1], joint_readings[1])
        limit_readings[2] = max(limit_readings[2], joint_readings[2])
        limit_readings[3] = min(limit_readings[3], joint_readings[3])
        limit_readings[4] = min(limit_readings[4], joint_readings[4])
        limit_readings[5] = min(limit_readings[5], joint_readings[5])

        print(time.time(), [f"{reading:.2f}" for reading in limit_readings])
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nStopped by user. Saving calibration...")

print("final readings at the limits (left leg only):")
print([f"{limit:.4f}" for limit in limit_readings])

offsets = limit_readings - ideal_values

print("offsets (left leg only):")
print([f"{offset:.4f}" for offset in offsets])

calibration_data = {
    "position_offsets_left": [float(offset) for offset in offsets],
}

with open("calibration_left.yaml", "w") as f:
    yaml.dump(calibration_data, f)

print("Saved to calibration_left.yaml")
robot.stop()

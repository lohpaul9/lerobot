#!/usr/bin/env python

"""
Interactive test using MuJoCo's passive viewer (with full UI).
"""

import logging
import time
from pathlib import Path

import mujoco
import mujoco.viewer
from lerobot.robots.so101_mujoco.configuration_so101_mujoco import SO101MujocoConfig
from lerobot.robots.so101_mujoco.robot_so101_mujoco import SO101MujocoRobot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Test robot with MuJoCo passive viewer."""
    # Create config
    config = SO101MujocoConfig(
        id="so101_test",
        xml_path=Path("gym-hil/gym_hil/assets/SO101/pick_scene.xml"),
    )

    # Create robot
    robot = SO101MujocoRobot(config)
    robot.connect()

    logger.info("\n" + "="*60)
    logger.info("SO-101 MuJoCo Interactive Viewer")
    logger.info("Controls:")
    logger.info("  Arrow keys: Move XY (Up=+X, Down=-X, Left=+Y, Right=-Y)")
    logger.info("  Shift / Shift+R: Move Z up/down")
    logger.info("  Ctrl / Alt: Wrist roll left/right")
    logger.info("  [ / ]: Gripper close/open")
    logger.info("  Mouse: Rotate/zoom camera (MuJoCo viewer)")
    logger.info("  Close viewer window to exit")
    logger.info("="*60 + "\n")

    # Launch passive viewer
    viewer = mujoco.viewer.launch_passive(robot.model, robot.data)

    # Control loop - viewer handles rendering automatically
    step_count = 0
    last_print_time = time.time()

    try:
        while viewer.is_running():
            step_start = time.time()

            # Get keyboard state from viewer
            # MuJoCo viewer doesn't easily expose keyboard state, so for now just idle
            # TODO: Hook into viewer's keyboard callbacks
            keyboard_action = {
                "up": False, "down": False, "left": False, "right": False,
                "shift": False, "shift_r": False, "ctrl": False, "alt": False,
                "[": False, "]": False,
            }
            robot._from_keyboard_to_base_action(keyboard_action)

            # Run control loop
            robot.send_action({})

            # Sync viewer (updates rendering)
            viewer.sync()

            # Print status every second
            if time.time() - last_print_time > 1.0:
                obs = robot.get_observation()
                ee_pos = obs["ee.pos_x"], obs["ee.pos_y"], obs["ee.pos_z"]
                logger.info(
                    f"Step {step_count:5d} | "
                    f"EE: [{ee_pos[0]:6.3f}, {ee_pos[1]:6.3f}, {ee_pos[2]:6.3f}] | "
                    f"Gripper: {obs['gripper.pos']:5.3f}"
                )
                last_print_time = time.time()

            step_count += 1

            # Maintain 30 FPS
            elapsed = time.time() - step_start
            sleep_time = (1.0 / config.record_fps) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        viewer.close()
        robot.disconnect()
        logger.info("Test completed")


if __name__ == "__main__":
    main()

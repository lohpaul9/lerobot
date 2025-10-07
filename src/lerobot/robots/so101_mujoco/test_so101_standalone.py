#!/usr/bin/env python

"""
Standalone test script for SO101MujocoRobot.

This script runs the robot in a simple control loop with keyboard input
to verify that the control logic matches orient_down.py.

Controls:
  W/S: Move in +X/-X
  A/D: Move in +Y/-Y
  E/Q: Move in +Z/-Z
  [/]: Wrist roll left/right
  ,/.: Gripper close/open
  R: Reset to home position
  ESC: Exit

Usage:
  python test_so101_standalone.py
"""

import logging
import time
from pathlib import Path

import glfw
import numpy as np


from lerobot.robots.so101_mujoco.configuration_so101_mujoco import SO101MujocoConfig
from lerobot.robots.so101_mujoco.robot_so101_mujoco import SO101MujocoRobot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KeyboardHandler:
    """Simple keyboard handler using GLFW."""

    def __init__(self, window):
        self.down = set()
        glfw.set_key_callback(window, self._callback)

    def _callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            self.down.add(key)
        elif action == glfw.RELEASE:
            self.down.discard(key)

    def is_pressed(self, key):
        return key in self.down

    def get_action_dict(self) -> dict[str, bool]:
        """Convert GLFW keys to keyboard action dict."""
        return {
            "w": self.is_pressed(glfw.KEY_W),
            "s": self.is_pressed(glfw.KEY_S),
            "a": self.is_pressed(glfw.KEY_A),
            "d": self.is_pressed(glfw.KEY_D),
            "e": self.is_pressed(glfw.KEY_E),
            "q": self.is_pressed(glfw.KEY_Q),
            "[": self.is_pressed(glfw.KEY_LEFT_BRACKET),
            "]": self.is_pressed(glfw.KEY_RIGHT_BRACKET),
            ",": self.is_pressed(glfw.KEY_COMMA),
            ".": self.is_pressed(glfw.KEY_PERIOD),
            "r": self.is_pressed(glfw.KEY_R),
            "esc": self.is_pressed(glfw.KEY_ESCAPE),
        }


class MouseHandler:
    """Mouse handler for camera control."""

    def __init__(self, window):
        self.button_left = False
        self.button_right = False
        self.last_x = 0
        self.last_y = 0
        glfw.set_cursor_pos_callback(window, self._cursor_callback)
        glfw.set_mouse_button_callback(window, self._button_callback)
        glfw.set_scroll_callback(window, self._scroll_callback)

    def _button_callback(self, window, button, action, mods):
        self.button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self.button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS

    def _cursor_callback(self, window, xpos, ypos):
        dx = xpos - self.last_x
        dy = ypos - self.last_y
        self.last_x = xpos
        self.last_y = ypos

        if self.button_left:
            self.azimuth_delta = -dx * 0.3
            self.elevation_delta = dy * 0.3
        elif self.button_right:
            self.zoom_delta = dy * 0.01
        else:
            self.azimuth_delta = 0
            self.elevation_delta = 0
            self.zoom_delta = 0

    def _scroll_callback(self, window, xoffset, yoffset):
        self.zoom_delta = -yoffset * 0.05

    def update_camera(self, cam):
        """Update camera based on mouse input."""
        if hasattr(self, 'azimuth_delta'):
            cam.azimuth += self.azimuth_delta
            self.azimuth_delta = 0
        if hasattr(self, 'elevation_delta'):
            cam.elevation += self.elevation_delta
            cam.elevation = np.clip(cam.elevation, -89, 89)
            self.elevation_delta = 0
        if hasattr(self, 'zoom_delta'):
            cam.distance += self.zoom_delta
            cam.distance = np.clip(cam.distance, 0.5, 5.0)
            self.zoom_delta = 0


def test_robot_control():
    """Test robot control with keyboard input."""
    # Create config
    config = SO101MujocoConfig(
        id="so101_test",
        xml_path=Path("gym-hil/gym_hil/assets/SO101/pick_scene.xml"),
    )

    # Create robot
    robot = SO101MujocoRobot(config)

    # Connect (loads model) - NOTE: This creates a mujoco.Renderer internally
    # which will conflict with GLFW rendering
    logger.info("Connecting to robot...")
    robot.connect()
    logger.info("Robot connected!")

    # Close the robot's renderer to avoid conflict with GLFW
    if robot._renderer is not None:
        try:
            robot._renderer.close()
        except Exception:
            pass
        robot._renderer = None
        logger.info("Closed robot's offscreen renderer for GLFW compatibility")

    # Initialize GLFW window for visualization
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    window = glfw.create_window(1280, 720, "SO-101 MuJoCo Test", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create window")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # Setup keyboard and mouse handlers
    kb = KeyboardHandler(window)
    mouse = MouseHandler(window)

    # Setup MuJoCo viewer (exactly like orient_down.py)
    import mujoco
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 1.3
    cam.azimuth = 140
    cam.elevation = -20
    scene = mujoco.MjvScene(robot.model, maxgeom=10000)
    ctx = mujoco.MjrContext(robot.model, mujoco.mjtFontScale.mjFONTSCALE_150)

    logger.info("\n" + "="*60)
    logger.info("SO-101 Robot Test - Controls:")
    logger.info("  W/S: +X/-X movement")
    logger.info("  A/D: +Y/-Y movement")
    logger.info("  E/Q: +Z/-Z movement")
    logger.info("  [/]: Wrist roll left/right")
    logger.info("  ,/.: Gripper close/open")
    logger.info("  R: Reset to home")
    logger.info("  Left mouse drag: Rotate camera")
    logger.info("  Right mouse drag / scroll: Zoom camera")
    logger.info("  ESC: Exit")
    logger.info("="*60 + "\n")

    # Control loop
    step_count = 0
    last_print_time = time.time()

    try:
        while not glfw.window_should_close(window) and not kb.is_pressed(glfw.KEY_ESCAPE):
            step_start = time.time()

            # Get keyboard input
            keyboard_action = kb.get_action_dict()

            # Handle reset
            if keyboard_action["r"]:
                logger.info("Resetting to home position...")
                robot._initialize_home_position()
                time.sleep(0.5)  # Debounce

            # Convert keyboard to velocities (stores in robot._keyboard_velocities)
            robot._from_keyboard_to_base_action(keyboard_action)

            # Send action (runs control loop and returns final positions)
            action_recorded = robot.send_action({})

            # Get observation
            obs = robot.get_observation()

            # Update camera from mouse
            mouse.update_camera(cam)

            # Print status every second
            if time.time() - last_print_time > 1.0:
                ee_pos = obs["ee.pos_x"], obs["ee.pos_y"], obs["ee.pos_z"]
                logger.info(
                    f"Step {step_count:5d} | "
                    f"EE: [{ee_pos[0]:6.3f}, {ee_pos[1]:6.3f}, {ee_pos[2]:6.3f}] | "
                    f"Gripper: {obs['gripper.pos']:5.3f}"
                )
                last_print_time = time.time()

            # Render (use framebuffer size for retina displays)
            width, height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, width, height)
            mujoco.mjv_updateScene(
                robot.model, robot.data, opt, None, cam,
                mujoco.mjtCatBit.mjCAT_ALL, scene
            )
            mujoco.mjr_render(viewport, scene, ctx)

            glfw.swap_buffers(window)
            glfw.poll_events()

            step_count += 1

            # Maintain 30 FPS (recording rate)
            elapsed = time.time() - step_start
            sleep_time = (1.0 / config.record_fps) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        # Cleanup
        robot.disconnect()
        glfw.terminate()
        logger.info("Test completed successfully")


if __name__ == "__main__":
    test_robot_control()

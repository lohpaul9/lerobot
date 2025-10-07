#!/usr/bin/env python

"""
Simple interactive test without using robot class's renderer.
"""

import logging
import time
from pathlib import Path

import glfw
import mujoco as mj
import numpy as np

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


def main():
    """Test MuJoCo rendering with GLFW."""
    # Load model directly
    model_path = "gym-hil/gym_hil/assets/SO101/pick_scene.xml"
    model = mj.MjModel.from_xml_path(model_path)
    data = mj.MjData(model)

    # Set home position
    JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
    dof_ids = {}
    for joint_name in JOINT_NAMES:
        dof_ids[joint_name] = model.jnt_dofadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)]

    data.qpos[dof_ids['shoulder_pan']] = 0.0
    data.qpos[dof_ids['shoulder_lift']] = np.deg2rad(30)
    data.qpos[dof_ids['elbow_flex']] = np.deg2rad(90)
    data.qpos[dof_ids['wrist_flex']] = np.deg2rad(-40)
    data.qpos[dof_ids['wrist_roll']] = 0.0
    data.qpos[dof_ids['gripper']] = 0.0

    mj.mj_forward(model, data)

    # Initialize GLFW
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    w, h = 1280, 720
    window = glfw.create_window(w, h, "SO-101 Simple Test", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create window")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # Setup keyboard
    kb = KeyboardHandler(window)

    # Setup MuJoCo viewer
    cam = mj.MjvCamera()
    opt = mj.MjvOption()
    mj.mjv_defaultCamera(cam)
    cam.distance = 1.3
    cam.azimuth = 140
    cam.elevation = -20
    scene = mj.MjvScene(model, maxgeom=10000)
    ctx = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150)

    logger.info("\n" + "="*60)
    logger.info("SO-101 Simple Test")
    logger.info("Press ESC to exit")
    logger.info("="*60 + "\n")

    # Main loop
    step_count = 0
    try:
        while not glfw.window_should_close(window) and not kb.is_pressed(glfw.KEY_ESCAPE):
            # Step physics
            mj.mj_step(model, data)

            # Render (use framebuffer size for retina displays)
            width, height = glfw.get_framebuffer_size(window)
            viewport = mj.MjrRect(0, 0, width, height)
            mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL, scene)
            mj.mjr_render(viewport, scene, ctx)

            glfw.swap_buffers(window)
            glfw.poll_events()

            step_count += 1
            if step_count % 100 == 0:
                logger.info(f"Step {step_count}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        glfw.terminate()
        logger.info("Test completed")


if __name__ == "__main__":
    main()

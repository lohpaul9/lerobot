#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Keyboard teleoperator for SO-101 MuJoCo robot.
Returns raw keyboard state (which keys are pressed) for the robot to interpret.
"""

import logging
import os
import sys
from queue import Queue
from typing import Any

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .configuration_keyboard import KeyboardTeleopConfig

PYNPUT_AVAILABLE = True
try:
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        logging.info("No DISPLAY set. Skipping pynput import.")
        raise ImportError("pynput blocked intentionally due to no display.")

    from pynput import keyboard
except ImportError:
    keyboard = None
    PYNPUT_AVAILABLE = False
except Exception as e:
    keyboard = None
    PYNPUT_AVAILABLE = False
    logging.info(f"Could not import pynput: {e}")


class SO101KeyboardTeleop(Teleoperator):
    """
    Keyboard teleop for SO-101 that returns raw key states.

    Returns a dict with boolean values for each key:
    {
        "up": bool, "down": bool, "left": bool, "right": bool,
        "shift": bool, "shift_r": bool, "ctrl": bool, "alt": bool,
        "[": bool, "]": bool, ",": bool, ".": bool,
        "z": bool, "x": bool,  # Alternative wrist roll controls
    }
    """

    name = "so101_keyboard"
    config_class = KeyboardTeleopConfig

    def __init__(self, config: KeyboardTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.event_queue = Queue()
        self.current_pressed = {}
        self.listener = None

    @property
    def action_features(self) -> dict:
        # Return empty since we don't directly control action space
        # The robot's _from_keyboard_to_base_action will handle conversion
        return {}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()

    @property
    def is_calibrated(self) -> bool:
        return True  # No calibration needed

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                "SO101KeyboardTeleop is already connected. Do not run `connect()` twice."
            )

        if PYNPUT_AVAILABLE:
            logging.info("pynput is available - enabling SO-101 keyboard listener.")
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self.listener.start()
        else:
            logging.info("pynput not available - skipping keyboard listener.")
            self.listener = None

    def calibrate(self) -> None:
        pass  # No calibration needed

    def configure(self) -> None:
        pass  # No configuration needed

    def _on_press(self, key):
        # Map special keys
        key_name = None
        if key == keyboard.Key.up:
            key_name = "up"
        elif key == keyboard.Key.down:
            key_name = "down"
        elif key == keyboard.Key.left:
            key_name = "left"
        elif key == keyboard.Key.right:
            key_name = "right"
        elif key == keyboard.Key.shift or key == keyboard.Key.shift_l:
            key_name = "shift"
        elif key == keyboard.Key.shift_r:
            key_name = "shift_r"
        elif key == keyboard.Key.ctrl or key == keyboard.Key.ctrl_l:
            key_name = "ctrl"
        elif key == keyboard.Key.alt or key == keyboard.Key.alt_l:
            key_name = "alt"
        elif hasattr(key, "char") and key.char in ["[", "]", ",", ".", "z", "x"]:
            key_name = key.char

        if key_name:
            self.event_queue.put((key_name, True))

    def _on_release(self, key):
        # Map special keys
        key_name = None
        if key == keyboard.Key.up:
            key_name = "up"
        elif key == keyboard.Key.down:
            key_name = "down"
        elif key == keyboard.Key.left:
            key_name = "left"
        elif key == keyboard.Key.right:
            key_name = "right"
        elif key == keyboard.Key.shift or key == keyboard.Key.shift_l:
            key_name = "shift"
        elif key == keyboard.Key.shift_r:
            key_name = "shift_r"
        elif key == keyboard.Key.ctrl or key == keyboard.Key.ctrl_l:
            key_name = "ctrl"
        elif key == keyboard.Key.alt or key == keyboard.Key.alt_l:
            key_name = "alt"
        elif hasattr(key, "char") and key.char in ["[", "]", ",", ".", "z", "x"]:
            key_name = key.char
        elif key == keyboard.Key.esc:
            logging.info("ESC pressed, disconnecting.")
            self.disconnect()
            return

        if key_name:
            self.event_queue.put((key_name, False))

    def _drain_pressed_keys(self):
        """Process all queued key events to get current state."""
        while not self.event_queue.empty():
            key_name, is_pressed = self.event_queue.get_nowait()
            self.current_pressed[key_name] = is_pressed

    def get_action(self) -> dict[str, Any]:
        """
        Get current keyboard state.

        Returns:
            Dict with boolean values for each key:
            {
                "up": bool, "down": bool, "left": bool, "right": bool,
                "shift": bool, "shift_r": bool, "ctrl": bool, "alt": bool,
                "[": bool, "]": bool,
            }
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "SO101KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        self._drain_pressed_keys()

        # Return current state of all keys we care about
        return {
            "up": self.current_pressed.get("up", False),
            "down": self.current_pressed.get("down", False),
            "left": self.current_pressed.get("left", False),
            "right": self.current_pressed.get("right", False),
            "shift": self.current_pressed.get("shift", False),
            "shift_r": self.current_pressed.get("shift_r", False),
            "ctrl": self.current_pressed.get("ctrl", False),
            "alt": self.current_pressed.get("alt", False),
            "[": self.current_pressed.get("[", False),
            "]": self.current_pressed.get("]", False),
            ",": self.current_pressed.get(",", False),
            ".": self.current_pressed.get(".", False),
            "z": self.current_pressed.get("z", False),
            "x": self.current_pressed.get("x", False),
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass  # No feedback for keyboard

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "SO101KeyboardTeleop is not connected. You need to run `connect()` before `disconnect()`."
            )
        if self.listener is not None:
            self.listener.stop()

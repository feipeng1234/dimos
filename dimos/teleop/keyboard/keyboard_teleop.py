#!/usr/bin/env python3
# Copyright 2025-2026 Dimensional Inc.
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

import os
import threading

import pygame

from dimos.core import Module, Out, rpc
from dimos.msgs.geometry_msgs import Twist, Vector3

# Force X11 driver to avoid OpenGL threading issues
os.environ.setdefault("SDL_VIDEODRIVER", "x11")


class KeyboardTeleop(Module):
    """Pygame-based keyboard control module.

    Outputs standard Twist messages on /cmd_vel for velocity control.
    """

    cmd_vel: Out[Twist]  # Standard velocity commands

    _stop_event: threading.Event
    _thread: threading.Thread | None = None
    _screen: pygame.Surface | None = None
    _clock: pygame.time.Clock | None = None
    _font: pygame.font.Font | None = None
    _has_focus: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._stop_event = threading.Event()

    @rpc
    def start(self) -> bool:
        super().start()

        self._stop_event.clear()

        self._thread = threading.Thread(target=self._pygame_loop, daemon=True)
        self._thread.start()

        return True

    @rpc
    def stop(self) -> None:
        self._publish_stop_twist()

        self._stop_event.set()

        if self._thread is None:
            raise RuntimeError("Cannot stop: thread was never started")
        self._thread.join(2)

        super().stop()

    def _publish_stop_twist(self) -> None:
        stop_twist = Twist()
        stop_twist.linear = Vector3(0, 0, 0)
        stop_twist.angular = Vector3(0, 0, 0)
        self.cmd_vel.publish(stop_twist)

    def _pygame_loop(self) -> None:
        pygame.init()
        self._screen = pygame.display.set_mode((500, 400), pygame.SWSURFACE)
        pygame.display.set_caption("Keyboard Teleop")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.Font(None, 24)

        while not self._stop_event.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._stop_event.set()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # Emergency stop - send zero twist immediately
                        self._publish_stop_twist()
                        print("EMERGENCY STOP!")
                    elif event.key == pygame.K_ESCAPE:
                        # ESC quits
                        self._stop_event.set()
                elif event.type == pygame.ACTIVEEVENT:
                    if getattr(event, "gain", 0) == 0:
                        self._has_focus = False
                        self._publish_stop_twist()
                    else:
                        self._has_focus = True

            keys = pygame.key.get_pressed()

            # Generate Twist message from held keys
            twist = Twist()
            twist.linear = Vector3(0, 0, 0)
            twist.angular = Vector3(0, 0, 0)

            if self._has_focus:
                # Forward/backward (W/S)
                if keys[pygame.K_w]:
                    twist.linear.x += 0.5
                if keys[pygame.K_s]:
                    twist.linear.x -= 0.5

                # Strafe left/right (Q/E)
                if keys[pygame.K_q]:
                    twist.linear.y += 0.5
                if keys[pygame.K_e]:
                    twist.linear.y -= 0.5

                # Turning (A/D)
                if keys[pygame.K_a]:
                    twist.angular.z += 0.8
                if keys[pygame.K_d]:
                    twist.angular.z -= 0.8

                # Apply speed modifiers (Shift = 2x, Ctrl = 0.5x)
                speed_multiplier = 1.0
                if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                    speed_multiplier = 2.0
                elif keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                    speed_multiplier = 0.5

                twist.linear.x *= speed_multiplier
                twist.linear.y *= speed_multiplier
                twist.angular.z *= speed_multiplier

            # Always publish twist at 50Hz
            self.cmd_vel.publish(twist)

            self._update_display(twist, keys)

            # Maintain 50Hz rate
            if self._clock is None:
                raise RuntimeError("_clock not initialized")
            self._clock.tick(50)

        pygame.quit()

    def _update_display(self, twist: Twist, keys: pygame.key.ScancodeWrapper) -> None:
        if self._screen is None or self._font is None:
            raise RuntimeError("Not initialized correctly")

        self._screen.fill((30, 30, 30))

        y_pos = 20

        # Determine active speed multiplier
        speed_mult_text = ""
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            speed_mult_text = " [BOOST 2x]"
        elif keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
            speed_mult_text = " [SLOW 0.5x]"

        active_keys = [
            pygame.key.name(key).upper()
            for key in (
                pygame.K_w,
                pygame.K_a,
                pygame.K_s,
                pygame.K_d,
                pygame.K_q,
                pygame.K_e,
                pygame.K_LSHIFT,
                pygame.K_RSHIFT,
                pygame.K_LCTRL,
                pygame.K_RCTRL,
                pygame.K_SPACE,
            )
            if keys[key]
        ]

        texts = [
            "Keyboard Teleop" + speed_mult_text + ("" if self._has_focus else " [PAUSED]"),
            "",
            f"Linear X (Forward/Back): {twist.linear.x:+.2f} m/s",
            f"Linear Y (Strafe L/R): {twist.linear.y:+.2f} m/s",
            f"Angular Z (Turn L/R): {twist.angular.z:+.2f} rad/s",
            "",
            "Keys: " + ", ".join(active_keys),
        ]

        for text in texts:
            if text:
                color = (0, 255, 255) if text.startswith("Keyboard Teleop") else (255, 255, 255)
                surf = self._font.render(text, True, color)
                self._screen.blit(surf, (20, y_pos))
            y_pos += 30

        if twist.linear.x != 0 or twist.linear.y != 0 or twist.angular.z != 0:
            pygame.draw.circle(self._screen, (255, 0, 0), (450, 30), 15)  # Red = moving
        else:
            pygame.draw.circle(self._screen, (0, 255, 0), (450, 30), 15)  # Green = stopped

        y_pos = 280
        help_texts = [
            "WS: Move | AD: Turn | QE: Strafe",
            "Shift: Boost | Ctrl: Slow",
            "Space: E-Stop | ESC: Quit",
        ]
        for text in help_texts:
            surf = self._font.render(text, True, (150, 150, 150))
            self._screen.blit(surf, (20, y_pos))
            y_pos += 25

        pygame.display.flip()


keyboard_teleop = KeyboardTeleop.blueprint

__all__ = ["KeyboardTeleop", "keyboard_teleop"]

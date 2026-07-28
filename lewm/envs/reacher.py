"""A 2-DoF planar reacher in MuJoCo — the M1 "simple task".

Top-down view of a two-link arm rotating in the horizontal plane. Actions are
joint torques in [-1, 1]^2 with an action repeat of 5 physics steps. There is
no target object in the scene: goals are *goal images* (the LeWM protocol),
so the arm pose itself is the entire task state — the simplest possible
setting in which goal-image planning is meaningful.

Design notes:
  * Headless rendering via EGL (set MUJOCO_GL=egl before importing mujoco);
    64x64 RGB, high-contrast colors (orange upper arm, yellow forearm, cyan
    fingertip on a dark floor) so a 192-dim latent has an easy life.
  * `get_state`/`set_state` expose (qpos, qvel) so evaluation can teleport
    the sim back to a start state after peeking at a future goal — the
    dataset-replay trick used by the stable-worldmodel benchmark.
  * `scripted_action` is a PD controller toward joint-space waypoints that
    are resampled every ~25 steps: covers the reachable pose space far
    better than random torques (which mostly fight the damping).
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

_XML = """
<mujoco model="reacher2d">
  <option timestep="0.01" gravity="0 0 0"/>
  <visual><headlight ambient="0.45 0.45 0.45" diffuse="0.55 0.55 0.55"/></visual>
  <worldbody>
    <geom name="floor" type="plane" size="0.35 0.35 0.1" rgba="0.10 0.11 0.16 1"/>
    <camera name="top" pos="0 0 0.29" euler="0 0 0" fovy="90"/>
    <body name="upper" pos="0 0 0.02">
      <joint name="shoulder" type="hinge" axis="0 0 1" damping="0.06"/>
      <geom type="capsule" fromto="0 0 0 0.12 0 0" size="0.018"
            rgba="1.0 0.55 0.10 1" mass="0.05"/>
      <body name="lower" pos="0.12 0 0">
        <joint name="elbow" type="hinge" axis="0 0 1" range="-2.6 2.6"
               damping="0.06"/>
        <geom type="capsule" fromto="0 0 0 0.11 0 0" size="0.015"
              rgba="1.0 0.80 0.25 1" mass="0.04"/>
        <geom name="tip" type="sphere" pos="0.11 0 0" size="0.022"
              rgba="0.15 0.85 0.95 1" mass="0.01"/>
        <site name="fingertip" pos="0.11 0 0" size="0.005"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="shoulder" gear="0.035" ctrlrange="-1 1"/>
    <motor joint="elbow"    gear="0.025" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

ACTION_REPEAT = 5


class Reacher:
    action_dim = 2

    def __init__(self, size: int = 64, seed: int | None = None):
        self.model = mujoco.MjModel.from_xml_string(_XML)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, size, size)
        self.rng = np.random.default_rng(seed)
        self._tip = self.model.site("fingertip").id
        self._waypoint = np.zeros(2)
        self._steps_to_waypoint = 0
        self.reset()

    # ------------------------------------------------------------- control --

    def reset(self) -> np.ndarray:
        self.data.qpos[:] = [self.rng.uniform(-np.pi, np.pi),
                             self.rng.uniform(-2.4, 2.4)]
        self.data.qvel[:] = self.rng.uniform(-0.5, 0.5, size=2)
        mujoco.mj_forward(self.model, self.data)
        self._steps_to_waypoint = 0
        return self.render()

    def step(self, action: np.ndarray) -> np.ndarray:
        self.data.ctrl[:] = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        for _ in range(ACTION_REPEAT):
            mujoco.mj_step(self.model, self.data)
        return self.render()

    def render(self) -> np.ndarray:
        """(H, W, 3) float32 in [0, 1]."""
        self.renderer.update_scene(self.data, camera="top")
        return self.renderer.render().astype(np.float32) / 255.0

    # --------------------------------------------------------------- state --

    @property
    def fingertip(self) -> np.ndarray:
        return self.data.site_xpos[self._tip, :2].copy()

    def get_state(self):
        return self.data.qpos.copy(), self.data.qvel.copy()

    def set_state(self, state) -> None:
        qpos, qvel = state
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    # ---------------------------------------------------------- data policy --

    def scripted_action(self) -> np.ndarray:
        """PD toward a joint-space waypoint, resampled every ~25 steps."""
        if self._steps_to_waypoint <= 0:
            self._waypoint = np.array([self.rng.uniform(-np.pi, np.pi),
                                       self.rng.uniform(-2.4, 2.4)])
            self._steps_to_waypoint = self.rng.integers(15, 35)
        self._steps_to_waypoint -= 1
        err = self._waypoint - self.data.qpos
        err[0] = (err[0] + np.pi) % (2 * np.pi) - np.pi  # shoulder wraps
        a = 2.0 * err - 0.4 * self.data.qvel + self.rng.normal(0, 0.15, size=2)
        return np.clip(a, -1, 1).astype(np.float32)

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
# Force the NVIDIA EGL vendor. Without this, GLVND probes Mesa first, which
# can't open /dev/dri (permission) and silently falls back to llvmpipe
# SOFTWARE rendering — measured 3.3x slower and not the GPU. Found the
# hard way; Reacher.__init__ verifies the renderer and refuses regressions.
_NV_ICD = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if os.path.exists(_NV_ICD):
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", _NV_ICD)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

_XML = """
<mujoco model="reacher2d">
  <option timestep="0.01" gravity="0 0 0"/>
  <visual><headlight ambient="0.45 0.45 0.45" diffuse="0.55 0.55 0.55"/></visual>
  <worldbody>
    <geom name="floor" type="plane" size="0.35 0.35 0.1" rgba="0.10 0.11 0.16 1"/>
    <camera name="top" pos="0 0 0.29" euler="0 0 0" fovy="90"/>
    <!-- contype/conaffinity 0: a planar arm needs no contacts, and the tip
         sphere otherwise penetrates the floor — the resulting friction
         stalls the arm entirely (cost a full pipeline run to find). -->
    <body name="upper" pos="0 0 0.03">
      <joint name="shoulder" type="hinge" axis="0 0 1" damping="0.09"/>
      <geom type="capsule" fromto="0 0 0 0.12 0 0" size="0.018"
            rgba="1.0 0.55 0.10 1" mass="0.05" contype="0" conaffinity="0"/>
      <body name="lower" pos="0.12 0 0">
        <joint name="elbow" type="hinge" axis="0 0 1" range="-2.6 2.6"
               damping="0.09"/>
        <geom type="capsule" fromto="0 0 0 0.11 0 0" size="0.015"
              rgba="1.0 0.80 0.25 1" mass="0.04" contype="0" conaffinity="0"/>
        <geom name="tip" type="sphere" pos="0.11 0 0" size="0.022"
              rgba="0.15 0.85 0.95 1" mass="0.01" contype="0" conaffinity="0"/>
        <site name="fingertip" pos="0.11 0 0" size="0.005"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="shoulder" gear="0.09" ctrlrange="-1 1"/>
    <motor joint="elbow"    gear="0.06" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

ACTION_REPEAT = 5


class Reacher:
    action_dim = 2

    def __init__(self, size: int = 64, seed: int | None = None,
                 demo_size: int = 256):
        self.model = mujoco.MjModel.from_xml_string(_XML)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, size, size)
        self._demo_renderer = None      # lazily built; only eval/demos need it
        self._demo_size = demo_size
        self.rng = np.random.default_rng(seed)
        self._tip = self.model.site("fingertip").id
        self._waypoint = np.zeros(2)
        self._steps_to_waypoint = 0
        self._verify_gpu_rendering()
        self.reset()

    def _verify_gpu_rendering(self) -> None:
        """One-time check that EGL is on the GPU, not llvmpipe software."""
        self.renderer.update_scene(self.data)
        self.renderer.render()
        try:
            from OpenGL import GL
            gl_renderer = GL.glGetString(GL.GL_RENDERER).decode()
        except Exception:
            return
        if not hasattr(Reacher, "_gl_reported"):
            Reacher._gl_reported = True
            print(f"[reacher] OpenGL renderer: {gl_renderer}")
            if "llvmpipe" in gl_renderer.lower():
                print("[reacher] WARNING: SOFTWARE rendering — GPU EGL not "
                      "active. Check __EGL_VENDOR_LIBRARY_FILENAMES.")

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

    def render_demo(self) -> np.ndarray:
        """High-res render of the CURRENT state for demo videos. (H, W, 3) u8."""
        if self._demo_renderer is None:
            self._demo_renderer = mujoco.Renderer(
                self.model, self._demo_size, self._demo_size)
        self._demo_renderer.update_scene(self.data, camera="top")
        return self._demo_renderer.render().copy()

    def render_pose_demo(self, qpos: np.ndarray) -> np.ndarray:
        """High-res render of an ARBITRARY pose, restoring state afterward.
        Used to visualize imagined latents via nearest-neighbor retrieval:
        LeWM has no decoder, so imagination panels re-render the dataset
        state whose embedding is closest to each imagined latent."""
        snap = self.get_state()
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        img = self.render_demo()
        self.set_state(snap)
        return img

    def sample_goal(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a goal POSE a guaranteed-nontrivial distance from the
        current pose and return (goal_qpos, goal_obs_64, goal_fingertip).

        Goals live in pose space, not at the end of a short scripted rollout
        — short rollouts barely move the arm, which once made the whole eval
        vacuous (goals spawned inside the success radius; every policy,
        including no-op, scored 100%). Found via the zero-action baseline;
        the baselines stay in eval.py as permanent guards.
        """
        dq_s = self.rng.uniform(0.7, 1.6) * self.rng.choice([-1.0, 1.0])
        goal = np.array([
            self.data.qpos[0] + dq_s,
            np.clip(self.data.qpos[1] + self.rng.uniform(-1.2, 1.2), -2.4, 2.4),
        ])
        snap = self.get_state()
        self.data.qpos[:] = goal
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        img = self.render()
        tip = self.fingertip
        self.set_state(snap)
        return goal, img, tip

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

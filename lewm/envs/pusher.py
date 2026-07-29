"""Planar pushing in MuJoCo — the first contact-rich task (roadmap M2).

The same 2-DoF arm as the reacher, plus a free cyan puck on the floor. The
task: push the puck to the position shown in a goal image. This is where
"does one 192-dim token survive contact dynamics?" starts getting answered —
contact is discontinuous, and errors in *where the tip meets the puck*
matter far more than errors in gross arm pose.

Differences from the reacher, all deliberate:
  * Arm geoms keep contacts ON (the tip must actually hit the puck); the
    floor stays contact-free for the ARM (raised z) but ON for the puck.
  * The puck slides with damping (pushed things stop — like PushT, unlike
    an air-hockey world; makes planning consequences persistent).
  * Success is measured on the PUCK, not the fingertip: the arm is a means.
  * Goal images show the puck at its goal with the arm at its current pose
    (the arm's pose in the goal is a nuisance variable the planner must
    learn to ignore — same convention as DINO-WM's PushT).
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
_NV_ICD = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if os.path.exists(_NV_ICD):
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", _NV_ICD)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

_XML = """
<mujoco model="pusher2d">
  <compiler angle="radian"/>
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <visual><headlight ambient="0.45 0.45 0.45" diffuse="0.55 0.55 0.55"/></visual>
  <worldbody>
    <geom name="floor" type="plane" size="0.35 0.35 0.1" rgba="0.10 0.11 0.16 1"
          friction="0.35 0.005 0.0001"/>
    <camera name="top" pos="0 0 0.30" euler="0 0 0" fovy="90"/>
    <body name="upper" pos="0 0 0.024">
      <joint name="shoulder" type="hinge" axis="0 0 1" damping="0.09"/>
      <geom type="capsule" fromto="0 0 0 0.12 0 0" size="0.016"
            rgba="1.0 0.55 0.10 1" mass="0.05" contype="0" conaffinity="0"/>
      <body name="lower" pos="0.12 0 0">
        <joint name="elbow" type="hinge" axis="0 0 1" range="-2.6 2.6"
               damping="0.09"/>
        <geom type="capsule" fromto="0 0 0 0.11 0 0" size="0.014"
              rgba="1.0 0.80 0.25 1" mass="0.04" contype="0" conaffinity="0"/>
        <geom name="tip" type="sphere" pos="0.11 0 0" size="0.02"
              rgba="1.0 0.80 0.25 1" mass="0.01" contype="1" conaffinity="1"/>
        <site name="fingertip" pos="0.11 0 0" size="0.005"/>
      </body>
    </body>
    <body name="puck" pos="0.14 0 0.012">
      <joint name="puck_x" type="slide" axis="1 0 0" damping="1.2"/>
      <joint name="puck_y" type="slide" axis="0 1 0" damping="1.2"/>
      <geom name="puck_geom" type="cylinder" size="0.028 0.012"
            rgba="0.15 0.85 0.95 1" mass="0.08"
            friction="0.05 0.005 0.0001"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="shoulder" gear="0.09" ctrlrange="-1 1"/>
    <motor joint="elbow"    gear="0.06" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

ACTION_REPEAT = 5


class Pusher:
    action_dim = 2
    EVAL_BUDGET = 80          # contact manipulation needs more steps
    SUCCESS_DIST = 0.05       # measured on the puck

    def __init__(self, size: int = 64, seed: int | None = None,
                 demo_size: int = 256):
        self.model = mujoco.MjModel.from_xml_string(_XML)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, size, size)
        self._demo_renderer = None
        self._demo_size = demo_size
        self.rng = np.random.default_rng(seed)
        self._tip = self.model.site("fingertip").id
        self._waypoint = np.zeros(2)
        self._steps_to_waypoint = 0
        self._mode = "seek"
        self._verify_gpu_rendering()
        self.reset()

    def _verify_gpu_rendering(self) -> None:
        self.renderer.update_scene(self.data)
        self.renderer.render()
        try:
            from OpenGL import GL
            gl_renderer = GL.glGetString(GL.GL_RENDERER).decode()
        except Exception:
            return
        if not hasattr(Pusher, "_gl_reported"):
            Pusher._gl_reported = True
            print(f"[pusher] OpenGL renderer: {gl_renderer}")
            if "llvmpipe" in gl_renderer.lower():
                print("[pusher] WARNING: SOFTWARE rendering — GPU EGL inactive.")

    # ------------------------------------------------------------- control --

    def reset(self) -> np.ndarray:
        self.data.qpos[:2] = [self.rng.uniform(-np.pi, np.pi),
                              self.rng.uniform(-2.0, 2.0)]
        self.data.qvel[:] = 0
        # Puck in an annulus the arm can reach (reach 0.23, puck r 0.028).
        r = self.rng.uniform(0.09, 0.19)
        th = self.rng.uniform(-np.pi, np.pi)
        self.data.qpos[2] = r * np.cos(th) - 0.14
        self.data.qpos[3] = r * np.sin(th)
        mujoco.mj_forward(self.model, self.data)
        self._steps_to_waypoint = 0
        self._mode = "seek"
        return self.render()

    def step(self, action: np.ndarray) -> np.ndarray:
        self.data.ctrl[:] = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        for _ in range(ACTION_REPEAT):
            mujoco.mj_step(self.model, self.data)
        return self.render()

    def render(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera="top")
        return self.renderer.render().astype(np.float32) / 255.0

    def render_demo(self) -> np.ndarray:
        if self._demo_renderer is None:
            self._demo_renderer = mujoco.Renderer(
                self.model, self._demo_size, self._demo_size)
        self._demo_renderer.update_scene(self.data, camera="top")
        return self._demo_renderer.render().copy()

    def render_pose_demo(self, qpos: np.ndarray) -> np.ndarray:
        snap = self.get_state()
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        img = self.render_demo()
        self.set_state(snap)
        return img

    # --------------------------------------------------------------- state --

    @property
    def fingertip(self) -> np.ndarray:
        return self.data.site_xpos[self._tip, :2].copy()

    @property
    def puck(self) -> np.ndarray:
        return self.data.qpos[2:4] + np.array([0.14, 0.0])

    # what success is measured on (eval reads this generically)
    @property
    def target_point(self) -> np.ndarray:
        return self.puck

    def get_state(self):
        return self.data.qpos.copy(), self.data.qvel.copy()

    def set_state(self, state) -> None:
        qpos, qvel = state
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    def sample_goal(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Goal = an ACTUAL future state reached by the scripted push policy
        once the puck has displaced >= 7 cm. Returns
        (goal_qpos, goal_obs_64, goal_puck_position).

        Why not 'teleport the puck, keep the arm'? Because that goal image
        is adversarial: the cost then rewards keeping the arm at its
        current pose, but pushing requires moving the arm away from it —
        staying still scores better than acting. Both models cratered to
        0-4% under that design before this was understood. A rolled-out
        goal has a consistent arm pose (the arm that just pushed) and is
        reachable by construction."""
        snap = self.get_state()
        p0 = self.puck.copy()
        goal_qpos = self.data.qpos.copy()
        for _ in range(240):
            self.step(self.scripted_action())
            if np.linalg.norm(self.puck - p0) >= 0.07:
                goal_qpos = self.data.qpos.copy()
                break
        else:
            goal_qpos = self.data.qpos.copy()   # best displacement achieved
        self.data.qpos[:] = goal_qpos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        img = self.render()
        goal_puck = self.puck.copy()
        self.set_state(snap)
        return goal_qpos, img, goal_puck

    # ---------------------------------------------------------- data policy --

    def _ik(self, target: np.ndarray) -> np.ndarray | None:
        """Analytic 2-link IK (l1=0.12, l2=0.11); nearest elbow branch."""
        l1, l2 = 0.12, 0.11
        r = float(np.linalg.norm(target))
        r = np.clip(r, abs(l1 - l2) + 1e-3, l1 + l2 - 1e-3)
        ce = (r * r - l1 * l1 - l2 * l2) / (2 * l1 * l2)
        best, bd = None, 9e9
        for sign in (1.0, -1.0):
            e = sign * np.arccos(np.clip(ce, -1, 1))
            if not (-2.4 <= e <= 2.4):
                continue
            s = np.arctan2(target[1], target[0]) - np.arctan2(
                l2 * np.sin(e), l1 + l2 * np.cos(e))
            dq = abs((s - self.data.qpos[0] + np.pi) % (2 * np.pi) - np.pi)
            if dq < bd:
                bd, best = dq, np.array([s, e])
        return best

    def _sample_push(self) -> None:
        """Pick a push direction through the puck that stays in the arm's
        reachable annulus: a pre-contact point on one side and a
        push-through point on the other."""
        for _ in range(20):
            phi = self.rng.uniform(-np.pi, np.pi)
            u = np.array([np.cos(phi), np.sin(phi)])
            pre = self.puck - 0.075 * u
            post = self.puck + self.rng.uniform(0.03, 0.08) * u
            if 0.04 <= np.linalg.norm(pre) <= 0.22 and np.linalg.norm(post) <= 0.22:
                self._pre, self._post = pre, post
                return
        self._pre = self._post = None

    def scripted_action(self) -> np.ndarray:
        """Push-biased exploration. A push has two phases: APPROACH a
        pre-contact point beside the puck, then DRIVE straight through it.
        (A single IK target past the puck fails: joint-space PD paths curve,
        and the tip orbits around the puck to the far side without touching
        it — cost a zero-contact dataset to learn.) 30% of segments wander
        instead, for pose diversity away from the puck."""
        if self._steps_to_waypoint <= 0:
            if self.rng.random() < 0.7:
                self._mode = "approach"
                self._sample_push()
                self._approach_left = 18      # approach timeout -> push anyway
                self._steps_to_waypoint = self.rng.integers(30, 45)
            else:
                self._mode = "wander"
                self._steps_to_waypoint = self.rng.integers(10, 20)
                self._wander_q = np.array([self.rng.uniform(-np.pi, np.pi),
                                           self.rng.uniform(-2.0, 2.0)])
            if self._mode == "approach" and self._pre is None:
                self._mode = "wander"
                self._wander_q = np.array([self.rng.uniform(-np.pi, np.pi),
                                           self.rng.uniform(-2.0, 2.0)])
        self._steps_to_waypoint -= 1

        if self._mode == "approach":
            self._approach_left -= 1
            # Loose tolerance + timeout: the joint PD has steady-state error
            # larger than a tight tolerance, and an approach that never
            # "arrives" must still graduate to the push-through phase.
            if (np.linalg.norm(self.fingertip - self._pre) < 0.05
                    or self._approach_left <= 0):
                self._mode = "push"

        if self._mode == "push":
            # Straight-line tip drive via the Jacobian. IK-to-endpoint fails
            # here: pre/post IK can pick different elbow branches and the
            # joint-space path swings the tip AROUND the puck. For a short
            # local push, task-space velocity control is exact.
            jac = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jac, None, self._tip)
            u = self._post - self.fingertip
            u = u / (np.linalg.norm(u) + 1e-8)
            a = 30.0 * (jac[:2, :2].T @ u) - 0.4 * self.data.qvel[:2]
            a = a + self.rng.normal(0, 0.05, size=2)
            return np.clip(a, -1, 1).astype(np.float32)

        goal_q = self._ik(self._pre) if self._mode == "approach" else None
        if goal_q is None:
            goal_q = getattr(self, "_wander_q", self.data.qpos[:2].copy())
        err = goal_q - self.data.qpos[:2]
        err[0] = (err[0] + np.pi) % (2 * np.pi) - np.pi
        a = 2.5 * err - 0.6 * self.data.qvel[:2] + self.rng.normal(0, 0.15, size=2)
        return np.clip(a, -1, 1).astype(np.float32)

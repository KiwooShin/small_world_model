"""PushWorld: a tiny 2D physics environment, exact and dependency-free.

A circular *pusher* (controlled by a 2D velocity action) and a passive *ball*
live in a square arena. The ball bounces elastically off walls and gets pushed
on contact with the pusher. Observations are rendered top-down as HxW RGB
float32 images in [0, 1].

Why hand-rolled instead of gym/MuJoCo: the physics here are exact and known,
so any prediction error in the labs is attributable to the *model*, never to
simulator noise. Every lab in the ladder reuses this env so results compare.
"""

from __future__ import annotations

import numpy as np

# Arena is the unit square [0,1]^2; rendering maps it to `size` pixels.
PUSHER_R = 0.08
BALL_R = 0.06
BALL_DAMPING = 0.985  # per-step velocity decay so the ball settles
MAX_ACTION = 0.05     # max pusher displacement per step (fraction of arena)


class PushWorld:
    def __init__(self, size: int = 64, seed: int | None = None):
        self.size = size
        self.rng = np.random.default_rng(seed)
        # Precompute the pixel-center coordinate grid once; rendering is then
        # two vectorized distance tests per frame.
        xs = (np.arange(size) + 0.5) / size
        self._gx, self._gy = np.meshgrid(xs, xs)  # gy is row -> world y
        self.reset()

    def reset(self) -> np.ndarray:
        m = 0.15  # keep initial positions off the walls
        self.pusher = self.rng.uniform(m, 1 - m, size=2)
        self.ball = self.rng.uniform(m, 1 - m, size=2)
        # Re-sample until the two bodies don't start in contact.
        while np.linalg.norm(self.ball - self.pusher) < PUSHER_R + BALL_R + 0.02:
            self.ball = self.rng.uniform(m, 1 - m, size=2)
        self.ball_vel = self.rng.uniform(-0.02, 0.02, size=2)
        return self.render()

    def step(self, action: np.ndarray) -> np.ndarray:
        """action: (2,) in [-1, 1], scaled to pusher displacement."""
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1) * MAX_ACTION
        self.pusher = np.clip(self.pusher + action, PUSHER_R, 1 - PUSHER_R)

        # Pusher->ball contact: resolve overlap along the contact normal and
        # transfer the pusher's motion into ball velocity.
        delta = self.ball - self.pusher
        dist = np.linalg.norm(delta)
        min_dist = PUSHER_R + BALL_R
        if dist < min_dist:
            normal = delta / (dist + 1e-8)
            self.ball = self.pusher + normal * min_dist
            # Impulse: push speed along the normal, plus a bit of carry.
            self.ball_vel += normal * max(np.dot(action, normal), 0.0) * 1.5

        self.ball = self.ball + self.ball_vel
        # Elastic wall bounces for the ball.
        for axis in range(2):
            lo, hi = BALL_R, 1 - BALL_R
            if self.ball[axis] < lo:
                self.ball[axis] = 2 * lo - self.ball[axis]
                self.ball_vel[axis] *= -1
            elif self.ball[axis] > hi:
                self.ball[axis] = 2 * hi - self.ball[axis]
                self.ball_vel[axis] *= -1
        self.ball_vel *= BALL_DAMPING
        return self.render()

    def render(self) -> np.ndarray:
        """(H, W, 3) float32 in [0,1]. Pusher orange, ball cyan, dark bg."""
        img = np.full((self.size, self.size, 3), 0.06, dtype=np.float32)
        # Soft-edged discs: 1px anti-aliasing band via clip of signed distance.
        aa = 1.5 / self.size
        for center, radius, color in (
            (self.pusher, PUSHER_R, (1.0, 0.55, 0.1)),
            (self.ball, BALL_R, (0.15, 0.85, 0.95)),
        ):
            d = np.sqrt((self._gx - center[0]) ** 2 + (self._gy - center[1]) ** 2)
            # float32 cast: positions are float64, and letting the blend promote
            # would leak float64 frames into rollout inputs.
            alpha = np.clip((radius - d) / aa, 0.0, 1.0)[..., None].astype(np.float32)
            img = img * (1 - alpha) + alpha * np.asarray(color, dtype=np.float32)
        return img

    def scripted_action(self) -> np.ndarray:
        """A weakly ball-seeking random policy: mostly random exploration with
        a bias toward the ball, so collected data actually contains contact
        events (pure random walks rarely touch the ball)."""
        to_ball = self.ball - self.pusher
        to_ball /= np.linalg.norm(to_ball) + 1e-8
        noise = self.rng.normal(0, 0.8, size=2)
        a = 0.6 * to_ball + noise
        return np.clip(a, -1, 1)

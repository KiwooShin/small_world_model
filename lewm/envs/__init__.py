"""Env registry. Each env exposes: action_dim, EVAL_BUDGET, SUCCESS_DIST,
reset/step/render, render_demo/render_pose_demo, get_state/set_state,
sample_goal() -> (goal_qpos, goal_obs, goal_point), target_point (what
success is measured on), and scripted_action() for data collection."""


def make(name: str, **kw):
    if name == "reacher":
        from .reacher import Reacher
        return Reacher(**kw)
    if name == "pusher":
        from .pusher import Pusher
        return Pusher(**kw)
    raise KeyError(f"unknown env '{name}' (have: reacher, pusher)")

"""Action skills as per-world Warp control kernels.

Each skill is one `_ctrl_<kind>` kernel launched before every `mjw.step` (both inside a
captured CUDA graph): it reads the previous step's state, updates the per-world sticky
phase latches, writes the ctrl rows it owns (rows it doesn't write persist from the
previous action), accumulates +1 cost per not-yet-successful step, and maintains `succ`
(the action's current success flag). `plan.py` orchestrates action boundaries: `_init_action`
resets the latches, `_prep_*` build IK warm starts/goals from the sim state, `_finish_action`
folds `succ` into the per-world completed-action count. The skills:

- move:  base (x,y) goal via the nav descent field; success when base within BASE_SUCCESS.
- pick:  phased top-down pick (pre-grasp above -> descend -> settle -> close -> lift);
         success = gripper stayed closed-on-object for GRIP_HOLD steps.
- place: drive the held object straight to a pose and open the gripper there.
- push:  drive the closed fist to a pose, slowing near it (a full-speed slam topples the
         target violently; a gentle contact tips it in a controlled arc).
"""

from __future__ import annotations

import numpy as np
import warp as wp

from tamp_ce_mjwarp.nav import descent_dir
from tamp_ce_mjwarp.sim import _sec2steps

MAX_VEL = float(np.deg2rad(45.0))  # arm joint speed cap — slower = gentler approach
BASE_LEAD = 0.06        # base target lead distance (~1 m/s given the actuator kv)
BASE_SLEW = 6.0         # max rate of change of the base velocity direction (1/s) — eases
#                         into turns / from rest instead of snapping (jolts a carried object)
BASE_SUCCESS = 0.10     # base goal tolerance (m)
BASE_TURN = 2.0         # max base yaw rate (rad/s) when turning to face a grasp
BASE_IK_VEL = 0.4       # max base translation speed (m/s) when full-body IK drives the base
GRASP_REACH = 0.04      # EE-to-goal distance that triggers the gripper switch
PRE_TOL = 0.05          # pick: distance to the pre-grasp point that triggers the descent
GRIP_WAIT = _sec2steps(0.30)   # settle at the grasp pose BEFORE closing
LIFT_WAIT = _sec2steps(0.12)   # keep holding at the grasp AFTER securing, BEFORE lifting
GRIP_HOLD = _sec2steps(0.09)   # gripper closed-on-object time that counts as held
GRIP_LO, GRIP_HI = 0.06, 0.45  # gripper-joint range that means "object held"
LIFT_H = 0.10           # pick: pre-grasp/lift height above the grasp pose (phased approach
#                         avoids the arc-in that knocks the object off-centre)
PUSH_VEL = float(np.deg2rad(12.0))  # push: slow arm speed near the pose
PUSH_SLOW = 0.15        # push: EE-to-pose distance under which the arm slows to PUSH_VEL
CLOSED, OPEN = 0.0, 0.5  # gripper ctrl
PI = float(np.pi)

# Early-exit settle tails (s): once ALL worlds latch success, run this much longer before
# ending the action — covering post-success motion the fixed timeout used to absorb
# (pick: LIFT_WAIT + the lift travel; place: release + fall; push: topple + block flight;
# move: braking). Without the pick tail the next action would start with the arm still
# down and the cube dragged along the table.
EXIT_TAIL = {"move": 0.3, "pick": 1.0, "place": 1.0, "push": 2.0}


def quat_to_mat(q):
    """Rotation matrices (.., 3, 3) from quaternions (.., 4) in (x, y, z, w)."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    rows = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    return np.stack([np.stack(r, axis=-1) for r in rows], axis=-2)


@wp.func
def _advance_base(cmd: wp.vec3, target: wp.vec3, vel: float, dt: float) -> wp.vec3:
    """Step the commanded base pose [x, y, th] toward `target`, capping translation at
    `vel` (m/s; 0 holds xy) and rotation at BASE_TURN. The stiff position actuator then
    tracks this leading command each step."""
    dx, dy = target[0] - cmd[0], target[1] - cmd[1]
    s = wp.min(1.0, vel * dt / (wp.sqrt(dx * dx + dy * dy) + 1e-9))
    dth = target[2] - cmd[2]
    dth -= 2.0 * PI * wp.floor((dth + PI) / (2.0 * PI))
    th = cmd[2] + wp.clamp(dth, -BASE_TURN * dt, BASE_TURN * dt)
    return wp.vec3(cmd[0] + dx * s, cmd[1] + dy * s, th)


@wp.func
def _rate_limited_arm(qc: wp.array2d(dtype=wp.float32), ctrl: wp.array2d(dtype=wp.float32),
                      arm_act: wp.array(dtype=wp.int32), q_goal: wp.array2d(dtype=wp.float32),
                      i: int, vlim: float, dt: float):
    """Advance the commanded arm config `qc` toward `q_goal` at `vlim` and write arm ctrl."""
    for j in range(6):
        v = qc[i, j] + wp.clamp(q_goal[i, j] - qc[i, j], -vlim * dt, vlim * dt)
        qc[i, j] = v
        ctrl[i, arm_act[j]] = v


@wp.func
def _write_base(ctrl: wp.array2d(dtype=wp.float32), base_act: wp.array(dtype=wp.int32),
                i: int, cmd: wp.vec3):
    ctrl[i, base_act[0]] = cmd[0]
    ctrl[i, base_act[1]] = cmd[1]
    ctrl[i, base_act[2]] = cmd[2]


@wp.kernel
def _ctrl_move(
    qpos: wp.array2d(dtype=wp.float32), ctrl: wp.array2d(dtype=wp.float32),
    base_qadr: wp.array(dtype=wp.int32), base_act: wp.array(dtype=wp.int32),
    goal_xy: wp.array(dtype=wp.vec2), gxy: wp.array3d(dtype=wp.vec2),
    use_nav: int, xmin: float, ymin: float, res: float, dt: float,
    vdir: wp.array(dtype=wp.vec2), done: wp.array(dtype=wp.int32),
    succ: wp.array(dtype=wp.int32), cost: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    pos = wp.vec2(qpos[i, base_qadr[0]], qpos[i, base_qadr[1]])
    th = qpos[i, base_qadr[2]]
    straight = goal_xy[i] - pos
    if wp.length(straight) < BASE_SUCCESS:
        done[i] = 1
    # far from the goal follow the VI descent field (routes around obstacles); near it a
    # straight line (the field resolution is coarser than the goal tolerance)
    raw = straight
    if use_nav != 0 and wp.length(straight) >= 2.5 * res:
        raw = descent_dir(gxy, i, pos, xmin, ymin, res)
    rmag = wp.length(raw)
    u = raw / (rmag + 1e-9)
    # rotate vdir toward u by at most BASE_SLEW*dt: the base's acceleration limit/soft start
    dv = u - vdir[i]
    v = vdir[i] + dv * wp.min(1.0, BASE_SLEW * dt / (wp.length(dv) + 1e-9))
    vdir[i] = v
    vel = v / (wp.length(v) + 1e-9) * rmag  # smoothed direction, raw magnitude
    n = wp.length(vel)
    tgt = pos + vel / (n + 1e-9) * wp.min(BASE_LEAD, n)
    if done[i] != 0:
        tgt = pos  # freeze once reached
    _write_base(ctrl, base_act, i, wp.vec3(tgt[0], tgt[1], th))  # hold heading
    succ[i] = done[i]
    if done[i] == 0:
        cost[i] += 1.0


@wp.kernel
def _ctrl_pick(
    qpos: wp.array2d(dtype=wp.float32), ctrl: wp.array2d(dtype=wp.float32),
    site_xpos: wp.array2d(dtype=wp.vec3),
    arm_act: wp.array(dtype=wp.int32), base_act: wp.array(dtype=wp.int32),
    gripper_act: int, pinch_site: int, grip_qadr: int, dt: float,
    q_a: wp.array2d(dtype=wp.float32), q_b: wp.array2d(dtype=wp.float32),
    goal_world: wp.array(dtype=wp.vec3), base_target: wp.array(dtype=wp.vec3),
    base_vel: wp.array(dtype=wp.float32),
    qc: wp.array2d(dtype=wp.float32), base_cmd: wp.array(dtype=wp.vec3),
    at_pre: wp.array(dtype=wp.int32), reached: wp.array(dtype=wp.int32),
    reach_t: wp.array(dtype=wp.int32), hold: wp.array(dtype=wp.int32),
    succ: wp.array(dtype=wp.int32), cost: wp.array(dtype=wp.float32),
):
    """Phased top-down pick: approach the pre-grasp point LIFT_H above the goal (arm target
    q_b), descend to the grasp (q_a), settle, close, hold, lift back to q_b."""
    i = wp.tid()
    ee = site_xpos[i, pinch_site]
    bc = _advance_base(base_cmd[i], base_target[i], base_vel[0], dt)
    base_cmd[i] = bc
    gw = goal_world[i]
    if wp.length(ee - (gw + wp.vec3(0.0, 0.0, LIFT_H))) < PRE_TOL:
        at_pre[i] = 1
    if at_pre[i] != 0 and wp.length(ee - gw) < GRASP_REACH:
        reached[i] = 1
    reach_t[i] = wp.where(reached[i] != 0, reach_t[i] + 1, 0)
    settled = reached[i] != 0 and reach_t[i] >= GRIP_WAIT  # wait at the pose before clamping
    lifting = hold[i] >= GRIP_HOLD + LIFT_WAIT

    # arm target: pre-grasp (up) -> descend to grasp -> (hold) -> lift (up)
    up = lifting or at_pre[i] == 0
    for j in range(6):
        qg = wp.where(up, q_b[i, j], q_a[i, j])
        v = qc[i, j] + wp.clamp(qg - qc[i, j], -MAX_VEL * dt, MAX_VEL * dt)
        qc[i, j] = v
        ctrl[i, arm_act[j]] = v
    ctrl[i, gripper_act] = wp.where(settled, CLOSED, OPEN)
    _write_base(ctrl, base_act, i, bc)

    gj = qpos[i, grip_qadr]  # gripper joint (read pre-step: one-step lag)
    hold[i] = wp.where(settled and gj > GRIP_LO and gj < GRIP_HI, hold[i] + 1, 0)
    held = hold[i] >= GRIP_HOLD
    succ[i] = wp.where(held, 1, 0)
    if not held:
        cost[i] += 1.0


@wp.kernel
def _ctrl_place(
    qpos: wp.array2d(dtype=wp.float32), ctrl: wp.array2d(dtype=wp.float32),
    site_xpos: wp.array2d(dtype=wp.vec3),
    arm_act: wp.array(dtype=wp.int32), base_act: wp.array(dtype=wp.int32),
    gripper_act: int, pinch_site: int, dt: float,
    q_a: wp.array2d(dtype=wp.float32), goal_world: wp.array(dtype=wp.vec3),
    base_target: wp.array(dtype=wp.vec3), base_vel: wp.array(dtype=wp.float32),
    qc: wp.array2d(dtype=wp.float32), base_cmd: wp.array(dtype=wp.vec3),
    reached: wp.array(dtype=wp.int32),
    succ: wp.array(dtype=wp.int32), cost: wp.array(dtype=wp.float32),
):
    """Drive the EE (holding the object) straight to the pose; open the gripper there."""
    i = wp.tid()
    ee = site_xpos[i, pinch_site]
    bc = _advance_base(base_cmd[i], base_target[i], base_vel[0], dt)
    base_cmd[i] = bc
    if wp.length(ee - goal_world[i]) < GRASP_REACH:
        reached[i] = 1
    _rate_limited_arm(qc, ctrl, arm_act, q_a, i, MAX_VEL, dt)
    ctrl[i, gripper_act] = wp.where(reached[i] != 0, OPEN, CLOSED)
    _write_base(ctrl, base_act, i, bc)
    succ[i] = reached[i]
    if reached[i] == 0:
        cost[i] += 1.0


@wp.kernel
def _ctrl_push(
    qpos: wp.array2d(dtype=wp.float32), ctrl: wp.array2d(dtype=wp.float32),
    site_xpos: wp.array2d(dtype=wp.vec3),
    arm_act: wp.array(dtype=wp.int32), base_act: wp.array(dtype=wp.int32),
    gripper_act: int, pinch_site: int, dt: float,
    q_a: wp.array2d(dtype=wp.float32), goal_world: wp.array(dtype=wp.vec3),
    base_target: wp.array(dtype=wp.vec3), base_vel: wp.array(dtype=wp.float32),
    qc: wp.array2d(dtype=wp.float32), base_cmd: wp.array(dtype=wp.vec3),
    reached: wp.array(dtype=wp.int32),
    succ: wp.array(dtype=wp.int32), cost: wp.array(dtype=wp.float32),
):
    """Drive the EE (closed fist) to the pose, dropping to PUSH_VEL within PUSH_SLOW."""
    i = wp.tid()
    ee = site_xpos[i, pinch_site]
    bc = _advance_base(base_cmd[i], base_target[i], base_vel[0], dt)
    base_cmd[i] = bc
    dist = wp.length(ee - goal_world[i])
    if dist < GRASP_REACH:
        reached[i] = 1
    vlim = wp.where(dist < PUSH_SLOW, PUSH_VEL, MAX_VEL)  # gentle near the pose
    _rate_limited_arm(qc, ctrl, arm_act, q_a, i, vlim, dt)
    ctrl[i, gripper_act] = CLOSED
    _write_base(ctrl, base_act, i, bc)
    succ[i] = reached[i]
    if reached[i] == 0:
        cost[i] += 1.0


@wp.kernel
def _init_action(
    qpos: wp.array2d(dtype=wp.float32),
    arm_qadr: wp.array(dtype=wp.int32), base_qadr: wp.array(dtype=wp.int32),
    qc: wp.array2d(dtype=wp.float32), base_cmd: wp.array(dtype=wp.vec3),
    vdir: wp.array(dtype=wp.vec2), at_pre: wp.array(dtype=wp.int32),
    reached: wp.array(dtype=wp.int32), reach_t: wp.array(dtype=wp.int32),
    hold: wp.array(dtype=wp.int32), done: wp.array(dtype=wp.int32),
    succ: wp.array(dtype=wp.int32),
):
    """Reset per-action carries/latches from the current state (action boundary)."""
    i = wp.tid()
    for j in range(6):
        qc[i, j] = qpos[i, arm_qadr[j]]
    base_cmd[i] = wp.vec3(qpos[i, base_qadr[0]], qpos[i, base_qadr[1]], qpos[i, base_qadr[2]])
    vdir[i] = wp.vec2(0.0, 0.0)
    at_pre[i] = 0
    reached[i] = 0
    reach_t[i] = 0
    hold[i] = 0
    done[i] = 0
    succ[i] = 0


@wp.kernel
def _finish_action(succ: wp.array(dtype=wp.int32), n_act: wp.array(dtype=wp.float32)):
    i = wp.tid()
    n_act[i] += wp.where(succ[i] != 0, 1.0, 0.0)


@wp.kernel
def _prep_qinit(
    qpos: wp.array2d(dtype=wp.float32), base_qadr: wp.array(dtype=wp.int32),
    home_arm: wp.array(dtype=wp.float32), q: wp.array2d(dtype=wp.float32),
):
    """IK warm start [current base pose, home arm]."""
    i = wp.tid()
    for j in range(3):
        q[i, j] = qpos[i, base_qadr[j]]
    for j in range(6):
        q[i, 3 + j] = home_arm[j]


@wp.kernel
def _prep_pick_goal(
    qpos: wp.array2d(dtype=wp.float32), obj_qadr: int, z_off: float,
    goal_world: wp.array(dtype=wp.vec3),
):
    """Grasp point = object center + z_off (read from the live state)."""
    i = wp.tid()
    goal_world[i] = wp.vec3(qpos[i, obj_qadr], qpos[i, obj_qadr + 1],
                            qpos[i, obj_qadr + 2] + z_off)


@wp.kernel
def _set_ik_target(goal_world: wp.array(dtype=wp.vec3), z_off: float,
                   target_pos: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    target_pos[i] = goal_world[i] + wp.vec3(0.0, 0.0, z_off)


@wp.kernel
def _store_q(q: wp.array2d(dtype=wp.float32), q_dst: wp.array2d(dtype=wp.float32),
             base_target: wp.array(dtype=wp.vec3), store_base: int):
    """Split an IK solution: arm part -> q_dst, base part -> base_target (if store_base)."""
    i = wp.tid()
    for j in range(6):
        q_dst[i, j] = q[i, 3 + j]
    if store_base != 0:
        base_target[i] = wp.vec3(q[i, 0], q[i, 1], q[i, 2])


@wp.kernel
def _record_probe(
    qpos: wp.array2d(dtype=wp.float32), site_xpos: wp.array2d(dtype=wp.vec3),
    base_qadr: wp.array(dtype=wp.int32), pinch_site: int, obj_qadr: int,
    t: wp.array(dtype=wp.int32), probe: wp.array3d(dtype=wp.float32),
):
    """Per-step (base_xy, ee_xy, obj_xy) tracks for population plots."""
    i = wp.tid()
    ee = site_xpos[i, pinch_site]
    probe[t[0], i, 0] = qpos[i, base_qadr[0]]
    probe[t[0], i, 1] = qpos[i, base_qadr[1]]
    probe[t[0], i, 2] = ee[0]
    probe[t[0], i, 3] = ee[1]
    probe[t[0], i, 4] = qpos[i, obj_qadr]
    probe[t[0], i, 5] = qpos[i, obj_qadr + 1]


@wp.kernel
def _record_qpos(qpos: wp.array2d(dtype=wp.float32), t: wp.array(dtype=wp.int32),
                 traj: wp.array3d(dtype=wp.float32)):
    """All worlds' qpos rows into the (T, n, nq) trajectory buffer."""
    i, j = wp.tid()
    traj[t[0], i, j] = qpos[i, j]


@wp.kernel
def _tick(t: wp.array(dtype=wp.int32)):
    t[0] += 1


class RolloutState:
    """Persistent per-world device buffers threaded through the control kernels."""

    def __init__(self, n):
        f, i, v2, v3 = wp.float32, wp.int32, wp.vec2, wp.vec3
        self.cost = wp.zeros(n, dtype=f)
        self.n_act = wp.zeros(n, dtype=f)
        self.succ = wp.zeros(n, dtype=i)
        self.done = wp.zeros(n, dtype=i)
        self.at_pre = wp.zeros(n, dtype=i)
        self.reached = wp.zeros(n, dtype=i)
        self.reach_t = wp.zeros(n, dtype=i)
        self.hold = wp.zeros(n, dtype=i)
        self.qc = wp.zeros((n, 6), dtype=f)
        self.base_cmd = wp.zeros(n, dtype=v3)
        self.vdir = wp.zeros(n, dtype=v2)
        self.q_a = wp.zeros((n, 6), dtype=f)     # grasp/place/push arm config
        self.q_b = wp.zeros((n, 6), dtype=f)     # pick lift/pre-grasp arm config
        self.goal_world = wp.zeros(n, dtype=v3)
        self.goal_xy = wp.zeros(n, dtype=v2)
        self.base_target = wp.zeros(n, dtype=v3)
        self.base_vel = wp.zeros(1, dtype=f)     # per-action scalar (device: graphs bake scalars)
        self.t = wp.zeros(1, dtype=i)            # recording step counter

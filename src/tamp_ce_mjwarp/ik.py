"""Whole-body damped-least-squares IK for the Dinova (holonomic base + Gen3 Lite arm),
batched over worlds on the GPU.

One solver, one minimal kinematic model (`dinova_fullbody_ik.xml`), one frame (WORLD):
given desired pinch-site poses it iterates Newton-DLS on the device buffer `q` (n, 9) =
[base_x, base_y, base_th, arm0..arm5]. `mode` picks a per-DOF mobility preset:
  - "full": the base is cheap to move, so the solver translates the base to reach while a
    nullspace bias keeps the arm near home.
  - "arm":  base mobility zeroed — only the arm moves, base_x/y return the warm start.

Key details that make it converge for top-down grasps: full SO(3) orientation error, a
nullspace bias toward the retract posture, and a per-DOF step cap. The Newton loop
[set_qpos, kinematics, com_pos, jac, dls] x iters is captured once as a CUDA graph;
`solve` writes the mode/ori_weight buffers and replays it.
"""

from __future__ import annotations

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from tamp_ce_mjwarp.sim import ARM_JOINTS, BASE_JOINTS, HOME_ARM, PINCH_SITE

DAMPING = 1e-4
MAX_STEP = np.deg2rad(45.0)
PI = float(np.pi)
# Top-down grasp orientation (gripper +z points to world -z): columns = local axes.
TOPDOWN_MAT = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])

# Per-DOF mobility [base_x, base_y, base_th, arm0..arm5] (0 = frozen, larger = moves more
# freely). NB "arm" mode does not face the target, so the target must be in the arm's
# workspace given the current base pose.
FULL_MOB = np.array([4.0, 4.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
ARM_MOB = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
# Per-DOF per-iteration step cap: base x,y in m, base th + arm in rad.
MAX_STEP_DOF = np.array([0.1, 0.1] + [MAX_STEP] * 7)

vec6f = wp.types.vector(length=6, dtype=wp.float32)
vec9f = wp.types.vector(length=9, dtype=wp.float32)
mat66f = wp.types.matrix(shape=(6, 6), dtype=wp.float32)
mat69f = wp.types.matrix(shape=(6, 9), dtype=wp.float32)


@wp.func
def _so3_log(R: wp.mat33) -> wp.vec3:
    """Rotation vector (axis*angle) of a rotation matrix; valid for large angles."""
    cos = wp.clamp((wp.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    angle = wp.acos(cos)
    w = wp.vec3(R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1])
    factor = wp.where(angle < 1e-6, 0.5, angle / (2.0 * wp.sin(angle) + 1e-12))
    return factor * w


@wp.func
def _chol6(A: mat66f) -> mat66f:
    L = mat66f(0.0)
    for j in range(6):
        s = A[j, j]
        for k in range(j):
            s -= L[j, k] * L[j, k]
        L[j, j] = wp.sqrt(wp.max(s, 1e-12))
        for r in range(j + 1, 6):
            s2 = A[r, j]
            for k in range(j):
                s2 -= L[r, k] * L[j, k]
            L[r, j] = s2 / L[j, j]
    return L


@wp.func
def _chol_solve6(L: mat66f, b: vec6f) -> vec6f:
    w = vec6f(0.0)
    for r in range(6):
        s = b[r]
        for k in range(r):
            s -= L[r, k] * w[k]
        w[r] = s / L[r, r]
    x = vec6f(0.0)
    for r in range(5, -1, -1):
        s2 = w[r]
        for k in range(r + 1, 6):
            s2 -= L[k, r] * x[k]
        x[r] = s2 / L[r, r]
    return x


@wp.kernel
def _set_qpos(q: wp.array2d(dtype=wp.float32), qadr: wp.array(dtype=wp.int32),
              qpos: wp.array2d(dtype=wp.float32)):
    i = wp.tid()
    for j in range(9):
        qpos[i, qadr[j]] = q[i, j]


@wp.kernel
def _site_point(site_xpos: wp.array2d(dtype=wp.vec3), site: int,
                point: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    point[i] = site_xpos[i, site]


@wp.kernel
def _dls(
    site: int,
    vadr: wp.array(dtype=wp.int32),
    site_xpos: wp.array2d(dtype=wp.vec3),
    site_xmat: wp.array2d(dtype=wp.mat33),
    jacp: wp.array3d(dtype=wp.float32),
    jacr: wp.array3d(dtype=wp.float32),
    target_pos: wp.array(dtype=wp.vec3),
    target_mat: wp.array(dtype=wp.mat33),
    winv: wp.array(dtype=wp.float32),      # (9,) per-DOF mobility
    tw: wp.array(dtype=wp.float32),        # (6,) task-row weights
    q0: wp.array(dtype=wp.float32),        # (9,) nullspace target (arm home)
    max_step: wp.array(dtype=wp.float32),  # (9,)
    damping: float,
    q: wp.array2d(dtype=wp.float32),
):
    """One weighted-DLS Newton step with nullspace arm-home bias and per-DOF step cap."""
    i = wp.tid()
    x = site_xpos[i, site]
    e_p = target_pos[i] - x
    e_r = _so3_log(target_mat[i] * wp.transpose(site_xmat[i, site]))
    e = vec6f(tw[0] * e_p[0], tw[1] * e_p[1], tw[2] * e_p[2],
              tw[3] * e_r[0], tw[4] * e_r[1], tw[5] * e_r[2])

    J = mat69f(0.0)
    for c in range(9):
        v = vadr[c]
        for r in range(3):
            J[r, c] = tw[r] * jacp[i, r, v]
            J[r + 3, c] = tw[r + 3] * jacr[i, r, v]

    A = mat66f(0.0)  # J diag(winv) J^T + damping I
    for r in range(6):
        for c in range(6):
            s = float(0.0)
            for k in range(9):
                s += J[r, k] * winv[k] * J[c, k]
            A[r, c] = s
    for r in range(6):
        A[r, r] += damping

    L = _chol6(A)
    y = _chol_solve6(L, e)

    # nullspace bias toward q0 on the arm DOFs: null@q0e = q0e - winv * J^T A^-1 (J q0e)
    q0e = vec9f(0.0)
    for k in range(3, 9):
        dq = q0[k] - q[i, k]
        q0e[k] = dq - 2.0 * PI * wp.floor((dq + PI) / (2.0 * PI))
    t = vec6f(0.0)
    for r in range(6):
        s = float(0.0)
        for k in range(9):
            s += J[r, k] * q0e[k]
        t[r] = s
    z = _chol_solve6(L, t)

    for k in range(9):
        jy = float(0.0)
        jz = float(0.0)
        for r in range(6):
            jy += J[r, k] * y[r]
            jz += J[r, k] * z[r]
        upd = winv[k] * jy + q0e[k] - winv[k] * jz
        q[i, k] += wp.clamp(upd, -max_step[k], max_step[k])


class IK:
    """Batched world-frame whole-body DLS IK; see module docstring. Callers write the warm
    start into `self.q` (n, 9) and the targets into `self.target_pos`/`self.target_mat`
    (device buffers), then call `solve`; the solution is left in `self.q`."""

    def __init__(self, n, xml_path="robot_models/dinova/dinova_fullbody_ik.xml",
                 arm_home=HOME_ARM, damping=DAMPING, pinch_z=None, iters=20):
        m = mujoco.MjModel.from_xml_path(xml_path)
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, PINCH_SITE)
        if pinch_z is not None:  # move the tool point along the gripper +z (grasp tuning)
            m.site_pos[self.site] = (0.0, 0.0, pinch_z)
        joints = BASE_JOINTS + ARM_JOINTS
        J = mujoco.mjtObj.mjOBJ_JOINT
        qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, J, nm)] for nm in joints]
        vadr = [m.jnt_dofadr[mujoco.mj_name2id(m, J, nm)] for nm in joints]

        self.m = mjw.put_model(m)
        self.d = mjw.make_data(m, nworld=n)
        self.n, self.iters, self.damping = n, iters, float(damping)

        self.q = wp.zeros((n, 9), dtype=wp.float32)
        self.target_pos = wp.zeros(n, dtype=wp.vec3)
        self.target_mat = wp.array(np.tile(TOPDOWN_MAT, (n, 1, 1)), dtype=wp.mat33)
        self.jacp = wp.zeros((n, 3, m.nv), dtype=wp.float32)
        self.jacr = wp.zeros((n, 3, m.nv), dtype=wp.float32)
        self.point = wp.zeros(n, dtype=wp.vec3)
        self.bodyid = wp.array(np.full(n, int(m.site_bodyid[self.site]), dtype=np.int32))
        self.qadr = wp.array(np.asarray(qadr, dtype=np.int32))
        self.vadr = wp.array(np.asarray(vadr, dtype=np.int32))
        self.winv = wp.zeros(9, dtype=wp.float32)
        self.tw = wp.zeros(6, dtype=wp.float32)
        self.q0 = wp.array(np.concatenate([np.zeros(3), arm_home]).astype(np.float32))
        self.max_step = wp.array(MAX_STEP_DOF.astype(np.float32))
        self.mob = {"full": FULL_MOB.astype(np.float32), "arm": ARM_MOB.astype(np.float32)}

        self._newton(1)  # warm-up (compiles all kernels), then capture the full loop
        with wp.ScopedCapture() as cap:
            self._newton(self.iters)
        self._graph = cap.graph

    def _newton(self, iters):
        for _ in range(iters):
            wp.launch(_set_qpos, dim=self.n, inputs=[self.q, self.qadr, self.d.qpos])
            mjw.kinematics(self.m, self.d)
            mjw.com_pos(self.m, self.d)
            wp.launch(_site_point, dim=self.n, inputs=[self.d.site_xpos, self.site, self.point])
            mjw.jac(self.m, self.d, self.jacp, self.jacr, self.point, self.bodyid)
            wp.launch(_dls, dim=self.n, inputs=[
                self.site, self.vadr, self.d.site_xpos, self.d.site_xmat, self.jacp,
                self.jacr, self.target_pos, self.target_mat, self.winv, self.tw,
                self.q0, self.max_step, self.damping, self.q])

    def solve(self, mode="full", ori_weight=1.0):
        """Run the captured Newton loop on the current q/target buffers. `ori_weight`
        scales the orientation task rows: 1.0 = full pose IK, 0.0 = position-only
        (`target_mat` is then ignored)."""
        self.winv.assign(self.mob[mode])
        ow = float(ori_weight)
        self.tw.assign(np.array([1.0, 1.0, 1.0, ow, ow, ow], dtype=np.float32))
        wp.capture_launch(self._graph)

"""Batched IK validation against host-side forward kinematics (ground truth): full-mode
solves must place the pinch site at the target pose; arm mode must freeze the base."""

import mujoco
import numpy as np
import warp as wp

from tamp_ce_mjwarp.ik import IK, TOPDOWN_MAT
from tamp_ce_mjwarp.sim import HOME_ARM

IK_XML = "robot_models/dinova/dinova_fullbody_ik.xml"


def _fk_errors(q, targets):
    m = mujoco.MjModel.from_xml_path(IK_XML)
    d = mujoco.MjData(m)
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
    perr, oerr = [], []
    for i in range(len(q)):
        d.qpos[:9] = q[i]
        mujoco.mj_kinematics(m, d)
        perr.append(np.linalg.norm(d.site_xpos[site] - targets[i]))
        R = d.site_xmat[site].reshape(3, 3)
        oerr.append(np.arccos(np.clip((np.trace(TOPDOWN_MAT @ R.T) - 1) / 2, -1, 1)))
    return np.array(perr), np.array(oerr)


def test_full_mode_converges():
    n = 16
    ik = IK(n)
    rng = np.random.default_rng(0)
    targets = np.stack([rng.uniform(0.5, 2.0, n), rng.uniform(-0.5, 0.5, n),
                        rng.uniform(0.2, 0.6, n)], axis=1).astype(np.float32)
    ik.q.assign(np.concatenate([np.zeros((n, 3)), np.tile(HOME_ARM, (n, 1))],
                               axis=1).astype(np.float32))
    ik.target_pos.assign(targets)
    ik.target_mat.assign(np.tile(TOPDOWN_MAT, (n, 1, 1)).astype(np.float32))
    ik.solve(mode="full")
    wp.synchronize()
    perr, oerr = _fk_errors(ik.q.numpy(), targets)
    assert perr.max() < 0.01
    assert np.rad2deg(oerr.max()) < 2.0


def test_arm_mode_freezes_base():
    n = 8
    ik = IK(n)
    rng = np.random.default_rng(1)
    base_xy = rng.uniform(-0.5, 0.5, (n, 2)).astype(np.float32)
    q0 = np.concatenate([base_xy, np.zeros((n, 1)), np.tile(HOME_ARM, (n, 1))],
                        axis=1).astype(np.float32)
    targets = np.concatenate([base_xy + [0.45, 0.0],
                              np.full((n, 1), 0.35, np.float32)], axis=1)
    ik.q.assign(q0)
    ik.target_pos.assign(targets.astype(np.float32))
    ik.target_mat.assign(np.tile(TOPDOWN_MAT, (n, 1, 1)).astype(np.float32))
    ik.solve(mode="arm")
    wp.synchronize()
    q = ik.q.numpy()
    assert np.allclose(q[:, :2], base_xy, atol=1e-6)
    perr, _ = _fk_errors(q, targets)
    assert perr.max() < 0.01

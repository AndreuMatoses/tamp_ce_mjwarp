"""Symbolic regions = the sampling distributions for action parameters.

A region produces a flat parameter vector (the thing CE optimizes). CE refits an
independent per-dimension Gaussian over the elite samples; `sample_normal` re-applies
per-kind fixups (quaternion renormalization). Kinds:

- "box"        : n-D uniform box  (base x,y goals; generic)
- "annulus"    : 2-D ring around a center  (be near a table)
- "quat"       : 4-D quat (x,y,z,w) around a top-down seed — a full `yaw`-bounded spin about
                 the gripper approach axis (local z) + a small `tilt`-radian cone about local
                 x/y, applied via the exponential map  (pick orientation)
- "point"      : 3-D point (box, z upward from center), no orientation  (orientation-free place)
- "point_quat" : 3-D point + 4-D quat (same yaw+tilt grasp)  (place/push pose)

Samplers take a `np.random.Generator`. The mapping from a parameter vector to control
goals lives in `plan.py`/`actions.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

TOPDOWN_EULER = (180.0, 0.0, 0.0)  # gripper points down


def topdown_quat():
    """Seed grasp quaternion (x, y, z, w)."""
    return Rotation.from_euler("xyz", TOPDOWN_EULER, degrees=True).as_quat()


@dataclass(frozen=True)
class Region:
    kind: str
    a: np.ndarray  # center / lo / qseed depending on kind
    b: np.ndarray  # size / hi / qbound depending on kind
    name: str = ""  # for plot legends


def box(center, lo, hi, name=""):
    """Uniform box spanning [center+lo, center+hi]. Pass center=0 to make lo/hi absolute."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = np.broadcast_to(np.asarray(center, float), lo.shape)
    return Region("box", c + lo, c + hi, name)


def annulus(center, r_in, r_out, name=""):
    return Region("annulus", np.asarray(center, float), np.array([r_in, r_out], float), name)


def quat(tilt=0.3, yaw=float(np.pi), name=""):
    """Grasp-orientation region around a top-down seed:
    - yaw=<angle>: anisotropic — a full `yaw`-half-range spin about the gripper approach axis
      (local z; default pi = whole circle) + a small `tilt`-rad cone about local x/y.
    - yaw=None: isotropic — a uniform cone of half-angle `tilt` about all axes equally."""
    b = np.asarray([tilt] if yaw is None else [tilt, yaw], float)  # len 1 => iso, 2 => aniso
    return Region("quat", topdown_quat(), b, name)


def point_quat(center, size, tilt=0.3, yaw=float(np.pi), name=""):
    """3-D point box (x,y centered on `center`, z from center_z upward by size_z) + a quat
    part as in quat(): yaw=None -> isotropic cone (half-angle tilt); else spin + tilt."""
    tail = [tilt] if yaw is None else [tilt, yaw]
    return Region("point_quat", np.asarray(center, float),
                  np.concatenate([np.asarray(size, float), np.asarray(tail, float)]), name)


def point(center, size, name=""):
    """3-D position-only goal (same box as point_quat, minus the quaternion) — for an
    orientation-free IK goal (the action solves position-only, ori_weight=0)."""
    return Region("point", np.asarray(center, float), np.asarray(size, float), name)


def dim(r: Region) -> int:
    return {"box": r.a.shape[0], "annulus": 2, "quat": 4, "point": 3, "point_quat": 7}[r.kind]


def _norm_quat(q):
    return q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)


def _exp_quat(omega):
    """Quaternion (x,y,z,w) of a rotation vector `omega` (.., 3) via the so(3) exp map."""
    theta = np.linalg.norm(omega, axis=-1, keepdims=True)
    s = np.where(theta < 1e-8, 0.5, np.sin(0.5 * theta) / (theta + 1e-12))  # ->1/2 as theta->0
    return np.concatenate([omega * s, np.cos(0.5 * theta)], axis=-1)


def _quat_mul(a, b):
    """Hamilton product a (x) b, both (.., 4) in (x, y, z, w)."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], axis=-1)


def _sample_cone(rng, seed, bound, n):
    """n quaternions within a `bound`-radian isotropic cone around `seed`, sampled uniformly
    in the rotation-vector ball and applied via the exp map: q = seed (x) exp(omega)."""
    g = rng.normal(size=(n, 3))
    axis = g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-12)
    rad = bound * rng.uniform(size=(n, 1)) ** (1.0 / 3.0)  # uniform within the ball
    dq = _exp_quat(axis * rad)
    return _norm_quat(_quat_mul(np.broadcast_to(seed, (n, 4)), dq))


def _sample_grasp(rng, seed, tilt, yaw, n):
    """n grasp quaternions around `seed`: a full spin of half-range `yaw` (rad) about the
    gripper approach axis (local z) composed with a small `tilt`-radian cone about local x/y.

    q = seed (x) exp(tilt_xy) (x) exp(yaw_z). NB yaw in [-pi, pi] keeps w = cos(yaw/2) >= 0,
    so the full circle is a continuous arc on one quaternion hemisphere — no double-cover
    wrap — and CE's per-dim Gaussian refit over the 4 quat components stays sound."""
    phi = rng.uniform(size=(n, 1)) * 2 * np.pi                # tilt direction in xy
    ang = tilt * np.sqrt(rng.uniform(size=(n, 1)))            # tilt magnitude (uniform disk)
    q_tilt = _exp_quat(np.concatenate([ang * np.cos(phi), ang * np.sin(phi), np.zeros((n, 1))], -1))
    spin = (2 * rng.uniform(size=(n, 1)) - 1) * yaw           # U[-yaw, yaw] about local z
    q_yaw = _exp_quat(np.concatenate([np.zeros((n, 2)), spin], -1))
    seed = np.broadcast_to(seed, (n, 4))
    return _norm_quat(_quat_mul(_quat_mul(seed, q_tilt), q_yaw))


def sample_uniform(r: Region, rng, n):
    if r.kind == "box":
        return r.a + rng.uniform(size=(n, dim(r))) * (r.b - r.a)
    if r.kind == "annulus":
        ri, ro = r.b
        rad = np.sqrt(rng.uniform(size=n) * (ro**2 - ri**2) + ri**2)
        th = rng.uniform(size=n) * 2 * np.pi
        return r.a + np.stack([rad * np.cos(th), rad * np.sin(th)], axis=1)
    if r.kind == "quat":  # b = (tilt,) isotropic cone, or (tilt, yaw) anisotropic spin+tilt
        if r.b.shape[0] == 1:
            return _sample_cone(rng, r.a, r.b[0], n)
        return _sample_grasp(rng, r.a, r.b[0], r.b[1], n)
    if r.kind == "point":  # 3-D box: x,y centered; z upward from center
        size = r.b
        off = rng.uniform(size=(n, 3)) * np.array([2 * size[0], 2 * size[1], size[2]]) \
            - np.array([size[0], size[1], 0.0])
        return r.a + off
    if r.kind == "point_quat":
        size = r.b[:3]
        off = rng.uniform(size=(n, 3)) * np.array([2 * size[0], 2 * size[1], size[2]]) \
            - np.array([size[0], size[1], 0.0])
        pts = r.a + off
        qs = (_sample_cone(rng, topdown_quat(), r.b[3], n) if r.b.shape[0] == 4
              else _sample_grasp(rng, topdown_quat(), r.b[3], r.b[4], n))
        return np.concatenate([pts, qs], axis=1)
    raise ValueError(r.kind)


def sample_normal(r: Region, rng, n, mean, std, n_std=3.0):
    x = mean + std * rng.normal(size=(n, mean.shape[0]))
    x = np.clip(x, mean - n_std * std, mean + n_std * std)
    if r.kind == "quat":
        return _norm_quat(x)
    if r.kind == "point_quat":
        return np.concatenate([x[:, :3], _norm_quat(x[:, 3:])], axis=1)
    return x

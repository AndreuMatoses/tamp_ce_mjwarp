"""MuJoCo Warp model loading and name-based indexing.

`Sim` is a host-side object: it owns the MuJoCo model (patched for fast batched rollout)
and its `mujoco_warp` counterpart, and resolves all joint/actuator/site indices once, by
name, so the rest of the code never touches magic DOF numbers. Batched simulation state
(`mjw.Data`) is owned by `plan.Rollout`.
"""

from __future__ import annotations

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

# Dinova name groups (the only robot in this repo).
BASE_JOINTS = ("joint_x", "joint_y", "joint_th")
ARM_JOINTS = tuple(f"Kinova_{i}" for i in range(6))
GRIPPER_ACT = "gripper_actuator"
PINCH_SITE = "pinch_site"

# Geom groups in dinova_robot.xml: 2 = visual meshes, 4 = simplified collision proxy.
VISUAL_GROUP = 2
ROBOT_COLLISION_GROUP = 4

# Retract home arm config (rad): gripper pointing down in front, wrist clear of its
# limit so top-down IK stays in a clean basin. All rollouts start here.
HOME_ARM = (-0.1169, -0.6632, 0.5009, 1.5708, 1.9775, 1.4539)
GRIPPER_OPEN = 0.5

# 0.003 is the largest timestep that keeps grasping at 100% (0.004 -> 44%: the stiff
# grasp contacts go unstable). Solver 20/10 iters holds grasp success at 100% (10 -> 81%).
DT = 0.003
SOLVER_ITERATIONS = 20
LS_ITERATIONS = 10


def _sec2steps(t):
    """Seconds -> number of sim steps at DT (does not hold for a custom Sim dt)."""
    return round(t / DT)


def _id(model, objtype, name):
    i = mujoco.mj_name2id(model, objtype, name)
    if i < 0:
        raise ValueError(f"no {objtype} named {name!r} in model")
    return i


class Sim:
    """Patched model + name-resolved indices. `enable_robot_collision` keeps the robot's
    simplified collision proxy (group 4 capsules/boxes) active; mesh geoms never collide
    (scene obstacles/objects must use collision primitives)."""

    def __init__(self, xml_path: str, dt: float = DT, iterations=SOLVER_ITERATIONS,
                 ls_iterations=LS_ITERATIONS, gripper_kp=None, enable_robot_collision=True):
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.enable_robot_collision = enable_robot_collision
        m = self.mj_model
        m.opt.timestep = dt
        if iterations is not None:
            m.opt.iterations = iterations
        if ls_iterations is not None:
            m.opt.ls_iterations = ls_iterations

        # Optional gripper clamp-force override (position actuator: gain kp, bias -kp).
        if gripper_kp is not None:
            gi = _id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACT)
            m.actuator_gainprm[gi, 0] = gripper_kp
            m.actuator_biasprm[gi, 1] = -gripper_kp

        # Mesh geoms (arm/chassis visuals) never collide; the proxy (group 4) does.
        mesh = m.geom_type == mujoco.mjtGeom.mjGEOM_MESH
        m.geom_contype[mesh] = 0
        m.geom_conaffinity[mesh] = 0

        if not enable_robot_collision:
            proxy = m.geom_group == ROBOT_COLLISION_GROUP
            m.geom_contype[proxy] = 0
            m.geom_conaffinity[proxy] = 0
        else:
            # Proxy ON, but EXCLUDE robot-vs-floor: the base box rests on the floor every
            # step in every world — a huge, useless contact load (the base height is fixed).
            # Bit split: floor on its own affinity bit the proxy doesn't match; free bodies
            # (the cube) keep BOTH bits so they still land on the floor if dropped.
            OBST, FLOOR = 1, 2
            proxy = m.geom_group == ROBOT_COLLISION_GROUP
            m.geom_contype[proxy] = OBST
            m.geom_conaffinity[proxy] = OBST
            planes = m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
            m.geom_contype[planes] = FLOOR
            m.geom_conaffinity[planes] = FLOOR
            free_bodies = [m.jnt_bodyid[j] for j in range(m.njnt)
                           if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
            free = np.isin(m.geom_bodyid, free_bodies)
            m.geom_contype[free] = OBST | FLOOR
            m.geom_conaffinity[free] = OBST | FLOOR

        # NB: no gravity compensation. A post-compile body_gravcomp=1 patch is a no-op in
        # MuJoCo/MJX (gated on compile-time ngravcomp==0), but MJWarp actually applies it —
        # including to free objects, which then float instead of falling.

        self.model = mjw.put_model(m)
        self.dt = float(m.opt.timestep)

        A, J, S = mujoco.mjtObj.mjOBJ_ACTUATOR, mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_SITE

        # ctrl indices (ctrl is laid out by actuator id)
        self.base_act = np.array([_id(m, A, n) for n in BASE_JOINTS])
        self.arm_act = np.array([_id(m, A, n) for n in ARM_JOINTS])
        self.gripper_act = _id(m, A, GRIPPER_ACT)

        # qpos / dof addresses
        self.base_qadr = np.array([m.jnt_qposadr[_id(m, J, n)] for n in BASE_JOINTS])
        self.arm_qadr = np.array([m.jnt_qposadr[_id(m, J, n)] for n in ARM_JOINTS])
        self.grip_qadr = int(self.arm_qadr[-1] + 1)  # gripper_joint qpos (after the arm)

        self.pinch_site = _id(m, S, PINCH_SITE)
        self.home_arm = np.asarray(HOME_ARM)

        # device copies for kernels
        self.base_act_wp = wp.array(self.base_act, dtype=wp.int32)
        self.arm_act_wp = wp.array(self.arm_act, dtype=wp.int32)
        self.base_qadr_wp = wp.array(self.base_qadr, dtype=wp.int32)
        self.arm_qadr_wp = wp.array(self.arm_qadr, dtype=wp.int32)

    def body_id(self, name):
        return _id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)

    def geom_xy(self, name):
        """World (x, y) of a named geom — lets regions inherit a scene object's center."""
        g = _id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        return tuple(float(v) for v in self.mj_model.geom_pos[g][:2])

    def body_xy(self, name):
        """Initial world (x, y) of a named body — for free bodies (start at the body's
        XML pos = freejoint qpos0), whose geom_pos is body-relative."""
        return tuple(float(v) for v in self.mj_model.body(name).pos[:2])

    def joint_qadr(self, name):
        """qpos address of a named joint (e.g. a free body's freejoint => its (x,y,z,quat))."""
        return int(self.mj_model.jnt_qposadr[_id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)])

    def geom_rect(self, name, inflate=0.0):
        """Planar AABB (xmin, xmax, ymin, ymax, inflate) of a named box geom, for use as
        a path-planner obstacle."""
        m = self.mj_model
        g = _id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        px, py = m.geom_pos[g][:2]
        sx, sy = m.geom_size[g][:2]
        return (float(px - sx), float(px + sx), float(py - sy), float(py + sy), float(inflate))

    def geom_top_z(self, name):
        """Top-surface world z of a named box geom (center z + half-height)."""
        m = self.mj_model
        g = _id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        return float(m.geom_pos[g][2] + m.geom_size[g][2])

    def home_mjd(self, graspable_qadr=None, graspable_xy=None):
        """Forward-evaluated host MjData at the home pose (arm held, gripper open), with the
        graspable object optionally moved to `graspable_xy`. Broadcast to worlds by put_data."""
        d = mujoco.MjData(self.mj_model)
        d.qpos[self.arm_qadr] = self.home_arm
        d.ctrl[self.arm_act] = self.home_arm
        d.ctrl[self.gripper_act] = GRIPPER_OPEN
        if graspable_xy is not None:
            d.qpos[graspable_qadr:graspable_qadr + 2] = graspable_xy
        mujoco.mj_forward(self.mj_model, d)
        return d

"""Plan = an ordered list of action specs, executed for all worlds in one batched rollout.

`Rollout` owns the batched `mjw.Data`, the per-world control state, and one captured CUDA
graph per action kind (control kernel + step [+ recording]); graphs are captured once and
replayed for every step of every action of every CE iteration — everything that varies
lives in device buffers. Each `_prep_<kind>` turns the sampled parameter batch into the
targets its control kernel needs (one whole-body `IK` solve per pick/place/push: the base
part is the base target the action drives to, the arm part the joint goal; `ActionSpec.ik`
picks the IK mobility preset).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco_warp as mjw
import numpy as np
import warp as wp

from tamp_ce_mjwarp import actions as A
from tamp_ce_mjwarp.nav import NavBatch
from tamp_ce_mjwarp.sim import _sec2steps


@dataclass(frozen=True)
class ActionSpec:
    kind: str                # "move" | "pick" | "place" | "push"
    timeout: float           # action horizon in seconds (-> steps via sim._sec2steps)
    obj_qadr: int = -1       # pick: object freejoint qpos addr (its center = position goal)
    z_off: float = 0.01      # pick: z offset above the object center
    ik: str = "arm"          # IK mobility preset: "arm" (base holds) | "full" (base drives in)
    free_ori: bool = False   # place: position-only IK (3-D `point` region, not 7-D `point_quat`)


class Rollout:
    """Batched plan executor. `run(params)` -> dict of per-world numpy results.

    `record_probe` keeps per-step (base_xy, ee_xy, obj_xy) tracks for population plots;
    `record_qpos` keeps world 0's full qpos trajectory (for replay/rendering)."""

    EXIT_CHECK = 128  # steps between all-done readbacks (each costs one small sync)

    def __init__(self, sim, ik, scenario, n, record_probe=False, record_qpos=False,
                 nconmax=None, njmax=256, early_exit=True, exit_frac=1.0):
        assert ik.n == n, f"IK batch size {ik.n} != rollout batch size {n}"
        self.sim, self.ik, self.scenario, self.n = sim, ik, scenario, n
        # exit_frac < 1: end an action once this fraction of worlds has succeeded (a few
        # stragglers otherwise force the full horizon at large n). Cut-short failed
        # worlds accrue less step cost, so keep 1.0 when exact cost parity matters.
        self.exit_frac = exit_frac
        self.steps = [_sec2steps(s.timeout) for s in scenario.specs]
        # njmax default: the ramp scenario peaks at ~155 constraint rows/world (overflow
        # prints a warning and silently degrades contacts, so keep headroom)
        self.d = mjw.put_data(sim.mj_model,
                              sim.home_mjd(scenario.graspable_qadr, scenario.graspable_xy),
                              nworld=n, nconmax=nconmax, njmax=njmax)
        self.state = A.RolloutState(n)
        self.home_arm_wp = wp.array(sim.home_arm.astype(np.float32))
        self.nav = NavBatch(scenario.nav, n) if scenario.nav is not None else None
        self._gxy = self.nav.gxy if self.nav else wp.zeros((1, 1, 1), dtype=wp.vec2)
        if self.nav:
            _, self._xmin, self._ymin = None, self.nav.bounds[0], self.nav.bounds[2]
            self._res = self.nav.res
        else:
            self._xmin = self._ymin = 0.0
            self._res = 1.0

        self.record_probe, self.record_qpos = record_probe, record_qpos
        self.early_exit = early_exit
        self.steps_run = sum(self.steps)  # actual steps of the last run (early exit)
        T = sum(self.steps)
        self.probe = wp.zeros((T, n, 6), dtype=wp.float32) if record_probe else None
        self.traj = wp.zeros((T, n, sim.mj_model.nq), dtype=wp.float32) if record_qpos else None

        self._pristine = {f: wp.clone(getattr(self.d, f))
                          for f in ("qpos", "qvel", "ctrl", "act", "qacc_warmstart", "time")}
        self._capture()

    # -- graph capture ---------------------------------------------------------------

    def _ctrl_launch(self, kind):
        sim, st, d = self.sim, self.state, self.d
        if kind == "move":
            wp.launch(A._ctrl_move, dim=self.n, inputs=[
                d.qpos, d.ctrl, sim.base_qadr_wp, sim.base_act_wp, st.goal_xy, self._gxy,
                int(self.nav is not None), self._xmin, self._ymin, self._res, sim.dt,
                st.vdir, st.done, st.succ, st.cost])
        elif kind == "pick":
            wp.launch(A._ctrl_pick, dim=self.n, inputs=[
                d.qpos, d.ctrl, d.site_xpos, sim.arm_act_wp, sim.base_act_wp,
                sim.gripper_act, sim.pinch_site, sim.grip_qadr, sim.dt,
                st.q_a, st.q_b, st.goal_world, st.base_target, st.base_vel,
                st.qc, st.base_cmd, st.at_pre, st.reached, st.reach_t, st.hold,
                st.succ, st.cost])
        elif kind in ("place", "push"):
            kern = A._ctrl_place if kind == "place" else A._ctrl_push
            wp.launch(kern, dim=self.n, inputs=[
                d.qpos, d.ctrl, d.site_xpos, sim.arm_act_wp, sim.base_act_wp,
                sim.gripper_act, sim.pinch_site, sim.dt,
                st.q_a, st.goal_world, st.base_target, st.base_vel,
                st.qc, st.base_cmd, st.reached, st.succ, st.cost])
        else:
            raise ValueError(kind)

    def _step_once(self, kind):
        self._ctrl_launch(kind)
        mjw.step(self.sim.model, self.d)
        st = self.state
        if self.record_probe:
            wp.launch(A._record_probe, dim=self.n, inputs=[
                self.d.qpos, self.d.site_xpos, self.sim.base_qadr_wp, self.sim.pinch_site,
                self.scenario.graspable_qadr, st.t, self.probe])
        if self.record_qpos:
            wp.launch(A._record_qpos, dim=(self.n, self.sim.mj_model.nq),
                      inputs=[self.d.qpos, st.t, self.traj])
        if self.record_probe or self.record_qpos:
            wp.launch(A._tick, dim=1, inputs=[st.t])

    def _capture(self):
        kinds = {s.kind for s in self.scenario.specs}
        self.graphs = {}
        for k in kinds:
            self._step_once(k)  # warm-up: compiles every kernel the graph will contain
        for k in kinds:
            with wp.ScopedCapture() as cap:
                self._step_once(k)
            self.graphs[k] = cap.graph

    # -- per-action prep -------------------------------------------------------------

    def _solve_arm_goal(self, p, spec, target_mat=None, lift=False):
        """Whole-body IK from [current base, home arm] toward the action's EE goal already
        in state.goal_world; fills q_a (+ q_b at goal+LIFT_H if `lift`) and base_target."""
        ik, st = self.ik, self.state
        wp.launch(A._prep_qinit, dim=self.n,
                  inputs=[self.d.qpos, self.sim.base_qadr_wp, self.home_arm_wp, ik.q])
        if target_mat is not None:
            ik.target_mat.assign(np.ascontiguousarray(target_mat, dtype=np.float32))
        wp.launch(A._set_ik_target, dim=self.n, inputs=[st.goal_world, 0.0, ik.target_pos])
        ik.solve(mode=spec.ik, ori_weight=0.0 if spec.free_ori else 1.0)
        wp.launch(A._store_q, dim=self.n, inputs=[ik.q, st.q_a, st.base_target, 1])
        if lift:
            wp.launch(A._set_ik_target, dim=self.n,
                      inputs=[st.goal_world, A.LIFT_H, ik.target_pos])
            ik.solve(mode=spec.ik)  # warm-started from the grasp config
            wp.launch(A._store_q, dim=self.n, inputs=[ik.q, st.q_b, st.base_target, 0])
        st.base_vel.assign(np.array([A.BASE_IK_VEL if spec.ik == "full" else 0.0], np.float32))

    def _prep(self, spec, p):
        st = self.state
        if spec.kind == "move":
            st.goal_xy.assign(np.ascontiguousarray(p[:, :2], dtype=np.float32))
            if self.nav:
                self.nav.compute(p[:, :2])
        elif spec.kind == "pick":
            wp.launch(A._prep_pick_goal, dim=self.n,
                      inputs=[self.d.qpos, int(spec.obj_qadr), float(spec.z_off),
                              st.goal_world])
            self._solve_arm_goal(p, spec, target_mat=A.quat_to_mat(p), lift=True)
        elif spec.kind in ("place", "push"):
            st.goal_world.assign(np.ascontiguousarray(p[:, :3], dtype=np.float32))
            mat = None if spec.free_ori else A.quat_to_mat(p[:, 3:])
            self._solve_arm_goal(p, spec, target_mat=mat)
        else:
            raise ValueError(spec.kind)

    # -- rollout ---------------------------------------------------------------------

    def reset(self):
        for f, src in self._pristine.items():
            wp.copy(getattr(self.d, f), src)
        st = self.state
        st.cost.zero_()
        st.n_act.zero_()
        st.t.zero_()
        mjw.forward(self.sim.model, self.d)  # refresh site_xpos etc. for the first ctrl read

    def _run_action(self, spec, steps):
        """Step one action to its timeout, ending EXIT_TAIL[kind] after every world has
        latched success (early exit; failed worlds keep the full horizon). Returns the
        number of steps executed."""
        g, st = self.graphs[spec.kind], self.state
        tail = _sec2steps(A.EXIT_TAIL[spec.kind])
        exit_at = None
        t = 0
        while t < steps:
            wp.capture_launch(g)
            t += 1
            if exit_at is not None:
                if t >= exit_at:
                    break
            elif self.early_exit and t % self.EXIT_CHECK == 0 and t + tail < steps:
                if st.succ.numpy().mean() >= self.exit_frac:
                    exit_at = t + tail
        return t

    def run(self, params):
        """Execute the plan for all worlds. `params`: one (n, dim) array per action.
        Returns {cost, n_act, ok, proxy, qpos} numpy arrays (per world)."""
        sim, st, scen = self.sim, self.state, self.scenario
        self.reset()
        self.steps_run = 0
        for spec, p, steps in zip(scen.specs, params, self.steps):
            wp.launch(A._init_action, dim=self.n, inputs=[
                self.d.qpos, sim.arm_qadr_wp, sim.base_qadr_wp, st.qc, st.base_cmd,
                st.vdir, st.at_pre, st.reached, st.reach_t, st.hold, st.done, st.succ])
            self._prep(spec, np.asarray(p, dtype=np.float32))
            self.steps_run += self._run_action(spec, steps)
            wp.launch(A._finish_action, dim=self.n, inputs=[st.succ, st.n_act])
        wp.synchronize()
        cost, n_act = st.cost.numpy(), st.n_act.numpy()
        qpos = self.d.qpos.numpy()
        if scen.goal is not None:
            ok, proxy = scen.goal(sim, qpos)
        else:
            ok, proxy = n_act >= len(scen.specs), np.zeros(self.n, np.float32)
        ok, proxy = np.asarray(ok, bool), np.asarray(proxy, np.float32)
        # exploded worlds (NaN state, e.g. a violent uniform-sampled contact in float32)
        # must never rank as elites
        bad = ~np.isfinite(qpos).all(axis=1) | ~np.isfinite(proxy)
        if bad.any():
            ok[bad], proxy[bad], n_act[bad], cost[bad] = False, np.inf, 0.0, np.inf
        return {"cost": cost, "n_act": n_act, "ok": ok, "proxy": proxy, "qpos": qpos}


def replay(sim, ik, scenario, params):
    """Re-run a plan for one world with the given per-action parameter vectors, returning
    the qpos trajectory (T, nq). Builds a fresh single-world Rollout (needs ik.n == 1)."""
    ro = Rollout(sim, ik, scenario, 1, record_qpos=True)
    ro.run([np.asarray(p, dtype=np.float32)[None] for p in params])
    return ro.traj.numpy()[:ro.steps_run, 0]

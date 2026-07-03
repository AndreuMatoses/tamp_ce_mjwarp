# CLAUDE.md

CE optimization over continuous parameters of symbolic TAMP plans, scored by batched
MuJoCo Warp rollouts. Port of `../tamp_ce_jax` (MJX); see README.md for usage.
Run things with `uv run python ...`; tests with `uv run pytest`.

## Layout

```
src/tamp_ce_mjwarp/
  sim.py        Sim: MjModel load/patch (collision proxy, solver iters) + name->index maps
  actions.py    per-kind Warp control kernels + init/prep/record kernels + RolloutState
  plan.py       ActionSpec + Rollout (batched Data, captured graphs, action boundaries, run())
  ik.py         batched whole-body DLS IK on the minimal model (one captured Newton graph)
  nav.py        VI nav field: GPU relaxation kernel + host gradient; occupancy helpers
  regions.py    sampling regions (box/annulus/quat/point/point_quat), numpy rng
  ce.py         CE loop: draw -> Rollout.run -> tiered elites -> per-dim Gaussian refit
  scenarios.py  scenario registry: scene + regions + specs + vectorized symbolic goal
  viz.py        rendering/viewers, population & convergence plots, HDF5 solution I/O
scenes/, robot_models/dinova/   MJCF assets (collision primitives only; meshes are visual)
examples/     runnable entry points (CE runs, replays, region inspection)
tests/        IK convergence + batched grasp canary
```

## Adding a skill (action kind)

1. `actions.py`: write `_ctrl_<kind>` — one kernel, dim `n`, reads prev-step `qpos`/
   `site_xpos` + `RolloutState` buffers, updates sticky latches, writes only the ctrl
   rows it owns, maintains `succ` and `cost += done ? 0 : 1`. Reuse the wp.funcs
   (`_advance_base`, `_rate_limited_arm`, `_write_base`).
2. `plan.py`: add its launch args in `Rollout._ctrl_launch` and a `_prep` branch that
   turns the sampled param batch into device targets (upload host params directly;
   call `_solve_arm_goal` if it needs IK). New per-world state -> add to `RolloutState`.
3. Use it: `ActionSpec("<kind>", timeout=...)` + one region of matching dim in a
   scenario. Graph capture, reset, and CE need no changes.

## Adding a scenario

Write `make_<name>(sim)` in `scenarios.py` returning a `Scenario`: scene XML path,
one region per ActionSpec (same order), the spec list, `graspable_qadr`/`graspable_xy`,
optional `nav` tuple (build with `nav.occupancy_from_rects` + `nearest_free`) and
`obstacles`, and a vectorized `goal(sim, qpos (n,nq)) -> (ok (n,), proxy (n,))` where
proxy is a smooth distance-to-goal CE falls back on. Register it in `build()`/
`scene_path()` so replays can rebuild it from the saved name. Read positions from scene
geoms (`sim.geom_xy` etc.) so regions follow the XML.

## Design philosophy

- **Rollout throughput is priority #1.** The hot path is [control kernel, `mjw.step`]
  captured once per action kind as a CUDA graph and replayed per step. Nothing that
  varies per action instance may be a kernel scalar (graphs bake scalars) — it lives in
  device buffers. No host readbacks between action boundaries; one `qpos` readback per
  CE iteration for goal evaluation.
- **Lean, minimal research code.** No abstraction until two call sites need it. Comments
  only for purpose/inputs/outputs and non-obvious constraints (physics tuning limits,
  API traps) — no tuning history or change rationale.
- **Batch-first**: every per-world quantity is a `wp.array` with a leading `nworld` dim;
  control logic is branchless per-world (sticky integer latches), a direct transcription
  of the old JAX `lax.scan` bodies. Host code (CE, sampling, elites, nav gradients,
  plots) is plain numpy.
- **Fidelity to the MJX repo's behavior** over cleverness: same constants, same phase
  logic, same cost (+1/step until action success), same CE tiering. Deviations must be
  deliberate and documented (e.g. no gravcomp, no collision curriculum).

## Traps (cost hours to rediscover)

- Post-compile `body_gravcomp` patches are no-ops in MuJoCo/MJX (`ngravcomp` gate) but
  MJWarp APPLIES them — free objects float. This repo runs without gravcomp on purpose.
- `mjw.put_data` default `njmax` overflows these scenes ("nefc overflow" warning =
  silently degraded contacts). `Rollout` defaults to `njmax=256`; keep headroom.
- dt=0.003 and Newton solver iters 20/10 are at the edge of grasp stability — don't
  raise dt or lower iters without re-running `tests/test_grasp.py`.
- `n` (worlds) is fixed per `Rollout`/`IK` instance — buffers and graphs are sized to it.
- Pylance flags warp annotations like `wp.array2d(dtype=...)` ("call expression in type
  expression") — that's the standard Warp kernel idiom; ignore it.

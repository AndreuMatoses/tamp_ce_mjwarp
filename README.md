# tamp_ce_mjwarp

Cross-entropy (CE) optimization over the continuous parameters of symbolic task-and-motion
plans, scored by massively batched GPU physics rollouts in
[MuJoCo Warp](https://mujoco.readthedocs.io/en/latest/mjwarp/) (MJWarp). Port of
`tamp_ce_jax` (MJX/JAX) to MJWarp's native `nworld` batching + CUDA-graph stepping for much
faster rollouts.

A plan is a fixed sequence of skills (`move`, `pick`, `place`, `push`) whose continuous
parameters (base standoffs, grasp orientations, place poses) are sampled from symbolic
regions. Each CE iteration rolls out all N parameter sets in parallel on the GPU, selects
elites by symbolic goal success + step cost, and refits per-dimension Gaussians.

## Setup

Needs an NVIDIA GPU (Warp/MJWarp are CUDA-only) and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Run

```bash
uv run python examples/pick_place_obstacle.py --variant wall   # or ramp | ramp_forced
uv run python examples/stick_and_box.py
uv run python examples/replay_solution.py solutions/<name>/<name>.h5   # viewer or --video
uv run python examples/replay_batch.py pick_place_obstacle_wall       # superposed batch
uv run python examples/visualize_scenario.py pick_place_obstacle_wall # inspect regions
uv run pytest                                                          # IK + grasp tests
```

For the ramp variant to reliably *discover* the slide solution (place the cube past the
ramp's high edge and let physics carry it to the table) use the multi-mode CE settings —
see `notes/ramp_mode_tuning.md` for the why:

```bash
uv run python examples/pick_place_obstacle.py --variant ramp --iters 14 --n 6144 \
    --n-elite 51 --elite-temp 2 --n-modes 3 --explore 0.15 --robust-k 16
# concatenated per-iteration population video of a finished run:
uv run python examples/replay_batch.py pick_place_obstacle_ramp \
    --from-solution solutions/pick_place_obstacle_ramp/pick_place_obstacle_ramp.h5 \
    --all-iters --video iterations.mp4 --n 32 --speed 4 --camera top
```

Outputs land in `solutions/<scenario>/`: the best solution (HDF5 with per-iteration
distributions, replayable at any iteration), an mp4 replay, per-iteration population
plots, and CE convergence diagnostics.

## Architecture

```
src/tamp_ce_mjwarp/
  sim.py        model load/patch (collision proxy, gravcomp, solver iters) + name->index maps
  actions.py    per-action-kind Warp control kernels (phase latches, base/arm/gripper ctrl)
  plan.py       ActionSpec + Rollout: batched Data, captured CUDA graphs, action boundaries
  ik.py         batched whole-body DLS IK on a minimal kinematic model (one captured graph)
  nav.py        value-iteration nav field: GPU relaxation kernel + host gradient smoothing
  regions.py    symbolic sampling regions (box/annulus/quat/point/point_quat)
  ce.py         the CE loop (numpy): tiered elite selection, per-dim Gaussian refit
  scenarios.py  scenario registry (scene + regions + specs + symbolic goal)
  viz.py        rendering, viewers, population/convergence plots, HDF5 solution I/O
```

One CUDA graph per action kind is captured once ([control kernel, `mjw.step`]) and
replayed for every step; everything that varies per action instance (IK targets, base
velocity caps, nav fields) lives in device buffers. Per pick/place/push action boundary,
a batched Newton-DLS IK (also one captured graph) maps sampled end-effector poses to
whole-body configs entirely on-device.

New to Warp/MJWarp? `notes/warp_tutorial.md` walks through how this code parallelizes,
what CUDA graph capture does, its pitfalls, and which performance levers to tune.

Physics notes: dt=0.003 with Newton solver iterations 20/10 is the largest/cheapest
setting that keeps grasping reliable; mesh geoms never collide (the robot uses a group-4
primitive proxy, excluded from floor contact); `body_gravcomp=1` so the
position-controlled arm doesn't sag.

# How this repo uses Warp + MuJoCo Warp — a tutorial

A walkthrough of the GPU machinery in this codebase for someone new to
[Warp](https://nvidia.github.io/warp/) and [MJWarp](https://mujoco.readthedocs.io/en/latest/mjwarp/).
Everything here is illustrated with real code from `src/tamp_ce_mjwarp/`.

## 1. The mental model

Warp is "CUDA in Python": you write kernels as decorated Python functions, Warp
JIT-compiles them to CUDA, and you launch them over a grid of threads. There is no
autograd-style program transformation like JAX — no `vmap`, no `scan`, no functional
purity. Instead:

- **State is mutable device memory** (`wp.array`). Kernels write into it in place.
- **Batching is explicit**: an array has a leading world dimension and each GPU thread
  handles one world (`i = wp.tid()`). Where JAX had `vmap(fn)`, we have a kernel whose
  thread `i` does what `fn` did for env `i`.
- **Time loops are Python loops** that launch kernels. Where JAX had `lax.scan` over
  `mjx.step`, we have `for _ in range(steps): wp.capture_launch(graph)`.

MJWarp provides the physics as a set of Warp kernels behind functions that mirror the
MuJoCo API (`mjw.step`, `mjw.forward`, `mjw.kinematics`, ...), operating on:

- `mjw.Model` — from `mjw.put_model(mj_model)`. Static. **Collision pairs and option
  scalars are baked here**: editing `geom_contype` on the device later does nothing,
  which is why `Sim` patches the `MjModel` *before* `put_model`.
- `mjw.Data` — from `mjw.put_data(mjm, mjd, nworld=n, ...)` or `mjw.make_data`. Every
  field has a leading `nworld` axis: `d.qpos` is `(nworld, nq)` float32, `d.site_xpos`
  is `(nworld, nsite)` of `wp.vec3`. One `mjw.step(m, d)` advances **all** worlds.

## 2. Writing a kernel (the control kernels)

A skill like `run_pick`'s JAX scan body became `_ctrl_pick` in `actions.py`:

```python
@wp.kernel
def _ctrl_pick(qpos: wp.array2d(dtype=wp.float32), ..., cost: wp.array(dtype=wp.float32)):
    i = wp.tid()                       # this thread's world index
    ee = site_xpos[i, pinch_site]      # read previous step's kinematics
    if at_pre[i] != 0 and wp.length(ee - gw) < GRASP_REACH:
        reached[i] = 1                 # sticky latch: set, never cleared
    ...
    ctrl[i, gripper_act] = wp.where(settled, CLOSED, OPEN)
```

Things to know:

- `wp.tid()` gives the thread index; launch with `wp.launch(kernel, dim=n, inputs=[...])`.
  `dim` can be multi-dimensional: the nav VI kernel uses `dim=(n, H, W)` and unpacks
  `i, y, x = wp.tid()` — one thread per grid cell per world.
- Type annotations on kernel parameters are **required** and are Warp types
  (`wp.array2d(dtype=wp.float32)`, scalar `float`/`int`, `wp.vec3`...). Pylance
  complains about them ("call expression in type expression") — ignore it.
- Module-level Python constants (`MAX_VEL`, `GRIP_HOLD`) referenced inside a kernel are
  **frozen at compile time**. Changing them requires reimporting/recompiling. That's fine
  for physics constants, wrong for anything that varies at runtime (see §4).
- Branches are fine (unlike JAX's `jp.where`-everything), but threads in a warp that
  diverge serialize. Our latch logic is short either way; not a bottleneck here.
- `for j in range(6)` with constant bounds unrolls; dynamic bounds also work (the 6×6
  Cholesky in `ik.py` uses nested triangular loops). Local fixed-size linear algebra
  uses `wp.types.matrix(shape=(6, 9), dtype=wp.float32)` etc., which live in registers.
- Reusable device-side helpers are `@wp.func` (e.g. `_advance_base`, `nav.descent_dir`)
  — they inline into kernels, and can take arrays.

The **one-kernel-per-step** design matters: all per-step control logic for a skill is a
single kernel, so a simulation step is [1 control kernel + `mjw.step`]. Per-world state
that JAX threaded through the scan carry (latches, commanded configs, counters) became
persistent buffers in `RolloutState`, allocated once and reused for the whole run.

## 3. Ordering, synchronization, and the host/device boundary

Kernel launches on the same CUDA stream execute **in order** — the control kernel always
completes before `mjw.step` reads `d.ctrl`. You never need explicit sync for
kernel-after-kernel dependencies.

You *do* pay for host<->device traffic:

- `arr.numpy()` copies device→host and **synchronizes** (blocks until the GPU catches
  up). Fine once per CE iteration (`Rollout.run` reads back `cost`, `n_act`, `qpos` at
  the end); catastrophic if done per step.
- `arr.assign(np_array)` copies host→device. We use it for per-iteration parameter
  uploads and per-action targets — dozens of small copies per action boundary, invisible
  next to thousands of steps.
- The design rule in this repo: **the step loop touches no host memory at all**, and
  action boundaries touch it only O(1) times. Anything needed per step must already be
  in a device buffer.

`wp.synchronize()` forces a full sync — used only before timing measurements and final
readbacks.

## 4. CUDA graph capture — the core throughput trick

A single `mjw.step` is ~50 small kernel launches. Python launch overhead (~10 µs each)
would dominate at our step counts: a plan is ~8300 steps × 50 kernels ≈ 400k launches
per CE iteration. CUDA graphs fix this: record the launch sequence once, then replay it
as **one** GPU operation.

```python
# plan.Rollout._capture, simplified
self._step_once(kind)                # warm-up: compiles every kernel the graph contains
with wp.ScopedCapture() as cap:
    self._step_once(kind)            # recorded, NOT executed
graph = cap.graph
...
for _ in range(steps):
    wp.capture_launch(graph)         # ~5 µs for the whole step
```

Rules that bit us (or would have):

1. **Capture records, it does not run.** State is unchanged by the capture itself.
   Conversely, anything your Python code *computes* during capture (shapes, flags,
   loop counts) is decided then, forever.
2. **Warm up first.** Kernel compilation cannot happen inside a capture. Launch every
   kernel once before capturing (that's why `_capture` runs `_step_once` uncaptured
   first, and why `IK.__init__` runs one Newton iteration before capturing 20).
3. **Scalar kernel arguments are baked into the graph.** A captured
   `wp.launch(k, inputs=[..., 0.4])` replays with `0.4` forever. Anything that varies
   between replays must be a device buffer the kernel reads. This is why
   `RolloutState.base_vel` is a 1-element `wp.array` instead of a float, and why IK mode
   weights (`winv`, `tw`) are buffers written by `assign` before replaying the IK graph.
4. **Buffers are bound by reference.** The graph replays against the exact arrays passed
   at capture. Reallocating a buffer (rather than writing into it) silently decouples it
   from the graph. All rollout buffers are allocated once in `__init__` and only ever
   `assign`ed / `zero_`ed.
5. **No host work inside a capture**: no `.numpy()`, no allocation, no conditional
   Python logic you expect to re-evaluate. If a branch must differ between replays,
   either move it into the kernel (read a flag buffer) or capture separate graphs —
   we capture one graph **per action kind** (4 total) rather than putting a switch
   anywhere.
6. **Graph size**: we capture *one step* and replay it in a Python loop, rather than
   capturing a whole action (~1700 steps ≈ 90k nodes) or plan. Replay overhead is a few
   µs/step (≪1% of runtime); giant graphs are slow to instantiate and freeze the step
   count. If profiling ever shows launch overhead, capture K=32 steps per graph — a
   3-line change.

The same pattern appears three times in the repo: the per-step rollout graphs
(`plan.Rollout`), the 20-iteration IK Newton loop (`ik.IK`, ~120 launches → 1 graph),
and the 220-iteration nav VI relaxation (`nav.NavBatch`).

## 5. What can and cannot be captured / done efficiently

**Great on GPU, captured:** the physics step; branchless-ish per-world control math;
fixed-iteration inner loops (Newton IK, VI relaxation); recording into preallocated
buffers with a device step counter (`_record_probe` + `_tick` — note the counter is
*incremented by a second kernel* so all threads of the recorder see a consistent value).

**Fine on GPU, uncaptured** (launched normally at action boundaries, ~10 launches per
action): `_init_action`, `_prep_*`, `_store_q`. These may take true scalars (`z_off`,
`obj_qadr`) precisely *because* they are not captured.

**Deliberately on the host (numpy):** CE elite selection and Gaussian refit, region
sampling, occupancy grids and the EDT, the nav gradient smoothing, goal evaluation,
all plotting/IO. Rule of thumb: runs once per iteration or per action, operates on
kilobytes → host. Runs per step or per world×step → device.

**Cannot do (vs JAX):** no autodiff through physics, no runtime-shaped arrays (every
buffer's size is fixed at allocation — hence `n` is fixed per `Rollout`/`IK`), no
cheap re-jit for a different batch size; you rebuild the object instead.

## 6. Levers to tune

| Lever | Where | Effect / caution |
|---|---|---|
| `n` (worlds) | examples, `Rollout(n)` | Throughput scales sub-linearly but strongly: n=256 → 0.5M steps/s, n=2048 → 1.34M on the 5090. Bigger CE populations are nearly free; VRAM and IK/nav buffers scale with n. |
| `njmax` / `nconmax` | `Rollout.__init__` → `put_data` | Constraint/contact capacity **per world**. Too small = "nefc overflow" warning and silently wrong contacts (dropped constraints). Default 256 covers the ramp (~155 peak). Oversizing costs memory and some solver time. |
| `dt`, solver `iterations/ls_iterations` | `sim.py` | dt=0.003 + 20/10 is the cheap edge of grasp stability (0.004 or 10 iters → grasps fail). Re-run `tests/test_grasp.py` after touching. |
| Action `timeout`s | `scenarios.py` specs | Cost is linear in total steps; every world always runs the full horizon (no early exit yet). Trimming timeouts is the most direct speedup. |
| Graph granularity | `plan._capture` | Currently 1 step/graph. Chunk K steps per capture if per-step replay overhead ever matters. |
| VI grid `res` / `n_iters` | scenario nav tuple | VI kernel cost ∝ n·H·W·iters; iters must be ≳ grid diameter in cells or distant cells never converge (they render blank in `viz.plot_value_field`). |
| Recording | `Rollout(record_probe/record_qpos)` | Adds 1–2 kernels inside the captured step + big buffers ((T, n, 6) / (T, n, nq)). Keep off for pure optimization runs. |
| Warp kernel cache | `~/.cache/warp/` | First-ever run compiles modules (seconds–minutes); cached afterwards. A Warp/driver upgrade invalidates it — a slow first run after upgrading is normal. |

## 7. Debugging tips

- `wp.config.verbose = True` (before first launch) prints module compiles;
  `wp.ScopedTimer("name")` times GPU work.
- NaNs don't raise on GPU — they propagate. `Rollout.run` marks worlds with non-finite
  `qpos` as failed so CE never selects them; if many worlds go NaN, suspect contact
  buffer overflow or a too-violent sampled contact (float32).
- To inspect any device state mid-run: `d.qpos.numpy()[world_id]` between steps (slow,
  fine for debugging), then feed it to plain-MuJoCo FK/rendering like
  `tests/test_grasp.py` and the trace scripts do.
- Determinism: same inputs on the same GPU replay identically in practice here, but
  MJWarp does not guarantee bitwise determinism across devices/versions — compare
  behavior statistically (success rates), not bitwise.
- If a captured graph seems to "ignore" a change you made, you almost certainly
  reallocated a buffer instead of writing into it (§4 rule 4), or changed a Python
  value that was baked at capture (§4 rules 1/3).

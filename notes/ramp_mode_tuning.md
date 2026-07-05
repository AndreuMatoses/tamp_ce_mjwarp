# Ramp-mode discovery: tuning log & findings

Goal: `pick_place_obstacle_ramp` must *discover* the slide solution (place the cube past
the ramp's high edge, physics slides it onto table2) in addition to the go-around
solutions — with the SAME region definitions as `pick_place_obstacle_wall`, so the
showcase is "same plan + same parameter priors, better solutions when the geometry
allows it". `ramp_forced` (hard region overrides) proves the slide physically works.

## Baselines (pre-tuning saved runs in solutions/)

| run | n | iters | best cost | mode |
|---|---|---|---|---|
| wall | 4096 | 10 | 2985 | go-around west |
| ramp | 256 | 8 | 5249 | go-around west (slide never found) |
| ramp_forced | 256 | 8 | 3021 | slide (place at y=1.78, z=0.71) |

Reference: the original isaac-gym repo (which found the slide mode; the JAX port did
not) used iter-0 populations of ~3000 (decaying to 200), Ne=50, and refit **only from
goal-reachers** — with 0 successes it skipped the refit entirely. Its CE math was
otherwise identical (per-dim Gaussian, no sigma floor, no smoothing). Caveat: its
fabrics controller and Isaac Gym physics differ enough that its success rates don't
transfer — hence the probes below.

## Why the slide mode was never found (probe results)

Feasibility probes (batched grid rollouts anchored on the ramp_forced winner):

1. **The workable slide standoffs were outside the shared move-2 annulus.** The slide
   succeeds robustly from base standoffs **south of y ~ 0.55** (x 1.6-2.4, all the way
   down to y=0) — that is radius >~ 1.9 from table2, while the shared annulus was
   r in [1.2, 1.7]. From in-annulus standoffs (y 0.85-1.25) the place action fails ~100%:
   the whole-body IK puts the base target deep near the ramp, the base drives straight
   in (place has no nav), wedges against the ramp base (~y 1.45-1.7), the EE never
   converges, and the subsequent nav move can't recover from inside the inflated
   occupancy zone. From y~0.7 the arm carries the cube to the exit without a valid
   release. So the annulus outer radius was the binding constraint — not the place box.
2. **The place drop window is decent and partially covered.** With a far-south standoff,
   drops at y in [1.74, 1.90], z in [0.66, 0.84] (x 1.9-2.1) slide in reliably; the old
   place region (y >= 1.825) covered only a sliver of it.
3. **Population size matters exactly as remembered.** With the widened regions below,
   uniform iteration-0 sampling yields **2-3 slide successes per 3072 samples** (seeds
   0/1) — and their cost (~3300) is *cheaper* than the best go-around (~4100) at iter 0,
   so the cost-ranked goal tier naturally promotes them once they exist. At n=256 the
   expected slide count per iteration is ~0.2: the mode statistically never appears, and
   the proxy-fallback tier meanwhile drags the distribution toward go-around.

## Change 1: widened shared regions (scenarios.py)

- move-2 standoff: `annulus(table2, 1.2, 1.7)` -> `annulus(table2, 1.2, 2.05)`
  (adds the slide-workable south band r in [1.9, 2.05] while keeping every go-around
  standoff).
- place: `point(center=(2, 2.225, 0.3), size=(0.2, 0.4, 0.6))` ->
  `point(center=(2, 2.2, 0.3), size=(0.2, 0.5, 0.6))` (y down to 1.7: covers the
  high-edge drop window; still covers all of table2 for the direct place).

Both variants (wall, ramp) share these definitions — no per-variant tuning.

## Experiment log

All runs: n=3072, n_elite=50, 10 iters. "iter-0 best" = mode of the cheapest iter-0
goal-reacher.

| cycle | change | run | iter-0 best | final mode | best cost | goal% last it |
|---|---|---|---|---|---|---|
| 1 | widened regions | ramp s0 | **slide 3307** | west | 2912 | 38 |
| 1 | | ramp s1 | **slide 3281** | west | 2950 | 50 |
| 1 | | ramp s2 | **slide 3116** (still best at it 1) | west | 2904 | 48 |
| 1 | | wall s0 | west 3972 | west | 2989 | 40 |

Cycle-1 verdict: the region widening works — every seed now *discovers* the slide mode
at iteration 0 and it is the cheapest plan found. But the count-weighted refit kills it:
2-3 slide elites are outvoted by ~30 go-around elites, the Gaussian mean migrates west,
and the slide zone is >3 sigma away by iter 2. Cost says slide, density says west —
density wins. This motivates change 2.

| cycle | change | run | iter-0 best | final mode | best cost | goal% last it |
|---|---|---|---|---|---|---|
| 2 | elite_temp=4 | ramp s0 | slide 3307 | west | 2716 | 5 |
| 2 | | ramp s1 | slide 3281 (it 1 too) | west | 2751 | 8 |
| 2 | | ramp s2 | west 4104 (slide lost to GPU nondet.) | west | 2756 | 9 |
| 2 | | wall s0 | west 3972 | west | 2911 | 2 |
| 3 | forced, temp=4 (slide floor) | forced s0 | slide 3125 | **slide** | **2315** | 16 |
| 3 | elite_temp=2 | ramp s0 | slide 3307 | west | 2745 | 14 |
| 3 | | ramp s1 | slide 3281 | **slide** | **2378** | 12 |

Cycle-2/3 verdicts:
- temp=4 exploits harder (west 2716 vs 2904 uniform) but slide still loses the
  iteration-1-2 "knife edge": with the weighted mean between the modes, iter-1 samples
  land in both basins, and go-around's much larger success basin usually produces the
  cheapest iter-1 sample, grabbing rank 0.
- **The slide mode is genuinely ~15% cheaper at convergence: 2315 (forced) vs 2716
  (best west ever).** So converging west is a real optimization failure, not a taste
  question.
- temp=2 flipped seed 1 to slide (2378 on the *plain* ramp variant, vs wall 2911).
  Seed-to-seed stochasticity at n=3072 decides the knife edge; slide has only ~2-3
  successes per early iteration. Note the converged slide standoff (1.46, 0.73)
  approaches from the SW corner — smarter than anything in the hand probes.
- Rank weighting trades final population success% (12-16% vs 38-50%) for best cost;
  the converged mean sits near a performance cliff (the base-wedging boundary).

| cycle | change | run | iter-0 best | final mode | best cost | goal% last it |
|---|---|---|---|---|---|---|
| 4 | temp=2, n=6144 | ramp s0 | slide 3603 | **slide** | **2347** | 6 |
| 4 | | ramp s1 | slide 3883 | **slide** | **2573** | 7 |
| 4 | | ramp s2 | slide 3433 | **slide** | **2335** | 1 |
| 4 | | wall s0 | west 3963 | west | 2892 | 1 |

Cycle-4 verdict: all three seeds converged slide (2335-2573, up to ~19% cheaper than
wall's 2892 with identical definitions) — but see cycle 5: this did not replicate.

| cycle | change | run | iter-0 best | final mode | best cost | goal% last it |
|---|---|---|---|---|---|---|
| 5 | + GPU nav, exit_frac=0.98 | ramp s2 | west 3705 | west | 2853 | 29 |
| 5 | | wall s0 | west 3963 | west | 2895 | 2 |
| 5b | + GPU nav only (exit_frac=1) | ramp s2 | west 3705 | west | 2803 | 15 |
| 6 | + smooth=0.5, 12 iters | ramp s0 | west 3968 | west | 2750 | 13 |
| 6 | | ramp s1 | slide 3883 | **slide** | **2408** | 17 |
| 6 | | ramp s2 | west 3705 | west | 2714 | 18 |

Cycle-5/6 verdicts — two hard lessons:
- **The single-Gaussian outcome is a replication coin flip, not a seed property.** The
  same seed 2 converged slide in cycle 4 and west in cycles 5/5b: GPU contact-solver
  nondeterminism flips the handful of marginal iter-0 slide successes, and whichever
  mode's cheapest sample wins rank 0 in iters 0-2 takes the whole run. Iter-0 cost
  distributions of the two modes overlap (slide ~3100-3900, west ~3700-4400); slide's
  real advantage only appears after refinement (2315 vs 2716 converged).
- exit_frac=0.98 is worse than a coin flip: the slide worlds are among the *slowest*
  placers, so the quantile exit systematically cuts exactly them (see the throughput
  section — discovery runs must keep exit_frac=1).
- Smoothing (0.5) only slows the collapse; the rank-0 winner still decides (1/3 slide).

## Change 3: multi-mode CE (`ce.run(..., n_modes=k)`)

The structural fix: cluster the elites each iteration (k-means on the standardized
concatenated plan params, farthest-point init anchored on the cheapest elite), maintain
one per-region Gaussian per cluster, and split the population evenly between clusters.
Competing modes then refine *independently* to their converged costs — the returned
best is whichever mode genuinely wins, instead of whichever won the first noisy rank-0
race. Cluster 0 always contains the current best elite; the per-iteration dist saved to
HDF5 (used by `--all-iters` replays) is cluster 0's. `n_modes=1` (default) is exactly
the original single-Gaussian path.

Cycle 7 (n_modes=2 alone): 1/3 slide. Two residual failure modes surfaced:
(a) iteration-0 slide successes are themselves a nondeterministic lottery — when a
replication produces none, there is nothing to cluster; (b) when the lone slide elite
IS captured, the singleton refit used a fixed sigma=0.1 (tiny vs the region scale), so
its half of the population orbited one possibly-fluke sample, produced nothing, and the
global top-50-by-cost cut then expelled slide as west flooded in.

## Change 4: explore share + per-cluster quotas + prior-scaled sigma floors

- `ce.run(..., explore=f)`: reserve a fraction of every population for fresh
  uniform-prior samples — rare-mode discovery gets a chance every iteration instead of
  only at iteration 0. Cycle 8 (n_modes=2 + explore=0.15): still 1/4 slide — discovery
  wasn't the bottleneck anymore; (b) above was.
- With `n_modes>1`, elites now come from a widened pool (top 4*n_elite by cost),
  clustered, with a per-cluster quota of n_elite/k — a rare mode's elites cannot be
  expelled by a flood of cheaper-for-now rivals. Per-mode stds are floored at 5% of the
  region's prior spread (25% for singleton clusters) so a mode anchored on one lucky
  sample searches its whole basin rather than orbiting it.

Cycle 9 (quotas + floors): 2/4 slide including the two best costs yet (2300, 2304),
but the seed-0 replication pair still split. Residual cause: the cluster pool was
top-4*n_elite *by cost* — once the incumbent mode converges below ~3000, a freshly
discovered slide sample (~3400 raw) no longer makes the pool, so the cost cut expels
it before clustering sees it. Fix: with any goal-reachers, the pool is now ALL
goal-reachers (cost-ordered, capped at 4096 for k-means cost); quotas select within
clusters.

Cycle 10 (full pool): 1/4 — seed 2 showcased the intended late-discovery path
(explore share found slide at iter 3, cluster captured it, converged 2324), but seeds
with an iter-0 slide anchor STILL lost it. Post-mortem via the saved iter-0 dist: the
k-means had **blended** the slide anchor into a mixed cluster (saved cluster-0 mean
p2=(0.64, 2.01), std~1 — neither mode). With ~10 of 13 standardized dims mode-
irrelevant (quats, exit box, ...), a typical go-around row sits ~5.1 sigma from its own
kind but only ~6.9 sigma from the slide row; farthest-point init (anchor vs an
*outlier* of the majority mass) puts half the majority closer to the rare anchor, and
Lloyd blends. Fix: init the second center at the pool *medoid* (center of the majority
mass, ~3.6 sigma from its members) — every majority row is then decisively closer to
it than to the rare anchor. Synthetic check (63 majors + 1-5 rares, 13 dims, 3-sigma
offset in 3 dims): the rare anchor is never lost; occasional mild contamination only.

Generality check (stick_and_box, single-mode scenario): defaults 346 best cost vs
flags (n_modes=2, temp=2, explore=0.15) **281** — the flags improved it (rank
weighting exploits harder); no regression. Iter time rose 2.1->4.8s because the
diverse explore share keeps some worlds failing, so the all-worlds early exit fires
less — inherent cost of exploration.

Cycle 11 (medoid init, k=2): replications now agree — but 1/4 slide. An instrumented
run showed the last gap: iteration-0 slide successes are ~Poisson(1) (this replication
had ZERO in 6144 uniform samples), and when the explore share rediscovered slide at
iteration 6-7, both k=2 clusters had already settled into two *west sub-basins*; the
lone slide row (raw cost 4088 vs their ~2800 top-25 cut) had no cluster to claim and
was silently absorbed. **Mode capture needs a center that lands on new outliers.**

Fix (no new mechanism, just k): with `n_modes=3` the cluster init is
[cheapest elite, pool medoid, farthest point from both] — the third center lands on
the most outlying goal-reacher every iteration, which is exactly what a fresh rare-
mode discovery is. It claims its own cluster the moment it appears, the singleton
floor widens its search, and it refines with a third of the population. k=3 also
matches this scenario's true mode count (around-west / around-east / slide).

## Final settings (accepted, cycle 12)

| run | final mode | best cost | goal% last it |
|---|---|---|---|
| ramp seed 0 | **slide** | **2398** | 30 |
| ramp seed 0 (replication) | **slide** | **2388** | 27 |
| ramp seed 1 (captured at iter 4) | **slide** | **2386** | 5 |
| ramp seed 2 | west | 2738 | 29 |
| wall seed 0 | west | 2938 | 17 |

- Shared regions as in Change 1 (scenarios.py) — permanent, identical for wall/ramp.
- CE flags: `--n-modes 3 --elite-temp 2 --explore 0.15 --n 6144 --n-elite 51
  --iters 14`. All default-off in `ce.run` — the plain call is still the original
  JAX-faithful solver.
- The seed-0 replication pair now agrees (2398/2388, converged params matching to
  ~0.01) — the mechanism, not luck: discovery can happen at any iteration (explore
  share), capture is guaranteed when it does (the farthest-point third center lands on
  the newest outlier), survival is guaranteed (per-cluster quota, no global cost cut),
  refinement is guaranteed (sigma floor + a third of the population).
- Residual stochasticity: ~3/4 of runs converge slide (seed 2 stayed west at 2738 —
  its explore discoveries never took); when slide is found it lands at ~2390 vs west
  ~2740 and wall ~2940, so the best-of-2-seeds protocol is decisive in practice.
- Reproduce:
  ```
  uv run python examples/pick_place_obstacle.py --variant ramp --iters 14 --n 6144 \
      --n-elite 51 --elite-temp 2 --n-modes 3 --explore 0.15 --seed 0 --tag demo
  uv run python examples/pick_place_obstacle.py --variant wall --iters 14 --n 6144 \
      --n-elite 51 --elite-temp 2 --n-modes 3 --explore 0.15 --seed 0 --tag demo
  # per-iteration population video (segment 0 = uniform prior, then each refit):
  uv run python examples/replay_batch.py pick_place_obstacle_ramp \
      --from-solution solutions/pick_place_obstacle_ramp_demo/pick_place_obstacle_ramp.h5 \
      --all-iters --video iterations_batch.mp4 --n 32 --speed 4 --camera top
  ```
- GPU nondeterminism in the contact solver means fixed seeds are not bit-identical
  across runs; marginal worlds flip. The multi-mode machinery exists precisely so the
  outcome no longer hinges on those flips.

## Change 2: cost-rank-weighted refit (`ce.run(..., elite_temp=t)`)

Elites are already selected best-first; weight them `w_i ~ exp(-rank_i / t)` in the
mean/std refit instead of uniformly. This pulls the fitted Gaussian toward the
best-scoring *mode* rather than the most-populated one, exactly the CE property this
scenario needs (rare-but-cheaper mode must not be outvoted). `elite_temp=None` (default)
keeps the original JAX/isaac-gym uniform-weight behavior. With t=4 over 50 elites the
top ~6 ranks carry most of the weight.

## Deviations from the JAX/isaac-gym behavior

- Population held constant across iterations (isaac-gym decayed 3000 -> 200). Rollout
  buffers/graphs are sized to a fixed n and large batches are nearly free on GPU.

## Other observations / method improvement ideas

- The plain per-dim Gaussian refit is unimodal: it can represent "slide vs go-around"
  only until the first refit. Whichever mode dominates the elite set captures the mean.
  A mixture / clustered-elite refit would preserve modes for longer.
- The proxy-fallback elite tier (no goal-reachers -> most-actions + proxy) actively
  works against rare modes: partial slide attempts score fewer completed actions than
  go-around attempts, so with 0 successes the refit steers away from the slide zone.
  The isaac-gym behavior (skip refit until successes exist) is gentler for multimodal
  problems; worth a `ce.run` option if collapse reappears.
- `regions.sample_normal` clips to mean+-3 sigma, not to the region bounds — converged
  solutions can (and do) drift outside their prior region.
- The place action drives the base straight at the IK base target with no collision
  awareness; that is what wedges the base against the ramp from mid-range standoffs.
  Clipping the base target to nav-free space would widen the slide-workable standoff
  band considerably.
- Pure cost-rank greed (elite_temp) converges onto performance *cliff edges* (the
  cheapest plans sit right at the wedging/timeout boundary), so the final population
  success rate drops to a few percent even as best cost improves. If robustness of the
  converged distribution matters, rank by a cost quantile under rollout noise (e.g.
  duplicate each elite candidate a few times) or add a small std floor — not needed for
  the showcase.
- GPU contact-solver nondeterminism flips marginal worlds between runs of the same
  seed; single-sample cost rankings near a cliff inherit that noise.

## Change 5: robust solutions (`ce.run(..., robust_k=k)`)

Measured problem: the raw-cost-greedy best plans are lottery tickets. Replaying the
cycle-12 winners 256x **exactly** (GPU contact nondeterminism is the only variance):
0-14% success; under 2%-of-prior-std parameter jitter: 1-8%. Cost-greedy CE converges
onto cliff edges and the argmin rollout is the luckiest sample on the cliff.

Two mechanisms, both free of extra rollouts during the loop:
1. **Neighbourhood-smoothed elite ranking**: rank goal-reachers by the mean cost of
   their k nearest sampled neighbours (standardized param space, failed neighbours
   counted at the full-horizon cost). The population IS the perturbation sample — a
   cliff-edge plan is surrounded by failures and scores badly even when its own
   rollout got lucky. Result: final-population goal rates jumped from 1-30% to 45-73%.
2. **Mean-plan validation pass** (one extra batched rollout at the end): the returned
   best had residual winner's curse (argmin over thousands of noisy k-NN scores picks
   lucky pockets: 18-27% replica success). The robust representative of a mode is its
   converged *mean*: measured 88% exact / 69% jittered for the slide mode at
   essentially the same cost (2514 vs 2495). With robust_k on, each mode's mean plan
   is tiled across one final batch, real success rates are measured, and the best mean
   by expected cost (rate*cost + (1-rate)*fail_cost) is returned; `robust_rate` in the
   HDF5 meta records it.

Honest trade-off surfaced by the robust objective: one intermediate robust seed
converged go-around-east at 98% replica success instead of the slide — under expected
cost with a full-horizon failure penalty that preference can be *correct* when only a
flaky slide pocket was found. With the validation pass in place this resolved itself:
the slide basin's core is genuinely robust.

**Final validated results (robust_k=16 + validation pass, n=6144, k=3 modes,
temp=2, explore=0.15, 14 iters)** — independent 256-replica re-measurement of the
saved best plans:

| run | mode | cost | exact-replica ok | 2%-jitter ok |
|---|---|---|---|---|
| ramp seed 0 | slide (conservative far-east standoff) | 3736 | 96% | 93% |
| ramp seed 1 | **slide** | **2494** | **96%** | **90%** |
| wall seed 0 | west | 3116 | 99% | 86% |

vs the pre-robust cycle-12 winners: 0-14% exact, 1-8% jittered. The robust slide
(2494) is still ~20% cheaper than the robust go-around (3116) — the showcase survives
the robustness requirement.

Recommended full command:
```
uv run python examples/pick_place_obstacle.py --variant ramp --iters 14 --n 6144 \
    --n-elite 51 --elite-temp 2 --n-modes 3 --explore 0.15 --robust-k 16
```

## Rollout throughput tuning

Profile of one CE iteration, n=3072, uniform population (no early exits fire), before
tuning: **13.65s total = 10.8s stepping (79%) + 2.9s nav (21%)**; IK 2ms, readbacks ~0.
Inside `nav.compute` (0.95s per move action) the 220 VI sweeps took **17ms** — the other
~0.93s was downloading the value fields, computing the descent gradient on the host,
and uploading it back. VI sweeps were bit-converged already at 100 (max dV vs 400
sweeps = 0), so sweep count was never the problem.

Changes:
1. **GPU `value_grad`** (nav.py): the fill/central-difference/3x-box-blur pipeline
   ported to three Warp kernels and appended to the captured VI graph; `compute` is now
   a single graph launch with no host round-trip. Verified bit-faithful vs the host
   reference (max diff 1.8e-7). `nav.compute`: 950ms -> **18ms**; CE iteration:
   13.65s -> 10.84s (**1.87 -> 2.35M world-steps/s**, +26%). The host `value_grad`
   stays as reference/plotting code.
2. **`Rollout(exit_frac=)`** (plan.py, opt-in, default 1.0 = exact old semantics): end
   an action once this fraction of worlds has succeeded. The old all-worlds check never
   fires at large n (one straggler in 6144 forces the full horizon). At 0.98 iteration
   time dropped to ~7.1s (n=6144) — but the parity run exposed a real quality trap:
   **the quantile exit selectively kills rare slow modes at discovery time.** The
   slide-capable worlds are among the slowest placers (long base drive + reach-up), so
   cutting the slowest 2% erased the slide mode from iteration 0 and seed 2 converged
   go-around (2853). **Do not use exit_frac < 1 for discovery runs**; it is safe for
   replays/converged sweeps where the mode is already locked in.

Not touched: dt/solver iters (grasp-stability trap), njmax (contact-degradation trap,
no measurable gain), EXIT_CHECK sync cadence (~13ms/iteration, noise), VI sweep count
(free after the GPU port; 220 keeps its safety margin).

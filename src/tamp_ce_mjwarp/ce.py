"""Cross-entropy optimization over symbolic plan parameters.

Iter 0 samples uniformly from each region; later iters refit an independent per-dim
Gaussian (mean, std) from the `n_elite` elite samples, then sample normal. Elites are
chosen by a tiered rule keyed on the scenario's symbolic goal: goal-reachers ranked by
step cost if any exist, else the most-completed-actions then the smallest proxy.

The batched rollout (captured CUDA graphs) is built once and reused every iteration, so
the sample count `n` is fixed across iterations (buffers/graphs are sized for it). The
robot collision proxy is always active — MJWarp rollouts are cheap enough that the old
off-then-on collision curriculum is unnecessary.
"""

from __future__ import annotations

import time

import numpy as np
from tqdm import tqdm

from tamp_ce_mjwarp import regions
from tamp_ce_mjwarp.log import log
from tamp_ce_mjwarp.plan import Rollout


def _select_elites(goal_ok, n_act, proxy, cost, n_elite):
    """Indices of the `n_elite` best plans, tiered: goal-reachers ranked by step cost; if
    none, the most-completed-actions then the smallest proxy (closest to feasible)."""
    if goal_ok.any():
        pool = np.where(goal_ok)[0]
        return pool[np.argsort(cost[pool])[:n_elite]]
    return np.lexsort((proxy, -n_act))[:n_elite]  # last key is primary: max actions


def _robust_cost(params, goal_ok, cost, k, fail_cost):
    """Neighborhood-smoothed cost per world: the mean over its k nearest sampled
    neighbours (standardized concatenated params, self included) of their cost, with
    failed neighbours counted at `fail_cost`. A plan whose neighbourhood is full of
    failures — a cliff edge — scores badly even if its own rollout got lucky, so
    ranking elites by this steers CE toward basins that survive perturbation. Free:
    the population IS the perturbation sample."""
    X = np.concatenate([np.asarray(p, np.float32) for p in params], axis=1)
    X = X / (X.std(0) + 1e-9)
    eff = np.where(goal_ok, cost, fail_cost).astype(np.float32)
    sq = (X ** 2).sum(1)
    out = np.empty(len(X), np.float32)
    step = max(1, 2 ** 24 // len(X))  # chunk the (n, n) distance matrix
    for i in range(0, len(X), step):
        d = sq[i:i + step, None] + sq[None] - 2.0 * (X[i:i + step] @ X.T)
        nb = np.argpartition(d, k, axis=1)[:, :k + 1]
        out[i:i + step] = eff[nb].mean(1)
    return out


def _refit_idx(params_a, idx, w=None, floor=None):
    """Per-dim Gaussian (mean, std) over the elite rows `idx` of params_a (n, d).
    `w` (sums to 1, same order as idx) weights the elites; None = uniform. `floor`
    (per-dim) keeps std from collapsing below it — a singleton/near-singleton cluster
    otherwise orbits one possibly-fluke sample instead of searching its basin."""
    elite = np.asarray(params_a)[idx]
    if len(idx) == 1:
        std = elite[0] * 0 + 0.1 if floor is None else 5.0 * floor
        return elite[0], std
    if w is None:
        mean, std = elite.mean(0), elite.std(0)
    else:
        mean = (w[:, None] * elite).sum(0)
        std = np.sqrt((w[:, None] * (elite - mean) ** 2).sum(0))
    return mean, (std if floor is None else np.maximum(std, floor))


def _cluster_elites(elite_mat, k):
    """Labels (m,) from k-means on the standardized elite rows (rows are cost-ordered
    best-first). Init: the cheapest elite + the pool medoid (+ farthest points for
    k > 2). The medoid anchors the majority mass — a farthest-point second center is an
    outlier of that mass, and with many mode-irrelevant dims half the mass then sits
    closer to a rare-mode anchor than to it, blending the modes (cost hours to find)."""
    X = elite_mat / (elite_mat.std(0) + 1e-9)
    medoid = X[((X - np.median(X, axis=0)) ** 2).sum(1).argmin()]
    C = [X[0]] if np.allclose(medoid, X[0]) else [X[0], medoid]
    while len(C) < k:
        d = np.min([((X - c) ** 2).sum(1) for c in C], axis=0)
        C.append(X[d.argmax()])
    C = np.stack(C[:k])
    for _ in range(8):
        lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    # relabel so cluster ids follow the best (lowest-rank) elite they contain
    order = np.argsort([lab.tolist().index(j) if (lab == j).any() else len(lab)
                        for j in range(k)])
    remap = np.empty(k, int)
    remap[order] = np.arange(k)
    return remap[lab]


def sample_population(scenario, dists, n, seed=0):
    """Draw a batch of `n` plan parameter sets from the (fitted) per-region dists, falling
    back to the region's uniform prior where a dist is None."""
    out = []
    for a, (r, dist) in enumerate(zip(scenario.regions, dists)):
        rng = np.random.default_rng([seed, a])
        out.append(regions.sample_uniform(r, rng, n) if dist is None
                   else regions.sample_normal(r, rng, n, dist[0], dist[1]))
    return out


def run(sim, ik, scenario, n_iters=8, n=512, n_elite=30, seed=0, verbose=True,
        record_paths=False, on_iter=None, elite_temp=None, smooth=None, n_modes=1,
        explore=0.0, robust_k=None, exit_frac=1.0):
    """Cross-entropy over the scenario's plan parameters.

    `elite_temp`: cost-rank-weight the refit (w ~ exp(-rank/temp), rank 0 = cheapest
    elite). Pulls the fitted Gaussian toward the best-scoring solution *mode* rather
    than the most-populated one — without it, a rare-but-cheaper mode (e.g. the ramp
    slide) is outvoted by a common one in the elite mean and collapses. None = uniform
    weights (the original JAX/isaac-gym behavior).

    `smooth`: retain this fraction of the previous (mean, std) in each refit (classic
    CE smoothing). Delays commitment for a few iterations so competing solution modes
    keep being sampled while their refined costs separate — the early rank-0 winner is
    otherwise a rollout-noise coin flip. None = no smoothing (original behavior).

    `n_modes`: k > 1 clusters a widened elite pool (top 4*n_elite) each iteration
    (k-means on the concatenated plan parameters) and keeps the top n_elite/k per
    cluster as that mode's elites, one Gaussian per cluster, splitting the population
    evenly between them. Competing solution modes then refine *independently* to their
    true converged costs — a rare mode can neither be outvoted in a unimodal fit nor
    expelled from the elite set by a flood of cheaper-for-now rivals. Per-mode stds are
    floored at 5% of the region's prior spread (25% for a singleton cluster) so a mode
    anchored on one lucky sample searches its whole basin. Cluster 0 always holds the
    current best elite; the per-iter HDF5 dist (and so `--all-iters` replays) records
    cluster 0. 1 = original single-Gaussian behavior (no pool widening, no floor).

    `explore`: reserve this fraction of every iteration's population for fresh
    uniform-prior samples. Rare-mode discovery is otherwise an iteration-0 lottery
    (a mode whose success probability is ~1e-3/sample either lands in the first elite
    set or is never sampled again); with an explore share every iteration re-rolls it,
    and n_modes can then capture a late discovery. 0 = original behavior.

    `robust_k`: rank goal-reachers (elite selection AND best-plan choice) by their
    k-NN neighbourhood-smoothed cost instead of their single rollout's cost (failed
    neighbours counted at the full-horizon cost; see _robust_cost). Raw-cost greed
    converges onto cliff edges — plans whose lucky rollout succeeded but whose
    neighbourhood mostly fails; this steers toward basins that survive perturbation.
    Also runs a final validation pass: each mode's converged *mean* plan is tiled
    across one extra batch, and the returned best becomes the mean plan with the
    lowest expected cost — `best["robust_rate"]` / meta `robust_rate` record its
    measured replica success rate. None = original single-sample ranking.

    Returns (best, mdists, pop) with `mdists` the per-mode fitted distributions
    (length `n_modes`). `best` carries {cost, params, meta, iters} where meta
    records the run config + per-iter timing/goal-success and `iters` is the per-iteration
    best plan + refitted distribution, for reproducible HDF5 solutions. If `record_paths`,
    every iter records the batch's base/EE/block tracks and `on_iter(it, pop)` is called;
    `pop` is the last iter's {cost, ok, n_act, proxy, tracks, best_idx}, else None."""
    regs = scenario.regions
    n_actions = len(scenario.specs)

    rollout = Rollout(sim, ik, scenario, n, record_probe=record_paths,
                      exit_frac=exit_frac)

    n_exp = int(round(n * explore))
    n_fit = n - n_exp
    shares = [n_fit // n_modes + (1 if c < n_fit % n_modes else 0) for c in range(n_modes)]
    if n_modes > 1:  # per-region std floors from the priors' spread (see docstring)
        floors = [0.05 * regions.sample_uniform(r, np.random.default_rng([seed, 77, a]),
                                                2048).std(0)
                  for a, r in enumerate(regs)]
    else:
        floors = [None] * len(regs)

    def draw(mdists, it):
        """Sample each mode's share from its dist (None = uniform prior) plus the
        explore share from the priors, concatenated in a fixed block order (consistent
        across actions, so rows stay aligned)."""
        out = []
        for a, r in enumerate(regs):
            cols = []
            for c, dists_c in enumerate(mdists):
                key = [seed, it, a] if n_modes == 1 else [seed, it, a, c]
                rng = np.random.default_rng(key)
                d = None if dists_c is None else dists_c[a]
                cols.append(regions.sample_uniform(r, rng, shares[c]) if d is None
                            else regions.sample_normal(r, rng, shares[c], d[0], d[1]))
            if n_exp:
                rng = np.random.default_rng([seed, it, a, n_modes])
                cols.append(regions.sample_uniform(r, rng, n_exp))
            out.append(np.concatenate(cols, axis=0))
        return out

    def _pop(out, tracks):
        cost, gok, n_act, proxy = out["cost"], out["ok"], out["n_act"], out["proxy"]
        bi = (int(np.where(gok)[0][cost[gok].argmin()]) if gok.any()
              else int(np.lexsort((proxy, -n_act))[0]))
        return {"cost": cost, "ok": gok, "n_act": n_act, "proxy": proxy,
                "tracks": tracks, "best_idx": bi}

    mdists = [None] * n_modes  # one per-region (mean, std) list per mode; None = prior
    best = {"cost": np.inf, "params": None}
    effort = {"proxy": np.inf, "cost": np.inf, "params": None}  # best non-goal env
    pop = None
    iter_times, success_hist, cost_hist, iters_log = [], [], [], []

    if verbose:
        log(f"Starting CE: {n_iters} iters x {n} samples")
    bar = tqdm(range(n_iters), disable=not verbose, desc="CE")
    for it in bar:
        params = draw(mdists, it)

        t0 = time.perf_counter()
        out = rollout.run(params)
        iter_times.append(time.perf_counter() - t0)

        cost, n_act = out["cost"], out["n_act"]
        gok, proxy = out["ok"], out["proxy"]
        # sel is the ranking metric: raw cost, or the neighbourhood-smoothed one
        sel = (_robust_cost(params, gok, cost, robust_k, float(sum(rollout.steps)))
               if robust_k else cost)
        tracks = rollout.probe.numpy()[:rollout.steps_run] if record_paths else None

        if record_paths:
            pop = _pop(out, tracks)
            if on_iter is not None:
                on_iter(it, pop)

        n_goal = int(gok.sum())
        best_c = cost[gok].min() if n_goal else np.inf
        success_hist.append(n_goal)
        cost_hist.append(float(best_c) if n_goal else None)
        # this iter's best env (goal-reacher by sel, else closest-to-feasible), recorded
        # so a saved solution can be replayed at any iteration
        bi_iter = (int(np.where(gok)[0][sel[gok].argmin()]) if n_goal
                   else int(np.lexsort((proxy, -n_act))[0]))
        if verbose:
            bar.set_postfix(goal=f"{n_goal}/{n}", acts=f"{int(n_act.max())}/{n_actions}",
                            best=f"{min(best_c, best['cost']):.0f}")

        # Always refit (even with zero goal success): the tiered fallback pulls the
        # population toward feasibility via action-count + proxy.
        def rank_w(m):
            if not elite_temp:
                return None
            w = np.exp(-np.arange(m) / float(elite_temp))
            return w / w.sum()

        prev = mdists
        if n_modes == 1:
            groups = [_select_elites(gok, n_act, proxy, sel, min(n_elite, n))]
        else:
            # cluster over ALL goal-reachers (cost-ordered; capped for k-means cost):
            # any cost cut here would expel a freshly discovered rare mode whose raw
            # cost trails the incumbent's refined one. Per-cluster quotas select after.
            pool_n = int(gok.sum()) if gok.any() else 4 * n_elite
            pool = _select_elites(gok, n_act, proxy, sel, min(pool_n, 4096))
            elite_mat = np.concatenate([np.asarray(params[a])[pool]
                                        for a in range(len(regs))], axis=1)
            lab = _cluster_elites(elite_mat, n_modes)
            groups = [pool[lab == c][:max(n_elite // n_modes, 2)]
                      for c in range(n_modes)]
        mdists = []
        for c, sub in enumerate(groups):
            if len(sub) == 0:  # empty cluster: keep sampling its previous dist
                mdists.append(prev[c])
                continue
            dc = [_refit_idx(params[a], sub, rank_w(len(sub)), floors[a])
                  for a in range(len(regs))]
            if smooth and prev[c] is not None:
                dc = [((1 - smooth) * m + smooth * pm, (1 - smooth) * s + smooth * ps)
                      for (m, s), (pm, ps) in zip(dc, prev[c])]
            mdists.append(dc)
        dists = mdists[0]  # cluster 0 holds the best elite; logged/saved below

        iters_log.append({
            "cost": float(best_c) if n_goal else float("nan"),
            "n_goal": n_goal,
            "params": [np.asarray(p[bi_iter]) for p in params],
            "mean": [m for (m, _) in dists],
            "std": [s for (_, s) in dists],
            "elite_cost": np.asarray(cost[np.concatenate(groups)]),
            "elite_by_goal": bool(n_goal > 0),
        })

        if n_goal:
            bi = np.where(gok)[0][np.argmin(sel[gok])]
            if sel[bi] < best.get("_sel", np.inf):
                best = {"cost": float(cost[bi]), "_sel": float(sel[bi]),
                        "params": [np.asarray(p[bi]) for p in params]}
        # track the closest-to-feasible env too, so there's always something to replay
        fb = int(np.lexsort((proxy, -n_act))[0])
        if proxy[fb] < effort["proxy"]:
            effort = {"proxy": float(proxy[fb]), "cost": float(cost[fb]),
                      "params": [np.asarray(p[fb]) for p in params]}

    best["reached_goal"] = best["params"] is not None
    if not best["reached_goal"]:  # best-effort fallback so callers can still replay
        best["cost"], best["params"] = effort["cost"], effort["params"]

    if robust_k and best["reached_goal"]:
        # Validation pass: any single winning rollout is a lottery ticket near a cliff
        # (argmin over noisy scores = winner's curse). The converged *mean* plan of a
        # mode is the robust representative — tile each mode's mean across the batch,
        # measure real success rates, return the best mean by expected cost.
        means = [d for d in mdists if d is not None]
        reps = n // len(means)
        fail_cost = float(sum(rollout.steps))
        batch = []
        for a, r in enumerate(regs):
            cols = []
            for c, d in enumerate(means):
                m = np.asarray(d[a][0], np.float32).copy()
                if r.kind == "quat":
                    m /= np.linalg.norm(m) + 1e-9
                elif r.kind == "point_quat":
                    m[3:7] /= np.linalg.norm(m[3:7]) + 1e-9
                cnt = n - reps * (len(means) - 1) if c == len(means) - 1 else reps
                cols.append(np.repeat(m[None], cnt, axis=0))
            batch.append(np.concatenate(cols))
        vout = rollout.run(batch)
        scores, stats = [], []
        for c in range(len(means)):
            lo, hi = c * reps, (c + 1) * reps if c < len(means) - 1 else n
            okc = vout["ok"][lo:hi]
            rate = float(okc.mean())
            mc = float(vout["cost"][lo:hi][okc].mean()) if okc.any() else fail_cost
            scores.append(rate * mc + (1.0 - rate) * fail_cost)
            stats.append((rate, mc))
        c = int(np.argmin(scores))
        rate, mc = stats[c]
        if rate > 0:
            best["cost"] = mc
            best["params"] = [np.asarray(batch[a][c * reps]) for a in range(len(regs))]
            best["robust_rate"] = rate
        if verbose:
            log(f"Robust validation: mode {c} mean plan reaches the goal "
                f"{100 * rate:.0f}% of replicas, mean cost {mc:.0f}")

    best["iters"] = iters_log
    best["meta"] = {
        "scenario": scenario.name, "device": "warp", "reached_goal": best["reached_goal"],
        "n": n, "n_elite": n_elite, "n_iters": n_iters, "seed": seed, "n_actions": n_actions,
        "elite_temp": 0.0 if elite_temp is None else float(elite_temp),
        "smooth": 0.0 if smooth is None else float(smooth), "n_modes": n_modes,
        "explore": float(explore), "robust_k": 0 if robust_k is None else int(robust_k),
        "robust_rate": float(best.get("robust_rate", -1.0)),
        "param_names": [r.name for r in regs], "param_kinds": [r.kind for r in regs],
        "param_dims": [regions.dim(r) for r in regs],
        "iter_times_s": iter_times, "goal_success_per_iter": success_hist,
        "best_cost_per_iter": cost_hist,
    }

    if verbose:
        if not best["reached_goal"]:
            log(f"CE finished — NO plan reached the symbolic goal over {n_iters} iters "
                f"(best effort last iter: {int(n_act.max())}/{n_actions} actions, "
                f"closest proxy {effort['proxy']:.2f}). Try more iters/samples.")
        else:
            log(f"CE finished. best goal cost {best['cost']:.0f} over {n_iters} iters, "
                f"{np.mean(iter_times):.1f}s/iter "
                f"(last iter: {n_goal}/{n} = {100 * n_goal / n:.0f}% reached goal)")

    return best, mdists, pop

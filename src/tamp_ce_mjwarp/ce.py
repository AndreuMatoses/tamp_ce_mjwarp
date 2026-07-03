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


def _refit_idx(params_a, idx):
    """Per-dim Gaussian (mean, std) over the elite rows `idx` of params_a (n, d)."""
    elite = np.asarray(params_a)[idx]
    mean = elite.mean(0)
    std = elite.std(0) if len(idx) > 1 else mean * 0 + 0.1
    return mean, std


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
        record_paths=False, on_iter=None):
    """Cross-entropy over the scenario's plan parameters.

    Returns (best, dists, pop). `best` carries {cost, params, meta, iters} where meta
    records the run config + per-iter timing/goal-success and `iters` is the per-iteration
    best plan + refitted distribution, for reproducible HDF5 solutions. If `record_paths`,
    every iter records the batch's base/EE/block tracks and `on_iter(it, pop)` is called;
    `pop` is the last iter's {cost, ok, n_act, proxy, tracks, best_idx}, else None."""
    regs = scenario.regions
    n_actions = len(scenario.specs)

    rollout = Rollout(sim, ik, scenario, n, record_probe=record_paths)

    def draw(dists, it):
        out = []
        for a, (r, dist) in enumerate(zip(regs, dists)):
            rng = np.random.default_rng([seed, it, a])
            out.append(regions.sample_uniform(r, rng, n) if dist is None
                       else regions.sample_normal(r, rng, n, dist[0], dist[1]))
        return out

    def _pop(out, tracks):
        cost, gok, n_act, proxy = out["cost"], out["ok"], out["n_act"], out["proxy"]
        bi = (int(np.where(gok)[0][cost[gok].argmin()]) if gok.any()
              else int(np.lexsort((proxy, -n_act))[0]))
        return {"cost": cost, "ok": gok, "n_act": n_act, "proxy": proxy,
                "tracks": tracks, "best_idx": bi}

    dists = [None] * len(regs)
    best = {"cost": np.inf, "params": None}
    effort = {"proxy": np.inf, "cost": np.inf, "params": None}  # best non-goal env
    pop = None
    iter_times, success_hist, cost_hist, iters_log = [], [], [], []

    if verbose:
        log(f"Starting CE: {n_iters} iters x {n} samples")
    bar = tqdm(range(n_iters), disable=not verbose, desc="CE")
    for it in bar:
        params = draw(dists, it)

        t0 = time.perf_counter()
        out = rollout.run(params)
        iter_times.append(time.perf_counter() - t0)

        cost, n_act = out["cost"], out["n_act"]
        gok, proxy = out["ok"], out["proxy"]
        tracks = rollout.probe.numpy() if record_paths else None

        if record_paths:
            pop = _pop(out, tracks)
            if on_iter is not None:
                on_iter(it, pop)

        n_goal = int(gok.sum())
        best_c = cost[gok].min() if n_goal else np.inf
        success_hist.append(n_goal)
        cost_hist.append(float(best_c) if n_goal else None)
        # this iter's best env (goal-reacher by cost, else closest-to-feasible), recorded
        # so a saved solution can be replayed at any iteration
        bi_iter = (int(np.where(gok)[0][cost[gok].argmin()]) if n_goal
                   else int(np.lexsort((proxy, -n_act))[0]))
        if verbose:
            bar.set_postfix(goal=f"{n_goal}/{n}", acts=f"{int(n_act.max())}/{n_actions}",
                            best=f"{min(best_c, best['cost']):.0f}")

        # Always refit (even with zero goal success): the tiered fallback pulls the
        # population toward feasibility via action-count + proxy.
        idx = _select_elites(gok, n_act, proxy, cost, min(n_elite, n))
        dists = [_refit_idx(params[a], idx) for a in range(len(regs))]

        iters_log.append({
            "cost": float(best_c) if n_goal else float("nan"),
            "n_goal": n_goal,
            "params": [np.asarray(p[bi_iter]) for p in params],
            "mean": [m for (m, _) in dists],
            "std": [s for (_, s) in dists],
            "elite_cost": np.asarray(cost[idx]),
            "elite_by_goal": bool(n_goal > 0),
        })

        if n_goal:
            bi = np.where(gok)[0][np.argmin(cost[gok])]
            if cost[bi] < best["cost"]:
                best = {"cost": float(cost[bi]), "params": [np.asarray(p[bi]) for p in params]}
        # track the closest-to-feasible env too, so there's always something to replay
        fb = int(np.lexsort((proxy, -n_act))[0])
        if proxy[fb] < effort["proxy"]:
            effort = {"proxy": float(proxy[fb]), "cost": float(cost[fb]),
                      "params": [np.asarray(p[fb]) for p in params]}

    best["reached_goal"] = best["params"] is not None
    if not best["reached_goal"]:  # best-effort fallback so callers can still replay
        best["cost"], best["params"] = effort["cost"], effort["params"]

    best["iters"] = iters_log
    best["meta"] = {
        "scenario": scenario.name, "device": "warp", "reached_goal": best["reached_goal"],
        "n": n, "n_elite": n_elite, "n_iters": n_iters, "seed": seed, "n_actions": n_actions,
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

    return best, dists, pop

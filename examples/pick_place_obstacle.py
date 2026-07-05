"""CE optimization on the pick/place-around-an-obstacle scenario.

Pick a cube off table1, carry/place it on table2, return to the exit. The same plan +
parameter regions yield different solutions depending on the obstacle between the tables
(wall -> route around; ramp -> release to slide). Runs CE, plots the population each
iteration, then saves and renders the best solution.

    uv run python examples/pick_place_obstacle.py [--variant wall|ramp|ramp_forced]
                                        [--iters 8] [--n 512] [--n-elite 30] [--seed 0]
                                        [--tag NAME] [--last-plot-only]
                                        [--camera behind|top|gripper]
"""

import argparse
import os


def main(variant, iters, n, last_plot_only, camera, seed=0, n_elite=30, tag=None,
         elite_temp=None, smooth=None, n_modes=1, explore=0.0, robust_k=None,
         exit_frac=1.0):
    from tamp_ce_mjwarp import ce, plan, scenarios, viz
    from tamp_ce_mjwarp.ik import IK
    from tamp_ce_mjwarp.log import log, timed
    from tamp_ce_mjwarp.sim import Sim

    log(f"Loading scenario 'pick_place_obstacle_{variant}'")
    sim = Sim(scenarios.scene_path(f"pick_place_obstacle_{variant}"))
    sc = scenarios.make_pick_place_obstacle(sim, variant)
    ik = IK(n)  # one whole-body solver; each action picks its mode via spec.ik

    outdir = f"solutions/{sc.name}" + (f"_{tag}" if tag else "")
    os.makedirs(outdir, exist_ok=True)

    with timed("Plotting parameter regions"):
        viz.plot_parameter_regions(sc, sim, f"{outdir}/parameter_regions.png",
                                   view=camera if camera != "gripper" else "behind")

    def on_iter(it, pop):
        if last_plot_only and it != iters - 1:
            return
        with timed(f"Plotting iter {it}"):
            viz.plot_paths(sc, pop, f"{outdir}/iter_{it:02d}.png", it=it)

    best, _, _ = ce.run(sim, ik, sc, n_iters=iters, n=n, n_elite=n_elite, seed=seed,
                        elite_temp=elite_temp, smooth=smooth, n_modes=n_modes,
                        explore=explore, robust_k=robust_k, exit_frac=exit_frac,
                        record_paths=True, on_iter=on_iter)

    # Always save + render: a real solution if the goal was reached, else the
    # closest-to-feasible "best-effort" rollout (so failures are inspectable on video).
    tag = "" if best["reached_goal"] else "_best_effort"
    h5 = f"{outdir}/{sc.name}{tag}.h5"
    viz.save_solution(h5, best["params"], best["cost"], best["meta"], iters=best["iters"])
    viz.plot_ce_diagnostics(h5)
    viz.plot_elite_costs(h5)
    if sc.nav is not None:  # VI cost-to-go for each base-move goal of the best plan
        for i, spec in enumerate(sc.specs):
            if spec.kind == "move":
                viz.plot_value_field(sc, best["params"][i][:2], f"{outdir}/vi_move{i}.png")

    label = "best-solution" if best["reached_goal"] else "best-effort (goal NOT reached)"
    with timed(f"Rendering {label} video"):
        traj = plan.replay(sim, IK(1), sc, best["params"])
        cam = viz.camera(sim.mj_model, camera)
        viz.render_trajectory(sim.mj_model, traj, f"{outdir}/{sc.name}{tag}.mp4",
                              camera=cam, every=11, dt=sim.dt)

    log(f"Saved {outdir}/ ({sc.name}{tag}.h5, .mp4, iter_*.png) — {label}, "
        f"cost {best['cost']:.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["wall", "ramp", "ramp_forced"], default="wall")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--n-elite", type=int, default=30)
    ap.add_argument("--elite-temp", type=float, default=None,
                    help="cost-rank weighting of the refit (see ce.run); default uniform")
    ap.add_argument("--smooth", type=float, default=None,
                    help="CE smoothing: fraction of the previous dist retained per refit")
    ap.add_argument("--n-modes", type=int, default=1,
                    help="cluster elites into this many modes, one Gaussian each (see ce.run)")
    ap.add_argument("--explore", type=float, default=0.0,
                    help="fraction of every population drawn fresh from the priors")
    ap.add_argument("--robust-k", type=int, default=None,
                    help="rank elites by k-NN neighbourhood-smoothed cost (see ce.run)")
    ap.add_argument("--exit-frac", type=float, default=1.0,
                    help="end an action once this fraction of worlds succeeded (see Rollout)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None,
                    help="suffix for the output dir, e.g. seed1 (avoids overwriting runs)")
    ap.add_argument("--last-plot-only", action="store_true",
                    help="only plot the final iteration (skip per-iter convergence plots)")
    ap.add_argument("--camera", choices=["behind", "top", "gripper"], default="behind")
    args = ap.parse_args()
    main(args.variant, args.iters, args.n, args.last_plot_only, args.camera,
         seed=args.seed, n_elite=args.n_elite, tag=args.tag, elite_temp=args.elite_temp,
         smooth=args.smooth, n_modes=args.n_modes, explore=args.explore,
         robust_k=args.robust_k, exit_frac=args.exit_frac)

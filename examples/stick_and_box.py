"""CE optimization on the stick_and_box scenario (non-prehensile push).

A free-standing stick has a block balanced on top; an open box sits a stick-height away.
The plan is move-to(a standoff around the stick) -> push (drive the closed hand into the
stick), toppling it so the block is flung into the box. Variants: "right" (stick on the
robot's right) and "between" (box between the robot and the stick).

    uv run python examples/stick_and_box.py [--variant right|between] [--iters 8] [--n 512]
                                            [--n-elite 30] [--seed 0] [--tag NAME]
                                            [--last-plot-only] [--camera behind|top|gripper]
"""

import argparse
import os


def main(iters, n, last_plot_only, camera, variant="right", seed=0, n_elite=30, tag=None,
         elite_temp=None, smooth=None, n_modes=1, explore=0.0, robust_k=None):
    from tamp_ce_mjwarp import ce, plan, scenarios, viz
    from tamp_ce_mjwarp.ik import IK
    from tamp_ce_mjwarp.log import log, timed
    from tamp_ce_mjwarp.sim import Sim

    log(f"Loading scenario 'stick_and_box_{variant}'")
    sim = Sim(scenarios.scene_path(f"stick_and_box_{variant}"))
    sc = scenarios.make_stick_and_box(sim, variant)
    ik = IK(n)

    outdir = f"solutions/{sc.name}" + (f"_{tag}" if tag else "")
    os.makedirs(outdir, exist_ok=True)

    def on_iter(it, pop):
        if last_plot_only and it != iters - 1:
            return
        with timed(f"Plotting iter {it}"):
            viz.plot_paths(sc, pop, f"{outdir}/iter_{it:02d}.png", it=it)

    best, _, _ = ce.run(sim, ik, sc, n_iters=iters, n=n, n_elite=n_elite, seed=seed,
                        elite_temp=elite_temp, smooth=smooth, n_modes=n_modes,
                        explore=explore, robust_k=robust_k,
                        record_paths=True, on_iter=on_iter)

    tag = "" if best["reached_goal"] else "_best_effort"
    h5 = f"{outdir}/{sc.name}{tag}.h5"
    viz.save_solution(h5, best["params"], best["cost"], best["meta"], iters=best["iters"])
    viz.plot_ce_diagnostics(h5)
    viz.plot_elite_costs(h5)

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
    ap.add_argument("--variant", choices=["right", "between"], default="right")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--n-elite", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None,
                    help="suffix for the output dir, e.g. seed1 (avoids overwriting runs)")
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
    ap.add_argument("--last-plot-only", action="store_true",
                    help="only plot the final iteration (skip per-iter convergence plots)")
    ap.add_argument("--camera", choices=["behind", "top", "gripper"], default="behind")
    args = ap.parse_args()
    main(args.iters, args.n, args.last_plot_only, args.camera, variant=args.variant,
         seed=args.seed, n_elite=args.n_elite, tag=args.tag, elite_temp=args.elite_temp,
         smooth=args.smooth, n_modes=args.n_modes, explore=args.explore,
         robust_k=args.robust_k)

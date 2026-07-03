"""CE optimization on the pick/place-around-an-obstacle scenario.

Pick a cube off table1, carry/place it on table2, return to the exit. The same plan +
parameter regions yield different solutions depending on the obstacle between the tables
(wall -> route around; ramp -> release to slide). Runs CE, plots the population each
iteration, then saves and renders the best solution.

    uv run python examples/pick_place_obstacle.py [--variant wall|ramp|ramp_forced]
                                        [--iters 8] [--n 512] [--last-plot-only]
                                        [--camera behind|top|gripper]
"""

import argparse
import os


def main(variant, iters, n, last_plot_only, camera):
    from tamp_ce_mjwarp import ce, plan, scenarios, viz
    from tamp_ce_mjwarp.ik import IK
    from tamp_ce_mjwarp.log import log, timed
    from tamp_ce_mjwarp.sim import Sim

    log(f"Loading scenario 'pick_place_obstacle_{variant}'")
    sim = Sim(scenarios.scene_path(f"pick_place_obstacle_{variant}"))
    sc = scenarios.make_pick_place_obstacle(sim, variant)
    ik = IK(n)  # one whole-body solver; each action picks its mode via spec.ik

    outdir = f"solutions/{sc.name}"
    os.makedirs(outdir, exist_ok=True)

    with timed("Plotting parameter regions"):
        viz.plot_parameter_regions(sc, sim, f"{outdir}/parameter_regions.png",
                                   view=camera if camera != "gripper" else "behind")

    def on_iter(it, pop):
        if last_plot_only and it != iters - 1:
            return
        with timed(f"Plotting iter {it}"):
            viz.plot_paths(sc, pop, f"{outdir}/iter_{it:02d}.png", it=it)

    best, _, _ = ce.run(sim, ik, sc, n_iters=iters, n=n, n_elite=30,
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
    ap.add_argument("--last-plot-only", action="store_true",
                    help="only plot the final iteration (skip per-iter convergence plots)")
    ap.add_argument("--camera", choices=["behind", "top", "gripper"], default="behind")
    args = ap.parse_args()
    main(args.variant, args.iters, args.n, args.last_plot_only, args.camera)

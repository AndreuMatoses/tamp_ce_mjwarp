"""CE optimization on the stick_and_box scenario (non-prehensile push).

A free-standing stick has a block balanced on top; an open box sits a stick-height away.
The plan is move-to(a standoff around the stick) -> push (drive the closed hand into the
stick), toppling it so the block is flung into the box.

    uv run python examples/stick_and_box.py [--iters 8] [--n 512] [--last-plot-only]
                                            [--camera behind|top|gripper]
"""

import argparse
import os


def main(iters, n, last_plot_only, camera):
    from tamp_ce_mjwarp import ce, plan, scenarios, viz
    from tamp_ce_mjwarp.ik import IK
    from tamp_ce_mjwarp.log import log, timed
    from tamp_ce_mjwarp.sim import Sim

    log("Loading scenario 'stick_and_box'")
    sim = Sim("scenes/stick_and_box.xml")
    sc = scenarios.make_stick_and_box(sim)
    ik = IK(n)

    outdir = f"solutions/{sc.name}"
    os.makedirs(outdir, exist_ok=True)

    def on_iter(it, pop):
        if last_plot_only and it != iters - 1:
            return
        with timed(f"Plotting iter {it}"):
            viz.plot_paths(sc, pop, f"{outdir}/iter_{it:02d}.png", it=it)

    best, _, _ = ce.run(sim, ik, sc, n_iters=iters, n=n, n_elite=30,
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
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--last-plot-only", action="store_true",
                    help="only plot the final iteration (skip per-iter convergence plots)")
    ap.add_argument("--camera", choices=["behind", "top", "gripper"], default="behind")
    args = ap.parse_args()
    main(args.iters, args.n, args.last_plot_only, args.camera)

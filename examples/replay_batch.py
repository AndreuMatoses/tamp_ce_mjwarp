"""Visualize a whole BATCH of plan realizations superposed in one scene.

Samples a population of plans (by default the iteration-0 uniform priors — exactly what
CE starts from), rolls them ALL out in parallel, and renders/replays them superposed:
env 0 is drawn solid, every other env is a translucent ghost.

    uv run python examples/replay_batch.py pick_place_obstacle_wall                  # viewer
    uv run python examples/replay_batch.py pick_place_obstacle_wall --video out.mp4  # headless
    uv run python examples/replay_batch.py pick_place_obstacle_wall --n 48 --filter fail
    uv run python examples/replay_batch.py <name> --from-solution <file.h5> --iter -1
    uv run python examples/replay_batch.py <name> --from-solution <file.h5> \
        --all-iters --video iters.mp4 --speed 4    # one segment per CE iteration

--filter all|success|fail keeps all envs / only goal-reachers / only failures.
--from-solution <h5> [--iter N] samples from a saved run's refitted Gaussian instead of
the uniform priors.
--all-iters (needs --from-solution + --video) renders one batch segment per CE iteration
into a single video: segment 0 from the uniform priors (what CE iter 0 sampled), segment
k from the dist refit after iter k-1. --speed multiplies the frame stride.
"""

import argparse


def main(name, video, n, filt, seed, from_solution, it, every, speed, alpha, camera,
         all_iters=False):
    import numpy as np

    from tamp_ce_mjwarp import ce, scenarios, viz
    from tamp_ce_mjwarp.ik import IK
    from tamp_ce_mjwarp.log import log, timed
    from tamp_ce_mjwarp.plan import Rollout
    from tamp_ce_mjwarp.sim import Sim

    sim = Sim(scenarios.scene_path(name))
    sc = scenarios.build(sim, name)

    if all_iters:
        import h5py
        import imageio.v2 as imageio

        assert from_solution and video, "--all-iters needs --from-solution and --video"
        with h5py.File(from_solution, "r") as f:
            n_segments = len(f["iters"]) + 1  # uniform prior + one per refit
        ro = Rollout(sim, IK(n), sc, n, record_qpos=True)
        cam = viz.camera(sim.mj_model, camera)
        stride = max(1, round(every * speed))  # video plays `speed`x faster than realtime
        writer = imageio.get_writer(video, fps=1.0 / (every * sim.dt))
        log(f"Rendering {n_segments} segments x {n} envs (stride {stride}) -> {video}")
        for k in range(n_segments):
            dists = ([None] * len(sc.regions) if k == 0
                     else viz.load_iter_dist(from_solution, k - 1))
            params = ce.sample_population(sc, dists, n, seed + k)
            out = ro.run(params)
            gok = out["ok"]
            keep = {"all": np.ones(n, bool), "success": gok, "fail": ~gok}[filt]
            traj = ro.traj.numpy()[:ro.steps_run].transpose(1, 0, 2)[keep]
            label = f"iter {k}   goal {int(gok.sum())}/{n}"
            log(f"  segment {k}: {label}, {len(traj)} env(s) kept")
            if len(traj):
                viz.render_batch(sim.mj_model, traj, None, camera=cam, every=stride,
                                 alpha=alpha, dt=sim.dt, writer=writer, label=label)
        writer.close()
        log(f"wrote {video}")
        return

    if from_solution:
        dists = viz.load_iter_dist(from_solution, it)
        log(f"Sampling {n} plans from {from_solution} (iter {it if it is not None else 'last'})")
    else:
        dists = [None] * len(sc.regions)
        log(f"Sampling {n} plans from the uniform region priors (iteration 0)")
    params = ce.sample_population(sc, dists, n, seed)

    with timed("Rolling out the batch"):
        ro = Rollout(sim, IK(n), sc, n, record_qpos=True)
        out = ro.run(params)
        traj = ro.traj.numpy().transpose(1, 0, 2)  # (N, T, nq)
    gok = out["ok"]

    keep = {"all": np.ones(n, bool), "success": gok, "fail": ~gok}[filt]
    idx = np.where(keep)[0]
    log(f"{int(gok.sum())}/{n} reached goal; showing {len(idx)} env(s) [filter={filt}]")
    if len(idx) == 0:
        log("no envs match the filter — nothing to show.")
        return
    batch = traj[idx]  # (K, T, nq)

    if video:
        cam = viz.camera(sim.mj_model, camera)
        with timed("Rendering superposed batch video"):
            viz.render_batch(sim.mj_model, batch, video, camera=cam, every=every,
                             alpha=alpha, dt=sim.dt)
        log(f"wrote {video}")
    else:
        log("Launching interactive viewer (close window or Ctrl-C to exit) ...")
        try:
            viz.play_viewer_batch(sim.mj_model, batch, dt=sim.dt, speed=speed,
                                  every=every, alpha=alpha)
        except Exception as e:  # no display -> point at the headless path
            log(f"viewer failed ({e}); re-run with --video out.mp4 for a headless render.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", help="e.g. pick_place_obstacle_wall / stick_and_box")
    ap.add_argument("--video", default=None, help="render to this mp4 instead of the viewer")
    ap.add_argument("--n", type=int, default=32, help="number of plans to sample & superpose")
    ap.add_argument("--filter", choices=["all", "success", "fail"], default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-solution", default=None,
                    help="sample from this saved solution's fitted distribution")
    ap.add_argument("--iter", type=int, default=None,
                    help="which iteration's dist (with --from-solution)")
    ap.add_argument("--all-iters", action="store_true",
                    help="render every CE iteration as one concatenated video segment")
    ap.add_argument("--every", type=int, default=11, help="step stride for playback/render")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="viewer playback speed / --all-iters video speedup")
    ap.add_argument("--alpha", type=float, default=0.3, help="ghost transparency")
    ap.add_argument("--camera", choices=["behind", "top", "gripper"], default="top")
    args = ap.parse_args()
    main(args.scenario, args.video, args.n, args.filter, args.seed,
         args.from_solution, args.iter, args.every, args.speed, args.alpha, args.camera,
         all_iters=args.all_iters)

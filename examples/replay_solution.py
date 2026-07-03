"""Replay a saved solution (.h5) to inspect it — interactive MuJoCo viewer by default.

Loads the parameters + metadata from a solution file, rebuilds the scenario from its
name, re-runs the plan deterministically, and either opens the interactive viewer or
renders a video headlessly.

    uv run python examples/replay_solution.py solutions/<name>/<name>.h5
    uv run python examples/replay_solution.py <file.h5> --video out.mp4   # headless render
    uv run python examples/replay_solution.py <file.h5> --speed 0.3       # slow-mo viewer
    uv run python examples/replay_solution.py <file.h5> --iter 3          # replay iter 3's best

By default this replays the LAST CE iteration's best plan; --iter N picks another (N
indexes the recorded iterations, negatives allowed).
"""

import argparse


def main(path, video, speed, every, camera, it, show):
    import numpy as np

    from tamp_ce_mjwarp import plan, scenarios, viz
    from tamp_ce_mjwarp.ik import IK
    from tamp_ce_mjwarp.log import log, timed
    from tamp_ce_mjwarp.sim import Sim

    params, cost, meta = viz.load_solution(path)
    name = str(meta["scenario"])
    log(f"Loaded {path}: scenario={name}  cost={cost:.0f}  "
        f"reached_goal={meta.get('reached_goal')}")

    params, info = viz.load_iter_params(path, it)
    if info["iter"] is not None:
        log(f"  replaying iter {info['iter']}/{info['n_iters'] - 1} "
            f"(cost={info['cost']:.0f}, n_goal={info['n_goal']})")
    log(f"  params: {[f'{n}{list(np.round(p, 3))}' for n, p in zip(meta.get('param_names', []), params)]}")

    sim = Sim(scenarios.scene_path(name))
    sc = scenarios.build(sim, name)

    with timed("Replaying plan (rollout)"):
        traj = plan.replay(sim, IK(1), sc, params)

    if video:
        cam = viz.camera(sim.mj_model, camera)
        viz.render_trajectory(sim.mj_model, traj, video, camera=cam, every=every,
                              show=show, dt=sim.dt)
        log(f"wrote {video}")
    else:
        log("Launching interactive viewer (close window or Ctrl-C to exit) ...")
        try:
            viz.play_viewer(sim.mj_model, traj, dt=sim.dt, speed=speed, every=every, show=show)
        except Exception as e:  # no display -> point at the headless path
            log(f"viewer failed ({e}); re-run with --video out.mp4 for a headless render.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="path to a solution .h5")
    ap.add_argument("--video", default=None, help="render to this mp4 instead of the viewer")
    ap.add_argument("--speed", type=float, default=1.0, help="viewer playback speed")
    ap.add_argument("--every", type=int, default=11, help="step stride for playback/render")
    ap.add_argument("--camera", choices=["behind", "top", "gripper"], default="gripper",
                    help="camera for --video (gripper = follow the grasp)")
    ap.add_argument("--iter", type=int, default=None,
                    help="which CE iteration's best plan to replay (default: last)")
    ap.add_argument("--show", choices=["visual", "collision", "both"], default="visual",
                    help="render the detailed meshes, the collision proxy, or both")
    args = ap.parse_args()
    main(args.file, args.video, args.speed, args.every, args.camera, args.iter, args.show)

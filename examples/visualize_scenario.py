"""Inspect a scenario's sampling regions in the LIVE MuJoCo viewer (no CE run).

Draws each parameter region as a translucent primitive (annulus -> standoff ring, box ->
ground box, place point_quat -> its 3D box) plus grasp/place approach arrows whose
direction is the gripper's approach axis and length encodes the yaw spin.

    uv run python examples/visualize_scenario.py pick_place_obstacle_wall
    uv run python examples/visualize_scenario.py stick_and_box --camera top

Needs a display. --n-quat sets orientation samples per quaternion region.
"""

import argparse


def main(scenario, n_quat, view):
    import time

    import mujoco
    import mujoco.viewer

    from tamp_ce_mjwarp import scenarios, viz
    from tamp_ce_mjwarp.sim import Sim

    sim = Sim(scenarios.scene_path(scenario))
    sc = scenarios.build(sim, scenario)
    m = sim.mj_model
    d, grasp_anchor = viz.region_home_data(sim, sc)

    with mujoco.viewer.launch_passive(m, d) as v:
        viz._apply_show(v.opt, "visual")
        v.user_scn.ngeom = 0
        legend = viz.add_region_geoms(v.user_scn, sc, grasp_anchor, n_quat=n_quat)
        cam = viz.camera(m, view)  # frame like the rendered videos
        v.cam.azimuth, v.cam.elevation, v.cam.distance = cam.azimuth, cam.elevation, cam.distance
        v.cam.lookat[:] = cam.lookat

        print(f"\nScenario '{scenario}' — {len(legend)} parameter regions "
              f"({n_quat} samples per quaternion region):")
        for nm, rgb in legend:
            print(f"  - {nm:<14} rgb=({rgb[0]:.2f}, {rgb[1]:.2f}, {rgb[2]:.2f})")
        print("\nApproach arrows: direction = gripper approach axis, length ~ yaw spin.")
        print("Viewer open — orbit to inspect; close the window to exit.\n")

        while v.is_running():
            v.sync()
            time.sleep(0.05)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", help="e.g. pick_place_obstacle_wall / stick_and_box")
    ap.add_argument("--n-quat", type=int, default=64,
                    help="orientation samples per quaternion region (default 64)")
    ap.add_argument("--camera", choices=["behind", "top", "gripper"], default="behind")
    args = ap.parse_args()
    main(args.scenario, args.n_quat, args.camera)

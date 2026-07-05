"""Scenario registry: each bundles a scene, the per-action regions + specs, and the
symbolic goal. `examples/*.py` pick one and call ce.run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tamp_ce_mjwarp import nav, regions
from tamp_ce_mjwarp.plan import ActionSpec


@dataclass
class Scenario:
    name: str
    scene_xml: str
    regions: list
    specs: list
    graspable_qadr: int  # freejoint qpos addr of the object CE tracks (for plot/probe)
    graspable_xy: tuple = None  # initial (x,y) to place that object at; None => keep XML pos
    nav: tuple = None  # (occ, bounds, res, n_iters, remap) for base planning, or None
    obstacles: list = field(default_factory=list)  # rects (xmin,xmax,ymin,ymax,inflate)
    goal: callable = None  # goal(sim, qpos (n, nq)) -> (ok (n,) bool, proxy (n,) float);
    #                        the symbolic objective + a smooth distance proxy CE falls
    #                        back to. If None, the goal is "all actions succeeded".


_PPO_PREFIX = "pick_place_obstacle_"


def _obstacle_scene_stem(variant):
    """ramp_forced reuses the plain ramp scene (only its priors differ)."""
    return "pick_place_obstacle_ramp" if variant.startswith("ramp") else _PPO_PREFIX + variant


def make_pick_place_obstacle(sim, variant="wall"):
    """Pick a cube off table1, carry/place it on table2, exit, routing the base around an
    obstacle between the tables. Variants: "wall" (a box barrier), "ramp" (a base prism +
    45 deg wedge, so a placed cube can slide down onto table2), and "ramp_forced" (the ramp
    scene with priors tuned to force the slide solution). Region centers are read from the
    scene geoms, so moving an object in the XML moves its region too."""
    cq = 11
    table1 = sim.geom_xy("table1")
    table2 = sim.geom_xy("table2")
    cube_xy = table1  # the cube starts on table1
    specs = [
        ActionSpec("move", timeout=5.0),
        # whole-body IK pick/place: the base drives in to reach (see the wide annuli below)
        ActionSpec("pick", timeout=5.0, obj_qadr=cq, z_off=0.0, ik="full"),
        ActionSpec("move", timeout=5.5),   # around the obstacle: longer path
        # position-only place (free_ori): a fixed top-down place can't reach over a table /
        # up to the ramp top, so the arm reaches the spot however it can
        ActionSpec("place", timeout=4.4, ik="full", free_ori=True),
        ActionSpec("move", timeout=5.0),
    ]
    regs = [
        # standoff rings beyond the ~0.6 m arm reach, so the full-body-IK pick/place must
        # translate the base in
        regions.annulus(table1, 1.1, 1.5, name="near table1"),
        regions.quat(tilt=0.3, name="pick quat"),
        # r_out 2.05 / place-y down to 1.7: keeps the ramp variant's slide mode samplable
        # (standoffs south of y~0.55 = r>1.9, drop point just past the high edge at y~1.8)
        # while the same definitions still cover the wall variant's go-around solutions
        regions.annulus(table2, 1.2, 2.05, name="near table2"),
        regions.point(center=(table2[0], 2.2, 0.3),
                      size=(0.2, 0.5, 0.6), name="place pose"),
        regions.box(center=(-1.0, 1.0), lo=(-0.3, -0.3), hi=(0.3, 0.3), name="exit"),
    ]
    if variant == "ramp_forced":
        # priors biased to force the ramp solution (drop over the high edge -> slide down)
        regs[2] = regions.box(center=(2.0, 0.85), lo=(-0.25, -0.25), hi=(0.25, 0.25),
                              name="approach ramp")
        regs[3] = regions.point(center=(2.0, 1.8, 0.55), size=(0.2, 0.35, 0.2),
                                name="place on ramp")
    # The grid IS the base's reachable region (a goal outside it gets clipped to the boundary).
    bounds = (-1.6, 4.2, -2.6, 4.2)
    res = 0.1
    # Tables get a thin pad (route around but still approach to grasp range); the obstacle
    # a wide pad — the base carries the cube ~0.5 m out in front, keep it clear.
    obstacles = [sim.geom_rect("table1", inflate=0.2), sim.geom_rect("table2", inflate=0.2)]
    if variant == "wall":
        obstacles.append(sim.geom_rect("wall", inflate=0.45))
    elif variant in ("ramp", "ramp_forced"):
        # geom_rect ignores the slope geom's rotation: pass the ramp's combined footprint
        obstacles.append((1.0, 3.0, 1.65, 2.25, 0.45))
    occ = nav.occupancy_from_rects(bounds, res, obstacles)
    navt = (occ, bounds, res, 220, nav.nearest_free(occ))  # VI iters ~ grid diameter

    # Symbolic goal: the cube rests on table2's surface AND the robot is parked in the exit
    # region (a place that merely "reached the pose and opened" may still drop the cube).
    tx0, tx1, ty0, ty1, _ = sim.geom_rect("table2")
    surf_z = sim.geom_top_z("table2") + 0.02  # cube half-height rests above the top
    ex_lo, ex_hi = regs[-1].a, regs[-1].b     # the exit box region
    bx, by = int(sim.base_qadr[0]), int(sim.base_qadr[1])

    def goal(sim, qpos):
        blk = qpos[:, cq:cq + 3]
        dz = np.abs(blk[:, 2] - surf_z)
        on_table = ((blk[:, 0] > tx0) & (blk[:, 0] < tx1) &
                    (blk[:, 1] > ty0) & (blk[:, 1] < ty1) & (dz < 0.06))
        base = qpos[:, [bx, by]]
        in_exit = np.all((base > ex_lo) & (base < ex_hi), axis=1)
        # smooth proxy (lower = closer): block->table2 surface + base->exit center
        dx = np.maximum(np.maximum(tx0 - blk[:, 0], blk[:, 0] - tx1), 0.0)
        dy = np.maximum(np.maximum(ty0 - blk[:, 1], blk[:, 1] - ty1), 0.0)
        proxy = np.sqrt(dx * dx + dy * dy) + dz + np.linalg.norm(
            base - 0.5 * (ex_lo + ex_hi), axis=1)
        return on_table & in_exit, proxy

    return Scenario(f"pick_place_obstacle_{variant}",
                    f"scenes/{_obstacle_scene_stem(variant)}.xml",
                    regs, specs, cq, cube_xy, navt, obstacles=obstacles, goal=goal)


def make_stick_and_box(sim):
    """Topple a free-standing stick (with a block balanced on top) so the block is flung
    into an open box a stick-height away — a non-prehensile push. The plan is move-to(a
    standoff around the stick) then push (drive the closed hand into the stick with
    whole-body IK, slowing on the final approach)."""
    stick_xy = sim.body_xy("stick")
    block_q = sim.joint_qadr("block_joint")  # the object CE tracks / the goal checks

    specs = [
        ActionSpec("move", timeout=5.0),
        ActionSpec("push", timeout=6.0, ik="full"),  # base drives in with the arm
    ]
    regs = [
        # wide ring beyond arm reach: the full-body-IK push must translate the base in
        regions.annulus(stick_xy, 0.8, 1.0, name="near stick"),
        regions.point_quat(center=(stick_xy[0], stick_xy[1] + 0.06, 0.34),
                           size=(0.1, 0.12, 0.16), tilt=0.5, name="push pose"),
    ]

    bounds = (-0.6, 2.6, -1.6, 1.2)
    res = 0.1
    sx, sy = stick_xy
    hsx, hsy = (float(v) for v in sim.mj_model.geom("stick_geom").size[:2])
    # The base must drive close to the stick to push it, so the stick gets only a thin pad;
    # the box keeps a wide one.
    obstacles = [(sx - hsx, sx + hsx, sy - hsy, sy + hsy, 0.05),
                 sim.geom_rect("box_floor", inflate=0.20)]
    occ = nav.occupancy_from_rects(bounds, res, obstacles)
    navt = (occ, bounds, res, 180, nav.nearest_free(occ))

    # Symbolic goal: the block ended up INSIDE the open box (xy within the footprint and
    # below the wall rim, i.e. it dropped in rather than sailing past).
    bx0, bx1, by0, by1, _ = sim.geom_rect("box_floor")
    rim_z = sim.geom_top_z("box_wb")  # top of a box wall

    def goal(sim, qpos):
        blk = qpos[:, block_q:block_q + 3]
        in_box = ((blk[:, 0] > bx0) & (blk[:, 0] < bx1) &
                  (blk[:, 1] > by0) & (blk[:, 1] < by1) & (blk[:, 2] < rim_z))
        dx = np.maximum(np.maximum(bx0 - blk[:, 0], blk[:, 0] - bx1), 0.0)
        dy = np.maximum(np.maximum(by0 - blk[:, 1], blk[:, 1] - by1), 0.0)
        proxy = np.sqrt(dx * dx + dy * dy) + np.maximum(blk[:, 2] - rim_z, 0.0)  # ->0 inside
        return in_box, proxy

    return Scenario("stick_and_box", "scenes/stick_and_box.xml", regs, specs, block_q,
                    nav=navt, obstacles=obstacles, goal=goal)


def simple_pickplace():
    """Pick a floor cube and place it nearby (no base motion). Smoke test for the loop."""
    cube_qadr = 11
    return Scenario(
        name="simple_pickplace",
        scene_xml="robot_models/dinova/dinova_block_scene.xml",
        regions=[regions.quat(0.3),
                 regions.point_quat(center=(0.45, 0.22, 0.04), size=(0.08, 0.08, 0.08), tilt=0.4)],
        specs=[ActionSpec("pick", timeout=7.5, obj_qadr=cube_qadr, z_off=0.01),
               ActionSpec("place", timeout=7.5)],
        graspable_qadr=cube_qadr,
        graspable_xy=(0.5, 0.0),
    )


def scene_path(name):
    """Scene XML for a scenario name (so a saved solution can be reloaded from name alone)."""
    if name.startswith(_PPO_PREFIX):
        return f"scenes/{_obstacle_scene_stem(name[len(_PPO_PREFIX):])}.xml"
    if name == "stick_and_box":
        return "scenes/stick_and_box.xml"
    if name == "simple_pickplace":
        return "robot_models/dinova/dinova_block_scene.xml"
    raise ValueError(f"unknown scenario {name!r}")


def build(sim, name):
    """Rebuild a Scenario from its name (e.g. to replay a saved solution)."""
    if name.startswith(_PPO_PREFIX):
        return make_pick_place_obstacle(sim, name[len(_PPO_PREFIX):])
    if name == "stick_and_box":
        return make_stick_and_box(sim)
    if name == "simple_pickplace":
        return simple_pickplace()
    raise ValueError(f"unknown scenario {name!r}")

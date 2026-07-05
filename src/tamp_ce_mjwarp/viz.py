"""Save and visualize found solutions.

Rendering uses Mesa EGL (surfaceless) which works headless. A solution is a qpos
trajectory (replayed deterministically) written to a video, plus HDF5 of the symbolic
params/cost/per-iteration distributions.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np

# Robot geom groups (see dinova_robot.xml): 2 = detailed visual meshes, 4 = simplified
# collision proxy. `show` lets a render/viewer swap between them.
_VISUAL_GROUP, _COLLISION_GROUP = 2, 4


def _apply_show(opt, show):
    """Set the robot visual/collision geom-group visibility on an existing MjvOption."""
    opt.geomgroup[_VISUAL_GROUP] = show in ("visual", "both")
    opt.geomgroup[_COLLISION_GROUP] = show in ("collision", "both")
    return opt


def vis_option(show="visual"):
    """A fresh MjvOption showing the robot's visual meshes, collision proxy, or both."""
    return _apply_show(mujoco.MjvOption(), show)


def camera(mj_model, view="behind"):
    """An MjvCamera: `behind` looks over the robot's start pose at the whole scene; `top`
    is near-top-down; `gripper` tracks the gripper so the grasp is always in frame."""
    cam = mujoco.MjvCamera()
    ext = float(mj_model.stat.extent)
    if view == "gripper":
        site = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = int(mj_model.site_bodyid[site])
        cam.azimuth, cam.elevation, cam.distance = 130.0, -20.0, 0.9
        return cam
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = mj_model.stat.center
    if view == "top":
        cam.azimuth, cam.elevation, cam.distance = 90.0, -89.0, 2.0 * ext
    else:  # behind the robot start, looking along +x: the obstacle wall is seen edge-on
        cam.azimuth, cam.elevation, cam.distance = 0.0, -25.0, 1.3 * ext
    return cam


def render_trajectory(mj_model, qpos_traj, path, camera=-1, width=640, height=480, fps=None,
                      every=10, show="visual", dt=None):
    """Replay a (T, nq) qpos trajectory and write a video to `path`. Pass `dt` (the sim
    timestep) so playback is real-time: fps defaults to 1/(every*dt), else 30."""
    if fps is None:
        fps = (1.0 / (every * dt)) if dt else 30
    qpos_traj = np.asarray(qpos_traj)
    data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height, width)
    opt = vis_option(show)
    frames = []
    for i in range(0, len(qpos_traj), every):
        data.qpos[:] = qpos_traj[i]
        mujoco.mj_forward(mj_model, data)
        renderer.update_scene(data, camera=camera, scene_option=opt)
        frames.append(renderer.render())
    renderer.close()
    imageio.mimsave(path, frames, fps=fps)
    return path


def save_frame(mj_model, qpos, path, camera=-1, width=640, height=480, show="visual"):
    data = mujoco.MjData(mj_model)
    data.qpos[:] = np.asarray(qpos)
    mujoco.mj_forward(mj_model, data)
    renderer = mujoco.Renderer(mj_model, height, width)
    renderer.update_scene(data, camera=camera, scene_option=vis_option(show))
    imageio.imwrite(path, renderer.render())
    renderer.close()
    return path


def _add_geom(scn, gtype, size, pos, mat, rgba, label=""):
    """Append one decorative primitive to a render scene (no-op if full)."""
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, int(gtype), np.asarray(size, float), np.asarray(pos, float),
                        np.asarray(mat, float).reshape(9), np.asarray(rgba, float))
    g.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
    g.label = label
    scn.ngeom += 1


def _add_label(scn, pos, text, rgb):
    _add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, (0.03, 0.03, 0.03), pos, np.eye(3),
              (rgb[0], rgb[1], rgb[2], 0.9), label=text)


def _add_ring(scn, center, r_in, r_out, rgba, z=0.012, n=64):
    """Draw a flat annulus on the ground as a fan of n thin boxes (conveys the hole)."""
    r_mid, half_rad = 0.5 * (r_in + r_out), 0.5 * (r_out - r_in)
    half_tan = r_mid * (2 * np.pi / n) * 0.6  # slight overlap so the ring reads continuous
    for k in range(n):
        th = k * 2 * np.pi / n
        c, s = np.cos(th), np.sin(th)
        mat = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        _add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, (half_rad, half_tan, 0.01),
                  (center[0] + r_mid * c, center[1] + r_mid * s, z), mat, rgba)


def _add_grasp_arrows(scn, region, anchors, rgba, k=12, L0=0.22):
    """For a quat / point_quat region: sample k grasp orientations and draw, at each anchor,
    an arrow along the gripper approach axis (local z) with the tip at the anchor. Length
    encodes the yaw spin about that axis; directions fan out by the tilt cone."""
    from scipy.spatial.transform import Rotation

    from tamp_ce_mjwarp import regions as _regions

    qs = np.asarray(_regions.sample_uniform(region, np.random.default_rng(0), k))
    qs = qs[:, 3:7] if region.kind == "point_quat" else qs[:, :4]
    anchors = np.atleast_2d(np.asarray(anchors, float))
    if anchors.shape[0] == 1:
        anchors = np.repeat(anchors, k, axis=0)
    seedR = Rotation.from_quat(np.asarray(_regions.topdown_quat())).as_matrix()
    for q, anc in zip(qs, anchors):
        R = Rotation.from_quat(q).as_matrix()
        approach = R[:, 2]  # gripper local z in world (the approach direction)
        yaw = Rotation.from_matrix(seedR.T @ R).as_rotvec()[2]
        length = L0 * (0.45 + 0.55 * min(abs(yaw) / np.pi, 1.0))
        pos = anc - length * approach  # so tip = anchor
        _add_geom(scn, mujoco.mjtGeom.mjGEOM_ARROW, (0.012, 0.012, length), pos, R, rgba)


def _region_colors(n):
    import matplotlib.pyplot as plt
    return plt.cm.tab10(np.linspace(0, 1, max(n, 1)))


def region_home_data(sim, scenario):
    """A forward-evaluated MjData at the home pose (arm posed, graspable object placed where
    the run will start it) plus the grasp anchor xyz the pick-orientation arrows point at."""
    m = sim.mj_model
    d = mujoco.MjData(m)
    d.qpos[sim.arm_qadr] = np.asarray(sim.home_arm)
    if scenario.graspable_xy is not None:
        gq = scenario.graspable_qadr
        d.qpos[gq:gq + 2] = np.asarray(scenario.graspable_xy)
    mujoco.mj_forward(m, d)
    anchor = (d.qpos[scenario.graspable_qadr:scenario.graspable_qadr + 3].copy()
              if scenario.graspable_xy is not None else np.array([0.0, 0.0, 0.3]))
    return d, anchor


def add_region_geoms(scn, scenario, grasp_anchor, n_quat=12, alpha=0.28):
    """Add each sampling region to a render/viewer scene as a translucent primitive +
    grasp/place approach arrows + a floating label. Returns [(name, rgb), ...] for a legend."""
    legend = []
    I = np.eye(3)
    for r, col in zip(scenario.regions, _region_colors(len(scenario.regions))):
        a, b = np.asarray(r.a, float), np.asarray(r.b, float)
        rgb = (float(col[0]), float(col[1]), float(col[2]))
        nm = r.name or r.kind
        if r.kind == "annulus":
            _add_ring(scn, a[:2], float(b[0]), float(b[1]), rgb + (0.45,))
            _add_label(scn, (a[0], a[1] + float(b[1]) + 0.05, 0.05), nm, rgb)
        elif r.kind == "box":  # a,b are world lo/hi corners
            ctr = 0.5 * (a[:2] + b[:2])
            half = 0.5 * (b[:2] - a[:2])
            _add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, (half[0], half[1], 0.02),
                      (ctr[0], ctr[1], 0.02), I, rgb + (alpha,))
            _add_label(scn, (ctr[0], ctr[1], 0.1), nm, rgb)
        elif r.kind in ("point_quat", "point"):  # box: x,y centered on a; z from a_z UP by b_z
            hz = 0.5 * float(b[2])
            _add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, (float(b[0]), float(b[1]), hz),
                      (a[0], a[1], a[2] + hz), I, rgb + (alpha,))
            if r.kind == "point_quat":
                _add_grasp_arrows(scn, r, _sample_region_points(r, n_quat), rgb + (0.95,), k=n_quat)
            _add_label(scn, (a[0], a[1], a[2] + float(b[2]) + 0.05), nm, rgb)
        elif r.kind == "quat":
            _add_grasp_arrows(scn, r, grasp_anchor, rgb + (0.95,), k=n_quat)
            _add_label(scn, (grasp_anchor[0], grasp_anchor[1], grasp_anchor[2] + 0.25), nm, rgb)
        legend.append((nm, rgb))
    return legend


def plot_parameter_regions(scenario, sim, path, view="behind", width=900, height=650, n_quat=12):
    """Offscreen render of the scene with each sampling region drawn as a translucent 3D
    primitive + a matplotlib legend, to sanity-check region sizes/locations before a run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    m = sim.mj_model
    d, grasp_anchor = region_home_data(sim, scenario)
    m.vis.global_.offwidth, m.vis.global_.offheight = width, height
    renderer = mujoco.Renderer(m, height, width, max_geom=20000)
    renderer.update_scene(d, camera=camera(m, view), scene_option=vis_option("visual"))
    legend = add_region_geoms(renderer.scene, scenario, grasp_anchor, n_quat=n_quat)
    rgb = renderer.render()
    renderer.close()

    fig, ax = plt.subplots(figsize=(width / 110, height / 110))
    ax.imshow(rgb)
    ax.axis("off")
    ax.legend(handles=[mpatches.Patch(color=c, label=nm) for nm, c in legend],
              loc="upper right", fontsize=8, framealpha=0.9, title="parameter regions")
    ax.set_title(f"{scenario.name}: parameter regions (arrows = grasp/place orientation)")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _sample_region_points(region, k=12):
    """k sampled xyz points of a point_quat region (the place-point box) for arrow anchors."""
    from tamp_ce_mjwarp import regions as _regions
    return np.asarray(_regions.sample_uniform(region, np.random.default_rng(1), k))[:, :3]


def _ghost(mj_model, datas, scn, opt, pert, alpha):
    """Append the robot+objects of every data in `datas` to `scn` as translucent ghosts
    (only DYNAMIC geoms, so the static scene isn't redrawn N times)."""
    dyn = int(mujoco.mjtCatBit.mjCAT_DYNAMIC)
    for d in datas:
        mujoco.mj_forward(mj_model, d)
        n0 = scn.ngeom
        mujoco.mjv_addGeoms(mj_model, d, opt, pert, dyn, scn)
        for g in range(n0, scn.ngeom):
            scn.geoms[g].rgba[3] *= alpha


def _stamp(frame, label):
    """Draw a text label onto a frame (top-left). Best-effort: returns the frame
    unchanged if PIL is unavailable."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame
    im = Image.fromarray(frame)
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:  # older PIL: no size kwarg
        font = ImageFont.load_default()
    draw.text((11, 9), label, fill=(0, 0, 0), font=font)
    draw.text((10, 8), label, fill=(255, 255, 255), font=font)
    return np.asarray(im)


def render_batch(mj_model, qpos_batch, path, camera=-1, width=640, height=480, fps=None,
                 every=10, alpha=0.45, max_geom=40000, show="visual", dt=None,
                 writer=None, label=None):
    """Superpose a batch of qpos trajectories (N, T, nq) into one render: env 0 is drawn
    solid (with the static scene), every other env as translucent ghosts. Writes an mp4
    (or a single PNG if there's only one frame). With `writer` (an open imageio writer),
    frames stream into it instead — used to concatenate segments into one video. `label`
    stamps a text overlay on every frame."""
    if fps is None:
        fps = (1.0 / (every * dt)) if dt else 30
    qpos_batch = np.asarray(qpos_batch)
    N, T = qpos_batch.shape[:2]
    datas = [mujoco.MjData(mj_model) for _ in range(N)]
    opt, pert = vis_option(show), mujoco.MjvPerturb()
    renderer = mujoco.Renderer(mj_model, height, width, max_geom=max_geom)
    frames = []
    for t in range(0, T, every):
        datas[0].qpos[:] = qpos_batch[0, t]
        mujoco.mj_forward(mj_model, datas[0])
        renderer.update_scene(datas[0], camera=camera, scene_option=opt)
        for d, q in zip(datas[1:], qpos_batch[1:, t]):
            d.qpos[:] = q
        _ghost(mj_model, datas[1:], renderer.scene, opt, pert, alpha)
        frame = renderer.render()
        if label is not None:
            frame = _stamp(frame, label)
        if writer is not None:
            writer.append_data(frame)
        else:
            frames.append(frame)
    renderer.close()
    if writer is not None:
        return path
    if len(frames) == 1:
        imageio.imwrite(path, frames[0])
    else:
        imageio.mimsave(path, frames, fps=fps)
    return path


def play_viewer_batch(mj_model, qpos_batch, dt=0.002, speed=1.0, every=4, alpha=0.45,
                      loop=True, show="visual"):
    """Interactive viewer replay of a batch (N, T, nq): env 0 live, others as ghosts in the
    user scene each frame (capped by the user-scene geom budget). Needs a display."""
    import time

    import mujoco.viewer

    qpos_batch = np.asarray(qpos_batch)
    N, T = qpos_batch.shape[:2]
    data = mujoco.MjData(mj_model)
    ghosts = [mujoco.MjData(mj_model) for _ in range(N - 1)]
    opt, pert = vis_option(show), mujoco.MjvPerturb()
    with mujoco.viewer.launch_passive(mj_model, data) as v:
        _apply_show(v.opt, show)
        scn = v.user_scn
        while v.is_running():
            for t in range(0, T, every):
                if not v.is_running():
                    break
                data.qpos[:] = qpos_batch[0, t]
                mujoco.mj_forward(mj_model, data)
                scn.ngeom = 0
                for g, q in zip(ghosts, qpos_batch[1:, t]):
                    g.qpos[:] = q
                _ghost(mj_model, ghosts, scn, opt, pert, alpha)
                v.sync()
                time.sleep(dt * every / speed)
            if not loop:
                break


def play_viewer(mj_model, qpos_traj, dt=0.002, speed=1.0, every=4, loop=True, show="visual"):
    """Replay a qpos trajectory in the interactive MuJoCo viewer (needs a display). `show`
    sets the initial visual/collision/both toggle (also switchable live with keys 2/4)."""
    import time

    import mujoco.viewer

    qpos_traj = np.asarray(qpos_traj)
    data = mujoco.MjData(mj_model)
    with mujoco.viewer.launch_passive(mj_model, data) as v:
        _apply_show(v.opt, show)
        while v.is_running():
            for i in range(0, len(qpos_traj), every):
                if not v.is_running():
                    break
                data.qpos[:] = qpos_traj[i]
                mujoco.mj_forward(mj_model, data)
                v.sync()
                time.sleep(dt * every / speed)
            if not loop:
                break


def _draw_regions(ax, plt, regions):
    """Overlay the xy projection of the sampling regions, faintly, with a legend entry.
    Quaternion-only regions have no xy footprint and are skipped."""
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(regions), 1)))
    for r, col in zip(regions, colors):
        a, b, nm = np.asarray(r.a), np.asarray(r.b), r.name or r.kind
        if r.kind == "annulus":
            ax.add_patch(plt.Circle(a[:2], b[1], color=col, alpha=0.12, zorder=0, label=nm))
            ax.add_patch(plt.Circle(a[:2], b[0], color="white", alpha=1.0, zorder=0))
        elif r.kind == "box":
            ax.add_patch(plt.Rectangle(a[:2], *(b[:2] - a[:2]), color=col, alpha=0.15,
                                       zorder=0, label=nm))
        elif r.kind in ("point_quat", "point"):
            ax.add_patch(plt.Rectangle(a[:2] - b[:2], *(2 * b[:2]), color=col, alpha=0.15,
                                       zorder=0, label=nm))


def plot_paths(scenario, pop, path, it=None, draw_ee="best", draw_block="best", every=15):
    """Top-down plot of a CE population: region footprints + obstacles + 2D base paths.

    pop: {cost, ok, tracks, best_idx}; tracks (T, N, 6) = base_xy, ee_xy, block_xy per step.
    draw_ee / draw_block: "best" (only the best env), "all", or None."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = np.asarray(pop["ok"]).astype(bool)
    tr = np.asarray(pop["tracks"]).transpose(1, 0, 2)[:, ::every]  # (N, t, 6)
    best_idx = pop.get("best_idx")
    fig, ax = plt.subplots(figsize=(7, 7))

    # axis limits = the nav grid (the base's reachable region); else the track extent
    if scenario.nav is not None:
        xmin, xmax, ymin, ymax = scenario.nav[1]
    else:
        xmin, xmax = tr[..., 0].min() - 0.3, tr[..., 0].max() + 0.3
        ymin, ymax = tr[..., 1].min() - 0.3, tr[..., 1].max() + 0.3

    _draw_regions(ax, plt, scenario.regions)

    # inflated region (gray) then raw obstacle footprint (black)
    for rx0, rx1, ry0, ry1, inf in scenario.obstacles:
        ax.add_patch(plt.Rectangle((rx0 - inf, ry0 - inf), rx1 - rx0 + 2 * inf,
                                   ry1 - ry0 + 2 * inf, color="0.8", zorder=1))
    for rx0, rx1, ry0, ry1, _ in scenario.obstacles:
        ax.add_patch(plt.Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0, color="0.15", zorder=2))

    # base paths: successes faint blue, failures fainter red, best bold
    for i in range(tr.shape[0]):
        c, a, lw = ("tab:blue", 0.3, 0.8) if ok[i] else ("tab:red", 0.25, 0.6)
        ax.plot(tr[i, :, 0], tr[i, :, 1], color=c, alpha=a, lw=lw, zorder=3)

    def _overlay(col, style, label):
        idxs = ([best_idx] if (style == "best" and best_idx is not None)
                else (np.where(ok)[0] if style == "all" else []))
        for j, i in enumerate(idxs):
            ax.plot(tr[i, :, col], tr[i, :, col + 1], "--", color=label[0],
                    alpha=0.9 if style == "best" else 0.2, lw=1.3,
                    label=label[1] if j == 0 else None, zorder=4)

    _overlay(2, draw_ee, ("tab:green", "EE"))
    _overlay(4, draw_block, ("tab:orange", "block"))

    if best_idx is not None:
        ax.plot(tr[best_idx, :, 0], tr[best_idx, :, 1], color="black", lw=2.2,
                label="best base", zorder=5)

    ax.scatter([tr[0, 0, 0]], [tr[0, 0, 1]], c="k", marker="s", s=40, zorder=6, label="start")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    title = f"{scenario.name}: {int(ok.sum())}/{len(ok)} reached goal"
    if it is not None:
        title = f"iter {it} — " + title
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_value_field(scenario, goal_xy, path, start_xy=None, stride=2):
    """Debug the base nav-function: heatmap of the VI cost-to-go field to `goal_xy`, the
    obstacle footprints (+inflation), the blurred descent directions (quiver), the goal, and
    an optional start. Unreachable cells render blank. Needs scenario.nav."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tamp_ce_mjwarp import nav as _nav

    occ, bounds, res, iters, remap = scenario.nav
    xmin, xmax, ymin, ymax = bounds
    H, W = occ.shape
    cx, cy = _nav.world_to_cell(np.asarray(goal_xy, float), bounds, res)
    gx = int(np.clip(round(float(cx)), 0, W - 1))
    gy = int(np.clip(round(float(cy)), 0, H - 1))
    V = _nav.value_field(occ, (gx, gy), iters, remap)
    sx, sy = int(remap[0][gy, gx]), int(remap[1][gy, gx])  # snapped goal cell
    dgx, dgy = _nav.value_grad(V)
    blank = V >= _nav.BIG * 0.5  # unreachable / obstacle cells
    V = np.where(blank, np.nan, V)

    fig, ax = plt.subplots(figsize=(7, 8))
    import matplotlib as mpl
    cmap = mpl.cm.viridis.copy()
    cmap.set_bad("0.75")  # NaN -> gray, not white, so quivers stay visible
    im = ax.imshow(V, origin="lower", extent=(xmin, xmax, ymin, ymax), cmap=cmap,
                   aspect="equal", zorder=0)
    fig.colorbar(im, ax=ax, label="cost-to-go [cells]", shrink=0.7)

    for rx0, rx1, ry0, ry1, inf in scenario.obstacles:  # inflation (dashed) + raw (black)
        ax.add_patch(plt.Rectangle((rx0 - inf, ry0 - inf), rx1 - rx0 + 2 * inf,
                                   ry1 - ry0 + 2 * inf, fill=False, edgecolor="tab:red",
                                   ls="--", lw=1.2, zorder=2))
        ax.add_patch(plt.Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0, color="0.15", zorder=2))

    xs = xmin + (np.arange(W) + 0.5) * res
    ys = ymin + (np.arange(H) + 0.5) * res
    gxx, gyy = np.meshgrid(xs, ys)
    qx, qy = np.where(blank, np.nan, dgx), np.where(blank, np.nan, dgy)
    s = slice(None, None, stride)
    ax.quiver(gxx[s, s], gyy[s, s], qx[s, s], qy[s, s], color="black", alpha=0.7,
              scale=40, width=0.0025, zorder=3)

    ax.scatter([goal_xy[0]], [goal_xy[1]], c="red", marker="*", s=180, zorder=4, label="goal")
    if (sx, sy) != (gx, gy):  # goal was inside an obstacle -> snapped to the boundary
        swx = xmin + (sx + 0.5) * res
        swy = ymin + (sy + 0.5) * res
        ax.scatter([swx], [swy], facecolors="none", edgecolors="red", marker="o", s=180,
                   linewidths=2, zorder=5, label="snapped goal")
        ax.plot([goal_xy[0], swx], [goal_xy[1], swy], "r--", lw=1, zorder=5)
    if start_xy is not None:
        ax.scatter([start_xy[0]], [start_xy[1]], c="cyan", marker="s", s=60, zorder=4,
                   label="start")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{scenario.name}: VI cost-to-go (blank = unreachable)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _write_param_group(parent, name, params):
    """Store a ragged list of per-action parameter vectors as datasets p0, p1, ... ."""
    g = parent.create_group(name)
    for i, p in enumerate(params):
        g.create_dataset(f"p{i}", data=np.asarray(p))


def _read_param_group(g):
    return [g[k][()] for k in sorted(g, key=lambda s: int(s[1:]))]


def save_solution(path, params, cost, meta=None, iters=None):
    """Write a solution as HDF5: the global-best per-action parameter vectors (group
    `params/p{i}`) plus `meta` (run config + per-iter timing/success). If `iters` is given
    (one dict per CE iteration), every iteration's best plan AND its refitted sampling
    distribution (mean/std) are stored under `iters/<NNN>/`."""
    meta = dict(meta or {})
    with h5py.File(path, "w") as f:
        f.attrs["cost"] = float(cost)
        for k, v in meta.items():
            if isinstance(v, (list, tuple)):
                if len(v) and isinstance(v[0], str):
                    f.create_dataset(k, data=np.array(v, dtype=h5py.string_dtype()))
                else:  # numeric (may contain None, e.g. best cost in a 0-success iter)
                    f.create_dataset(k, data=np.array([np.nan if x is None else x for x in v],
                                                      dtype=float))
            else:
                f.attrs[k] = v
        _write_param_group(f, "params", params)
        if iters:
            gi = f.create_group("iters")
            for it, rec in enumerate(iters):
                gg = gi.create_group(f"{it:03d}")
                gg.attrs["cost"] = float(rec.get("cost", np.nan))
                gg.attrs["n_goal"] = int(rec.get("n_goal", 0))
                for key in ("params", "mean", "std"):
                    _write_param_group(gg, key, rec[key])
                if "elite_cost" in rec:
                    gg.create_dataset("elite_cost", data=np.asarray(rec["elite_cost"]))
                if "elite_by_goal" in rec:
                    gg.attrs["elite_by_goal"] = bool(rec["elite_by_goal"])
    return path


def load_solution(path):
    """Returns (params, cost, meta) for the global-best plan."""
    with h5py.File(path, "r") as f:
        params = _read_param_group(f["params"])
        cost = float(f.attrs["cost"])
        meta = {k: f.attrs[k] for k in f.attrs if k != "cost"}
        for k in f:
            if k in ("params", "iters"):
                continue
            d = f[k][()]
            meta[k] = [s.decode() if isinstance(s, bytes) else s for s in d] \
                if d.dtype.kind in "OS" else d
    return params, cost, meta


def load_iter_params(path, it=None):
    """Best plan parameters for one CE iteration (`it` supports negatives; default last).
    Returns (params, info); falls back to the global-best params if no per-iter records."""
    with h5py.File(path, "r") as f:
        if "iters" not in f:
            return _read_param_group(f["params"]), {"iter": None, "cost": float(f.attrs["cost"])}
        gi = f["iters"]
        keys = sorted(gi, key=int)
        key = keys[-1 if it is None else it]
        gg = gi[key]
        info = {"iter": int(key), "n_iters": len(keys),
                "cost": float(gg.attrs["cost"]), "n_goal": int(gg.attrs["n_goal"])}
        return _read_param_group(gg["params"]), info


def load_iter_dist(path, it=None):
    """Refitted sampling distribution (per-action (mean, std) list) for one CE iteration."""
    with h5py.File(path, "r") as f:
        gi = f["iters"]
        key = sorted(gi, key=int)[-1 if it is None else it]
        gg = gi[key]
        return list(zip(_read_param_group(gg["mean"]), _read_param_group(gg["std"])))


def load_elite_costs(path):
    """Per-iteration elite step-cost vectors and the selection tier used. Returns
    (costs, iters, by_goal); empty lists if the file recorded none."""
    with h5py.File(path, "r") as f:
        if "iters" not in f:
            return [], [], []
        gi = f["iters"]
        keys = [k for k in sorted(gi, key=int) if "elite_cost" in gi[k]]
        costs = [gi[k]["elite_cost"][()] for k in keys]
        by_goal = [bool(gi[k].attrs.get("elite_by_goal", gi[k].attrs["n_goal"] > 0)) for k in keys]
        return costs, [int(k) for k in keys], by_goal


def plot_elite_costs(path, out=None):
    """Violin of the elite rollout-cost distribution per CE iteration, colored by the
    selection tier (goal-reachers vs proxy fallback). Shows the elite front dropping and
    tightening as CE converges. No-op (returns None) if no elite costs recorded."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    costs, iters, by_goal = load_elite_costs(path)
    if not costs:
        return None
    GOAL_C, PROXY_C = "tab:green", "tab:orange"
    cols = [GOAL_C if g else PROXY_C for g in by_goal]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8, 5))

    # violins only where a KDE is well-defined; singletons rely on points
    vio = [i for i, c in enumerate(costs) if len(c) > 1 and np.ptp(c) > 0]
    if vio:
        parts = ax.violinplot([costs[i] for i in vio], positions=[iters[i] for i in vio],
                              showextrema=False, widths=0.7)
        for b, i in zip(parts["bodies"], vio):
            b.set_facecolor(cols[i])
            b.set_alpha(0.25)
            b.set_zorder(1)

    for it, c, col in zip(iters, costs, cols):  # individual elites, x-jittered
        ax.scatter(it + rng.uniform(-0.13, 0.13, size=len(c)), c, s=7, color=col,
                   alpha=0.6, edgecolors="none", zorder=3)

    ml, = ax.plot(iters, [np.median(c) for c in costs], "-o", color="tab:blue",
                  label="elite median", zorder=4)
    bl, = ax.plot(iters, [np.min(c) for c in costs], "-o", color="tab:red",
                  label="elite best", zorder=4)
    tiers = ([mpatches.Patch(color=GOAL_C, alpha=0.5, label="elites: goal-reachers (by cost)")]
             if any(by_goal) else []) + \
            ([mpatches.Patch(color=PROXY_C, alpha=0.5, label="elites: proxy fallback")]
             if not all(by_goal) else [])

    ax.legend(handles=[*tiers, ml, bl], fontsize=8)
    ax.set_xlabel("CE iteration")
    ax.set_ylabel("rollout cost [steps]")
    ax.set_title("Elite cost distribution per iteration")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = out or os.path.splitext(path)[0] + "_elite.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def load_iter_history(path):
    """Full per-iteration distribution history: (means, stds, iters), where means/stds are
    lists (one per action) of (n_iter, dim) arrays."""
    with h5py.File(path, "r") as f:
        gi = f["iters"]
        keys = sorted(gi, key=int)
        n_act = len(_read_param_group(gi[keys[0]]["mean"]))
        means = [np.stack([_read_param_group(gi[k]["mean"])[a] for k in keys])
                 for a in range(n_act)]
        stds = [np.stack([_read_param_group(gi[k]["std"])[a] for k in keys])
                for a in range(n_act)]
    return means, stds, [int(k) for k in keys]


# Per-region-kind labels for the parameter dimensions (for the diagnostics legend).
_DIM_LABELS = {"quat": ["qx", "qy", "qz", "qw"],
               "point_quat": ["x", "y", "z", "qx", "qy", "qz", "qw"],
               "annulus": ["x", "y"]}


def _dim_labels(kind, d):
    if kind in _DIM_LABELS:
        return _DIM_LABELS[kind]
    return (["x", "y", "z"][:d] if d <= 3 else [f"d{i}" for i in range(d)])


def plot_ce_diagnostics(path, out=None):
    """CE convergence diagnostics for a saved solution: best goal cost + success rate per
    iteration, and each action's sampling-distribution evolution (per-dim mean ± std)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, _, meta = load_solution(path)
    names, kinds = meta.get("param_names"), meta.get("param_kinds")
    n = int(meta.get("n", 1))
    cost = np.asarray(meta.get("best_cost_per_iter", []), float)
    succ = np.asarray(meta.get("goal_success_per_iter", []), float)
    means, stds, iters = load_iter_history(path)
    iters = np.asarray(iters)
    n_act = len(means)

    fig = plt.figure(figsize=(11, 3 * (1 + n_act)))
    gs = fig.add_gridspec(1 + n_act, 2)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(np.arange(len(cost)), cost, "-o", color="tab:red")
    ax.set_xlabel("CE iteration")
    ax.set_ylabel("best goal cost [steps]")
    ax.set_title("Cost convergence (gaps = no goal reached)")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(np.arange(len(succ)), 100 * succ / n, "-o", color="tab:green")
    ax.set_xlabel("CE iteration")
    ax.set_ylabel("rollouts reaching goal [%]")
    ax.set_title(f"Success rate (n={n})")
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    for a in range(n_act):
        ax = fig.add_subplot(gs[1 + a, :])
        m, s = means[a], stds[a]  # (n_iter, dim)
        labels = _dim_labels(kinds[a] if kinds else "", m.shape[1])
        colors = plt.cm.tab10(np.linspace(0, 1, max(m.shape[1], 1)))
        for j in range(m.shape[1]):
            ax.plot(iters, m[:, j], color=colors[j], label=labels[j])
            ax.fill_between(iters, m[:, j] - s[:, j], m[:, j] + s[:, j],
                            color=colors[j], alpha=0.15)
        nm = names[a] if names else f"action {a}"
        ax.set_title(f"{nm} — sampling distribution (mean ± std)")
        ax.set_xlabel("CE iteration")
        ax.set_ylabel("param value")
        ax.legend(fontsize=7, ncol=4, loc="best")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out = out or os.path.splitext(path)[0] + "_ce.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out

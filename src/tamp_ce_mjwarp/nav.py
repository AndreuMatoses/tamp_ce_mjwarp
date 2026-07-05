"""Value-iteration navigation function for the mobile base, batched over per-world goals.

A shared static occupancy grid is built once per scene (host numpy). Per CE iteration,
`NavBatch.compute` runs one captured CUDA graph per move action: fixed-iteration VI
relaxation (one field per world's goal) followed by the fill/gradient/blur kernels that
produce the smoothed descent grid `gxy` for the per-step bilinear lookup inside the move
control kernel. Everything stays on device (the host `value_grad` remains as the
reference implementation and for plots/tests).

Convention: grids are indexed [row=iy, col=ix]; world x -> column, world y -> row.
`bounds = (xmin, xmax, ymin, ymax)`. A scenario's nav tuple is
(occ, bounds, res, n_iters, remap) with `remap` from `nearest_free`.
"""

from __future__ import annotations

import numpy as np
import warp as wp

BIG = 1e6
_SQRT2 = float(np.sqrt(2.0))


def grid_shape(bounds, res):
    xmin, xmax, ymin, ymax = bounds
    return int(np.ceil((ymax - ymin) / res)), int(np.ceil((xmax - xmin) / res))  # (H, W)


def occupancy_from_rects(bounds, res, rects):
    """Boolean (H, W) occupancy from axis-aligned rects (xmin,xmax,ymin,ymax,inflate),
    each padded by its own `inflate`."""
    xmin, _, ymin, _ = bounds
    H, W = grid_shape(bounds, res)
    occ = np.zeros((H, W), dtype=bool)
    xs = xmin + (np.arange(W) + 0.5) * res
    ys = ymin + (np.arange(H) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys)
    for rx0, rx1, ry0, ry1, inflate in rects:
        occ |= ((gx >= rx0 - inflate) & (gx <= rx1 + inflate) &
                (gy >= ry0 - inflate) & (gy <= ry1 + inflate))
    return occ


def world_to_cell(pos_xy, bounds, res):
    cx = (pos_xy[..., 0] - bounds[0]) / res
    cy = (pos_xy[..., 1] - bounds[2]) / res
    return cx, cy  # fractional (col, row)


def nearest_free(occ):
    """Per-cell map to the nearest FREE cell (a cell is its own target if free), so a goal
    sampled inside an obstacle can be snapped to the obstacle boundary before VI — otherwise
    the field pins a zero-cost sink inside the obstacle and the base is driven straight
    through it. Returns (remap_x, remap_y) int (H, W) arrays with the nearest free (col, row)."""
    from scipy import ndimage
    occ = np.asarray(occ)
    if not occ.any():
        H, W = occ.shape
        return (np.broadcast_to(np.arange(W), (H, W)).copy(),
                np.broadcast_to(np.arange(H)[:, None], (H, W)).copy())
    row, col = ndimage.distance_transform_edt(occ, return_distances=False, return_indices=True)
    return col, row


def snap_goals(goals_xy, occ, bounds, res, remap):
    """Goal cells (n, 2) int32 (col, row) for world goals (n, 2), clipped to the grid and
    snapped to the nearest free cell."""
    H, W = occ.shape
    cx, cy = world_to_cell(np.asarray(goals_xy), bounds, res)
    gx = np.clip(np.round(cx).astype(int), 0, W - 1)
    gy = np.clip(np.round(cy).astype(int), 0, H - 1)
    rx, ry = remap
    return np.stack([rx[gy, gx], ry[gy, gx]], axis=1).astype(np.int32)


@wp.kernel
def _vi_init(occ: wp.array2d(dtype=wp.int32), goal: wp.array2d(dtype=wp.int32),
             V: wp.array3d(dtype=wp.float32)):
    i, y, x = wp.tid()
    v = BIG
    if x == goal[i, 0] and y == goal[i, 1]:
        v = 0.0
    V[i, y, x] = v


@wp.kernel
def _vi_relax(occ: wp.array2d(dtype=wp.int32), goal: wp.array2d(dtype=wp.int32),
              V_in: wp.array3d(dtype=wp.float32), V_out: wp.array3d(dtype=wp.float32)):
    i, y, x = wp.tid()
    if occ[y, x] != 0:
        V_out[i, y, x] = BIG
        return
    if x == goal[i, 0] and y == goal[i, 1]:
        V_out[i, y, x] = 0.0
        return
    H, W = occ.shape[0], occ.shape[1]
    v = V_in[i, y, x]
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            yn, xn = y + dy, x + dx
            if (dy != 0 or dx != 0) and yn >= 0 and yn < H and xn >= 0 and xn < W:
                c = wp.where(dy != 0 and dx != 0, _SQRT2, 1.0)
                v = wp.min(v, V_in[i, yn, xn] + c)
    V_out[i, y, x] = wp.min(v, BIG)


@wp.kernel
def _grad_fill(V: wp.array3d(dtype=wp.float32), Vf: wp.array3d(dtype=wp.float32)):
    """Fill BIG cells (obstacle/unreachable) from their smallest 3x3 neighbour, so the
    descent gradient has no BIG cliff pushing it off walls (GPU port of _nbr_min)."""
    i, y, x = wp.tid()
    v = V[i, y, x]
    if v < BIG * 0.5:
        Vf[i, y, x] = v
        return
    H, W = V.shape[1], V.shape[2]
    m = BIG
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            yn, xn = y + dy, x + dx
            if yn >= 0 and yn < H and xn >= 0 and xn < W:
                m = wp.min(m, V[i, yn, xn])
    Vf[i, y, x] = m


@wp.kernel
def _grad_unit(Vf: wp.array3d(dtype=wp.float32), g: wp.array3d(dtype=wp.vec2)):
    """Unit descent direction (-d/dcol, -d/drow): central differences inside, one-sided
    at the grid border (matches np.gradient)."""
    i, y, x = wp.tid()
    H, W = Vf.shape[1], Vf.shape[2]
    xm, xp = wp.max(x - 1, 0), wp.min(x + 1, W - 1)
    ym, yp = wp.max(y - 1, 0), wp.min(y + 1, H - 1)
    gx = -(Vf[i, y, xp] - Vf[i, y, xm]) / float(xp - xm)
    gy = -(Vf[i, yp, x] - Vf[i, ym, x]) / float(yp - ym)
    mag = wp.sqrt(gx * gx + gy * gy) + 1e-9
    g[i, y, x] = wp.vec2(gx / mag, gy / mag)


@wp.kernel
def _grad_blur(V: wp.array3d(dtype=wp.float32), src: wp.array3d(dtype=wp.vec2),
               dst: wp.array3d(dtype=wp.vec2)):
    """One masked 3x3 box-blur pass over free cells (edge-padded via index clamping;
    non-free cells pass through). GPU port of the value_grad blur loop."""
    i, y, x = wp.tid()
    if V[i, y, x] >= BIG * 0.5:
        dst[i, y, x] = src[i, y, x]
        return
    H, W = V.shape[1], V.shape[2]
    s = wp.vec2(0.0, 0.0)
    den = float(0.0)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            yn = wp.clamp(y + dy, 0, H - 1)
            xn = wp.clamp(x + dx, 0, W - 1)
            if V[i, yn, xn] < BIG * 0.5:
                s += src[i, yn, xn]
                den += 1.0
    dst[i, y, x] = s / (den + 1e-9)


@wp.func
def descent_dir(gxy: wp.array3d(dtype=wp.vec2), i: int, pos: wp.vec2,
                xmin: float, ymin: float, res: float) -> wp.vec2:
    """Bilinearly interpolated descent direction at a continuous world position (smooth,
    no snapping between the 8 grid directions)."""
    H, W = gxy.shape[1], gxy.shape[2]
    cx = (pos[0] - xmin) / res
    cy = (pos[1] - ymin) / res
    x0 = wp.clamp(int(wp.floor(cx)), 0, W - 1)
    y0 = wp.clamp(int(wp.floor(cy)), 0, H - 1)
    x1 = wp.min(x0 + 1, W - 1)
    y1 = wp.min(y0 + 1, H - 1)
    fx = wp.clamp(cx - float(x0), 0.0, 1.0)
    fy = wp.clamp(cy - float(y0), 0.0, 1.0)
    g00, g01, g10, g11 = gxy[i, y0, x0], gxy[i, y0, x1], gxy[i, y1, x0], gxy[i, y1, x1]
    return (g00 * (1.0 - fx) + g01 * fx) * (1.0 - fy) + (g10 * (1.0 - fx) + g11 * fx) * fy


class NavBatch:
    """Device-side batched VI: one cost-to-go field per world's goal, relaxed `n_iters`
    times, then filled/differentiated/blurred into the smoothed gradient field
    `self.gxy` — all inside one captured CUDA graph (no host round-trip)."""

    BLUR = 3  # box-blur passes over the unit direction field (mirrors value_grad)

    def __init__(self, nav, n):
        occ, self.bounds, self.res, self.n_iters, self.remap = nav
        self.occ = np.asarray(occ)
        H, W = self.occ.shape
        self.occ_wp = wp.array(self.occ.astype(np.int32), dtype=wp.int32)
        self.goal = wp.zeros((n, 2), dtype=wp.int32)
        self.V = [wp.zeros((n, H, W), dtype=wp.float32) for _ in range(2)]
        self.Vf = wp.zeros((n, H, W), dtype=wp.float32)
        self.gtmp = wp.zeros((n, H, W), dtype=wp.vec2)
        self.gxy = wp.zeros((n, H, W), dtype=wp.vec2)
        self._n, self._graph = n, None

    def _vi(self):
        n, (H, W) = self._n, self.occ.shape
        wp.launch(_vi_init, dim=(n, H, W), inputs=[self.occ_wp, self.goal, self.V[0]])
        for it in range(self.n_iters):
            src, dst = self.V[it % 2], self.V[(it + 1) % 2]
            wp.launch(_vi_relax, dim=(n, H, W), inputs=[self.occ_wp, self.goal, src, dst])
        return self.V[self.n_iters % 2]

    def _pipeline(self):
        n, (H, W) = self._n, self.occ.shape
        V = self._vi()
        wp.launch(_grad_fill, dim=(n, H, W), inputs=[V, self.Vf])
        # BLUR is odd: gtmp -> gxy -> gtmp -> gxy leaves the final field in gxy
        wp.launch(_grad_unit, dim=(n, H, W), inputs=[self.Vf, self.gtmp])
        src, dst = self.gtmp, self.gxy
        for _ in range(self.BLUR):
            wp.launch(_grad_blur, dim=(n, H, W), inputs=[V, src, dst])
            src, dst = dst, src

    def compute(self, goals_xy):
        """Fill `self.gxy` with the smoothed descent field for per-world goals (n, 2)."""
        self.goal.assign(snap_goals(goals_xy, self.occ, self.bounds, self.res, self.remap))
        if self._graph is None:
            self._pipeline()  # warm-up compile, then capture
            with wp.ScopedCapture() as cap:
                self._pipeline()
            self._graph = cap.graph
        wp.capture_launch(self._graph)


def _box3(a):
    """3x3 box sum over the last two axes (edge-padded)."""
    ap = np.pad(a, [(0, 0)] * (a.ndim - 2) + [(1, 1), (1, 1)], mode="edge")
    H, W = a.shape[-2:]
    return sum(ap[..., i:i + H, j:j + W] for i in range(3) for j in range(3))


def _nbr_min(a):
    """3x3 min (padded with BIG): fills a cell from its smallest neighbour."""
    ap = np.pad(a, [(0, 0)] * (a.ndim - 2) + [(1, 1), (1, 1)], constant_values=BIG)
    H, W = a.shape[-2:]
    return np.stack([ap[..., i:i + H, j:j + W] for i in range(3) for j in range(3)]).min(axis=0)


def value_grad(V, blur=3):
    """Unit descent-direction grids (-d/dcol, -d/drow) of cost-to-go fields V (.., H, W).

    Two boundary cleanups so the base doesn't oscillate or weave: obstacle cells are first
    filled from their smallest neighbour (no BIG cliff pushing the gradient off the wall),
    and the unit direction field is box-blurred over free cells (the 8-connected field's
    gradient direction quantizes toward the grid directions)."""
    occ = V >= BIG * 0.5
    Vf = np.where(occ, _nbr_min(V), V)
    gy = np.gradient(Vf, axis=-2)
    gx = np.gradient(Vf, axis=-1)
    gx, gy = -gx, -gy
    mag = np.sqrt(gx * gx + gy * gy) + 1e-9
    gx, gy = gx / mag, gy / mag
    free = (~occ).astype(V.dtype)
    den = _box3(free)
    for _ in range(blur):
        gx = np.where(free > 0, _box3(gx * free) / (den + 1e-9), gx)
        gy = np.where(free > 0, _box3(gy * free) / (den + 1e-9), gy)
    return gx, gy


def value_field(occ, goal_cell, n_iters, remap=None):
    """Single-goal host VI (for plots): cost-to-go (H, W) for goal_cell=(ix, iy)."""
    occ = np.asarray(occ)
    H, W = occ.shape
    gx, gy = int(goal_cell[0]), int(goal_cell[1])
    if remap is not None:
        rx, ry = remap
        gx, gy = int(rx[gy, gx]), int(ry[gy, gx])
    V = np.full((H, W), BIG, dtype=np.float64)
    V[gy, gx] = 0.0
    Vp = np.full((H + 2, W + 2), BIG)
    for _ in range(n_iters):
        Vp[1:-1, 1:-1] = V
        cands = [V] + [Vp[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx] + c
                       for dy, dx, c in [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                                         (-1, -1, _SQRT2), (-1, 1, _SQRT2),
                                         (1, -1, _SQRT2), (1, 1, _SQRT2)]]
        V = np.minimum(np.stack(cands).min(axis=0), BIG)
        V[occ] = BIG
        V[gy, gx] = 0.0
    return V

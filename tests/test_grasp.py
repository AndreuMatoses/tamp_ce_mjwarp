"""Batched grasp validation through the real rollout path: the phased pick action at many
randomized cube positions in parallel, with exact top-down grasps. This is the physics
canary for the MJWarp port (float32 + implicit integrator vs the MJX baseline): the same
setup holds ~100% grasp success at dt=0.003, solver iters 20/10.

    uv run python tests/test_grasp.py
"""

import numpy as np

from tamp_ce_mjwarp.ik import IK
from tamp_ce_mjwarp.plan import ActionSpec, Rollout
from tamp_ce_mjwarp.regions import topdown_quat
from tamp_ce_mjwarp.scenarios import Scenario
from tamp_ce_mjwarp.sim import Sim

CUBE_QADR = 11


def run(n=16, seed=0, verbose=True):
    sim = Sim("robot_models/dinova/dinova_block_scene.xml")
    ik = IK(n)
    scen = Scenario(name="grasp_probe",
                    scene_xml="robot_models/dinova/dinova_block_scene.xml",
                    regions=[], specs=[ActionSpec("pick", timeout=7.5,
                                                  obj_qadr=CUBE_QADR, z_off=0.01)],
                    graspable_qadr=CUBE_QADR, graspable_xy=(0.46, 0.0))
    ro = Rollout(sim, ik, scen, n)

    # randomize the cube xy per world (within arm reach of the base at the origin)
    rng = np.random.default_rng(seed)
    cubes = np.stack([rng.uniform(0.40, 0.52, n), rng.uniform(-0.12, 0.12, n)], axis=1)
    qpos0 = ro._pristine["qpos"].numpy()
    qpos0[:, CUBE_QADR:CUBE_QADR + 2] = cubes
    ro._pristine["qpos"].assign(qpos0)

    params = [np.tile(topdown_quat(), (n, 1))]  # exact top-down grasp for every world
    out = ro.run(params)
    lifted = out["qpos"][:, CUBE_QADR + 2] - 0.02
    ok = (out["n_act"] >= 1) & (lifted > 0.05)
    if verbose:
        for i in range(n):
            print(f"  cube=({cubes[i, 0]:.2f},{cubes[i, 1]:+.2f}) lifted={lifted[i]:+.3f} "
                  f"held={int(out['n_act'][i])} cost={out['cost'][i]:.0f} "
                  f"{'OK' if ok[i] else 'FAIL'}", flush=True)
        print(f"success: {ok.sum()}/{n} ({100 * ok.mean():.0f}%)", flush=True)
    return ok.mean()


def test_grasp_success_rate():
    # cubes at the near edge of the workspace (x~0.40) are marginal for a top-down grasp
    # (bad DLS basin); the MJX repo's grasp test asserted the same >= 0.8
    assert run(n=16, verbose=False) >= 0.8


if __name__ == "__main__":
    run(n=64)

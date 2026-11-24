"""Simulated SO-101 arm in MuJoCo, with a side camera like the real setup.

The arm model is the maker's own MJCF (TheRobotStudio/SO-ARM100, Simulation/SO101), loaded unchanged;
the table, light and camera are added here in code. Its "new calibration" puts every joint's zero at the
middle of its range, which is the same convention as the degrees logged by arm_collect.py, and the signs
match the real arm (decreasing elbow_flex raises the gripper, increasing wrist_flex lowers it). The
gripper is the exception: here it is the jaw angle in degrees (-10 closed .. 100 open).

The servos are position actuators with no integral term. The model file ships a stiff gain (998 N m/rad,
0.02 deg of sag); the default here is the maker's measured STS3215 value from joints_properties.xml
(17.8), so a loaded joint stops about 1 deg short of its goal, as the real one does (about 2 deg).
Log measured angles, never goals.

Setup on the desktop (once):
    uv pip install --python ~/projects/lerobot/.venv/bin/python mujoco
    git clone --depth 1 --filter=blob:none --sparse https://github.com/TheRobotStudio/SO-ARM100.git \
        ~/projects/sim_assets/SO-ARM100 && git -C ~/projects/sim_assets/SO-ARM100 sparse-checkout set Simulation/SO101

    from local.sim_arm import SimArm
    arm = SimArm()
    arm.reset(arm.ready)
    arm.step(goal_deg)          # advance one camera frame (1/30 s)
    q, img = arm.q_deg, arm.render()
"""

import os
import sys

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering on the GPU; must be set before the import

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ASSETS = os.environ.get("SIM_ASSETS", os.path.expanduser("~/projects/sim_assets"))
SO101_XML = "SO-ARM100/Simulation/SO101/so101_new_calib.xml"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# The real arm's "table" ready pose (arm_collect.READY_POSES) in degrees, gripper closed.
READY_DEG = [0.0, -21.4, 66.9, -37.0, 0.0, -10.0]
# Side view across the arm, base on the right, looking slightly down: the real camera placement.
CAMERA = {"lookat": [0.17, 0.0, 0.09], "distance": 0.62, "azimuth": -90.0, "elevation": -25.0, "fovy": 32.0}
SUBSTEPS = 20


def build_model(assets=ASSETS, fps=30.0, camera=None, kp=17.8, arm_rgba=(0.08, 0.08, 0.09, 1)):
    """The maker's SO-101 plus table, lamp and side-camera settings. Returns (model, camera dict)."""
    spec = mujoco.MjSpec.from_file(os.path.join(assets, SO101_XML))
    if arm_rgba is not None:  # the real arm is printed in black
        for mat in spec.materials:
            mat.rgba = arm_rgba
    spec.add_texture(
        name="sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
        rgb1=[0.75, 0.75, 0.75],
        rgb2=[0.75, 0.75, 0.75],
        width=64,
        height=384,
    )
    world = spec.worldbody
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], pos=[0, 0, -0.02],
                   rgba=[0.55, 0.55, 0.55, 1], contype=0, conaffinity=0)
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.35, 0.3, 0.01], pos=[0.2, 0, -0.01],
                   rgba=[0.9, 0.88, 0.84, 1], contype=0, conaffinity=0)
    world.add_light(name="lamp", pos=[0.2, -0.3, 1.2], dir=[0, 0.2, -1], castshadow=True)
    model = spec.compile()
    model.opt.timestep = 1.0 / (fps * SUBSTEPS)
    if kp is not None:  # position actuator: force = kp * (goal - q) - kv * qvel
        model.actuator_gainprm[:, 0] = kp
        model.actuator_biasprm[:, 1] = -kp
    cam = dict(CAMERA, **(camera or {}))
    model.vis.global_.fovy = cam["fovy"]
    model.vis.headlight.ambient[:] = 0.35
    model.vis.headlight.diffuse[:] = 0.45
    names = [model.joint(i).name for i in range(model.njnt)]
    assert names == JOINTS and model.nu == len(JOINTS), f"unexpected model joints {names}"
    return model, cam


class SimArm:
    def __init__(self, assets=ASSETS, width=640, height=360, fps=30.0, camera=None, kp=17.8,
                 arm_rgba=(0.08, 0.08, 0.09, 1)):
        self.model, cam = build_model(assets, fps, camera, kp, arm_rgba)
        self.data = mujoco.MjData(self.model)
        self._scratch = mujoco.MjData(self.model)  # for kinematics that must not disturb the simulation
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = cam["lookat"]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = cam["distance"], cam["azimuth"], cam["elevation"]
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.fps = fps
        self.ready = np.array(READY_DEG)
        self.range_deg = np.degrees(self.model.jnt_range)  # [6, 2]
        self._tip = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

    def reset(self, q_deg):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = np.radians(q_deg)
        self.data.ctrl[:] = np.radians(q_deg)
        for _ in range(SUBSTEPS * 15):  # let the joints settle under gravity
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0.0

    def step(self, goal_deg):
        """Hold these servo goals for one camera frame."""
        self.data.ctrl[:] = np.radians(goal_deg)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

    @property
    def t(self):
        return float(self.data.time)

    @property
    def q_deg(self):
        return np.degrees(self.data.qpos).copy()

    @property
    def qvel_deg(self):
        return np.degrees(self.data.qvel).copy()

    @property
    def torque(self):
        """Actuator torque per joint, N m (limit 3.35)."""
        return self.data.actuator_force.copy()

    def tip(self, q_deg=None):
        """Gripper tip position (m) now, or for a hypothetical pose."""
        if q_deg is None:
            return self.data.site_xpos[self._tip].copy()
        self._scratch.qpos[:] = np.radians(q_deg)
        mujoco.mj_kinematics(self.model, self._scratch)
        return self._scratch.site_xpos[self._tip].copy()

    def lowest(self, q_deg):
        """Height (m) of the lowest moving-link origin or gripper tip above the table for a hypothetical pose."""
        self._scratch.qpos[:] = np.radians(q_deg)
        mujoco.mj_kinematics(self.model, self._scratch)
        return float(min(self._scratch.xpos[3:, 2].min(), self._scratch.site_xpos[self._tip, 2]))

    def render(self):
        """Camera frame, BGR uint8 [H, W, 3] like OpenCV."""
        self.renderer.update_scene(self.data, camera=self.cam)
        return self.renderer.render()[..., ::-1]

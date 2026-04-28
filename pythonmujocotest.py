import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path("google_robot/robot.xml")
data = mujoco.MjData(model)

start_time = time.time()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        # -------------------------
        # 1. Time-based smooth target
        # -------------------------
        t = time.time() - start_time

        # slow oscillation (gentle arm motion)
        target_angle = 0.5 * np.sin(0.5 * t)

        # -------------------------
        # 2. Apply to all actuated joints
        # -------------------------
        kp = 15.0   # softer than before
        kd = 3.0    # damping for smoothness

        qpos = data.qpos
        qvel = data.qvel

        for i in range(model.nu):
            error = target_angle - qpos[i]
            data.ctrl[i] = kp * error - kd * qvel[i]

        # -------------------------
        # 3. Step simulation
        # -------------------------
        mujoco.mj_step(model, data)
        viewer.sync()
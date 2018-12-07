import numpy as np
import mujoco_py
from gym.envs.robotics import rotations, robot_env


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


def robot_get_obs(sim):
    """Returns all joint positions and velocities associated with
    a robot.
    """
    if sim.data.qpos is not None and sim.model.joint_names:
        names = [n for n in sim.model.joint_names if n.startswith('robot')]
        return (
            np.array([sim.data.get_joint_qpos(name) for name in names]),
            np.array([sim.data.get_joint_qvel(name) for name in names]),
        )
    return np.zeros(0), np.zeros(0)


def ctrl_set_action(sim, action):
    """For torque actuators it copies the action into mujoco ctrl field.
    For position actuators it sets the target relative to the current qpos.
    """
    if sim.model.nmocap > 0:
        _, action = np.split(action, (sim.model.nmocap * 7, ))
    if sim.data.ctrl is not None:
        for i in range(action.shape[0]):
            if sim.model.actuator_biastype[i] == 0:
                sim.data.ctrl[i] = action[i]
            else:
                idx = sim.model.jnt_qposadr[sim.model.actuator_trnid[i, 0]]
                sim.data.ctrl[i] = sim.data.qpos[idx] + action[i]


def mocap_set_action(sim, action):
    """The action controls the robot using mocaps. Specifically, bodies
    on the robot (for example the gripper wrist) is controlled with
    mocap bodies. In this case the action is the desired difference
    in position and orientation (quaternion), in world coordinates,
    of the of the target body. The mocap is positioned relative to
    the target body according to the delta, and the MuJoCo equality
    constraint optimizer tries to center the welded body on the mocap.
    """
    if sim.model.nmocap > 0:
        action, _ = np.split(action, (sim.model.nmocap * 7, ))
        action = action.reshape(sim.model.nmocap, 7)

        pos_delta = action[:, :3]
        quat_delta = action[:, 3:]

        reset_mocap2body_xpos(sim)
        sim.data.mocap_pos[:] = sim.data.mocap_pos + pos_delta
        sim.data.mocap_quat[:] = sim.data.mocap_quat + quat_delta


def reset_mocap_welds(sim):
    """Resets the mocap welds that we use for actuation.
    """
    if sim.model.nmocap > 0 and sim.model.eq_data is not None:
        for i in range(sim.model.eq_data.shape[0]):
            if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                sim.model.eq_data[i, :] = np.array(
                    [0., 0., 0., 1., 0., 0., 0.])
    sim.forward()


def reset_mocap2body_xpos(sim):
    """Resets the position and orientation of the mocap bodies to the same
    values as the bodies they're welded to.
    """

    if (sim.model.eq_type is None or
        sim.model.eq_obj1id is None or
        sim.model.eq_obj2id is None):
        return
    for eq_type, obj1_id, obj2_id in zip(sim.model.eq_type,
                                         sim.model.eq_obj1id,
                                         sim.model.eq_obj2id):
        if eq_type != mujoco_py.const.EQ_WELD:
            continue

        mocap_id = sim.model.body_mocapid[obj1_id]
        if mocap_id != -1:
            # obj1 is the mocap, obj2 is the welded body
            body_idx = obj2_id
        else:
            # obj2 is the mocap, obj1 is the welded body
            mocap_id = sim.model.body_mocapid[obj2_id]
            body_idx = obj1_id

        assert (mocap_id != -1)
        sim.data.mocap_pos[mocap_id][:] = sim.data.body_xpos[body_idx]
        sim.data.mocap_quat[mocap_id][:] = sim.data.body_xquat[body_idx]


class FrankaEnv(robot_env.RobotEnv):

    """Base class for all Franka based environments."""

    def __init__(
        self,
        model_path,
        n_substeps,
        gripper_extra_height,
        block_gripper,
        has_object,
        target_in_the_air,
        target_offset,
        obj_range,
        target_range,
        distance_threshold,
        initial_qpos,
        reward_type
    ):
        """Creates a new FrankaEnv instance.

        :param model_path path to the environments XML file
        :param n_substeps number of substeps the simulation runs on every call to step
        :param gripper_extra_height additional height above the table when positioning the gripper
        :param block_gripper whether or not the gripper is blocked (i.e. not movable) or not
        :param has_object whether or not the environment has an object
        :param target_in_the_air whether or not the target should be in the air above the table or on the table surface
        :param target_offset offset of the target
        :param obj_range range of a uniform distribution for sampling initial object positions
        :param target_range range of a uniform distribution for sampling a target
        :param distance_threshold the threshold after which a goal is considered achieved
        :param initial_qpos a dictionary of joint names and values that define the initial configuration
        :param reward_type the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type

        super().__init__(
                model_path=model_path,
                n_substeps=n_substeps,
                n_actions=4,
                initial_qpos=initial_qpos
        )

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        # Force the finger position to their default value
        if self.block_gripper:
            self.sim.data.set_joint_qpos("robot:finger1", 0.0)
            self.sim.data.set_joint_qpos("robot:finger2", 0.0)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        # Scale position change down
        pos_ctrl *= 0.05
        # Fixing end-effector rotation
        rot_ctrl = [0.0, 1.0, 0.0, 0.0]

        gripper_ctrl = np.array([gripper_ctrl])
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        assert gripper_ctrl.shape == (1,)

        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        ctrl_set_action(self.sim, action)
        mocap_set_action(self.sim, action)

    def _get_obs(self):
        # Gripper data
        grip_pos = self.sim.data.get_site_xpos("grip")
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("grip") * dt
        robot_qpos, robot_qvel = robot_get_obs(self.sim)

        # Object data
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos("object0")
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat("object0"))
            object_velp = self.sim.data.get_site_xvelp("object0") * dt
            object_velr = self.sim.data.get_site_xvelr("object0") * dt
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())

        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    # def _viewer_setup(self):
    #     body_id = self.sim.model.body_name2id('robot0:gripper_link')
    #     lookat = self.sim.data.body_xpos[body_id]
    #     for idx, value in enumerate(lookat):
    #         self.viewer.cam.lookat[idx] = value
    #     self.viewer.cam.distance = 2.5
    #     self.viewer.cam.azimuth = 132.
    #     self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id("target0")
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            offset = self.np_random.uniform(-self.obj_range, self.obj_range, size=3)
            offset[2] = 0.1
            table_pos = self.sim.data.get_body_xpos("table0") + offset

            # object_xpos = self.initial_gripper_xpos[:2]
            # while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
            #     object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            # object_qpos = self.sim.data.get_joint_qpos("object0:joint")
            # assert object_qpos.shape == (7,)
            # object_qpos[:2] = object_xpos
            # self.sim.data.set_joint_qpos("object0:joint", object_qpos)

            obj_qpos = self.sim.data.get_joint_qpos("object0:joint")
            obj_qpos[:3] = table_pos
            self.sim.data.set_joint_qpos("object0:joint", obj_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        # gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos("grip")
        gripper_target = self.sim.data.get_body_xpos("table0") + \
                np.array([0.0, 0.0, 0.5 + self.gripper_extra_height])

        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos("robot:mocap", gripper_target)
        self.sim.data.set_mocap_quat("robot:mocap", gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos("grip").copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos("object0")[2]

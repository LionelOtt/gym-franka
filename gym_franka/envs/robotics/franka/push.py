import math
import os

from gym import utils
from gym_franka.envs.robotics import franka_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join(
        os.path.dirname(__file__),
        "..",
        "assets",
        "franka",
        "push.xml"
)


class FrankaPushEnv(franka_env.FrankaEnv, utils.EzPickle):

    def __init__(self, reward_type="sparse"):
        initial_qpos = {
            # 'robot0:slide0': 0.405,
            # 'robot0:slide1': 0.48,
            # 'robot0:slide2': 0.0,
            # "robot:joint1": math.pi/2,
            "object0:joint": [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        franka_env.FrankaEnv.__init__(
                self,
                MODEL_XML_PATH,
                has_object=True,
                block_gripper=True,
                n_substeps=20,
                gripper_extra_height=0.0,
                target_in_the_air=False,
                target_offset=0.0,
                obj_range=0.15,
                target_range=0.15,
                distance_threshold=0.05,
                initial_qpos=initial_qpos,
                reward_type=reward_type
        )
        utils.EzPickle.__init__(self)

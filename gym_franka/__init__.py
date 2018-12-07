from gym.envs.registration import register


register(
    id="FrankaReach-v0",
    entry_point="gym_franka.envs.robotics:FrankaReachEnv",
    max_episode_steps=50
)

register(
    id="FrankaPush-v0",
    entry_point="gym_franka.envs.robotics:FrankaPushEnv",
    max_episode_steps=50
)

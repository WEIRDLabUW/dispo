import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env

from .rewards import tolerance


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, seed):
        mujoco_env.MujocoEnv.__init__(self, "hopper.xml", 4)
        utils.EzPickle.__init__(self)
        self.spec = AttrDict(id="hopper")
        self.seed(seed)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()
        reward = self.compute_reward(obs)
        return obs, reward, False, {}

    def compute_reward(self, obs):
        speed = obs[5]
        reward = speed * 2
        return reward

    def _get_obs(self):
        return np.concatenate(
            [self.sim.data.qpos.flat[1:], np.clip(self.sim.data.qvel.flat, -10, 10)]
        ).astype(np.float32)

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20


class HopperBackwardEnv(HopperEnv):
    def compute_reward(self, obs):
        speed = obs[5]
        reward = -speed * 2
        return reward


class HopperStandEnv(HopperEnv):
    def compute_reward(self, obs):
        height = obs[0]
        reward = tolerance(height, (1.2, np.inf), margin=1.2)
        return reward


class HopperJumpEnv(HopperEnv):
    def compute_reward(self, obs):
        height, vspeed = obs[0], obs[6]
        reward = tolerance(height, (1.5, np.inf), margin=1.5)
        reward = reward + np.clip(vspeed, a_min=0, a_max=None)
        return reward

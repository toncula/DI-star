from distar.envs.env import SC2Env

from ding.envs import BaseEnv

class DIStarEnv(SC2Env,BaseEnv):

    def __init__(self,cfg):
        super(DIStarEnv, self).__init__(cfg)

    def reset(self):
        return super(DIStarEnv,self).reset()

    def close(self):
        super(DIStarEnv,self).close()

    def step(self,actions):
        return super(DIStarEnv,self).step(actions)

    def seed(self, seed, dynamic_seed=False):
        self._random_seed = seed
    
    @property
    def observation_space(self):
        #TODO
        pass

    @property
    def action_space(self):
        #TODO
        pass

    @property
    def reward_space(self):
        #TODO
        pass

    def __repr__(self):
        return "DI-engine DI-star Env"
import numpy as np
import numpy.random as npr

from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.planners.block_stacking_planner import BlockStackingPlanner
from helping_hands_rl_envs.planners.block_adjacent_planner import BlockAdjacentPlanner
from helping_hands_rl_envs.planners.play_planner import PlayPlanner

class MultiTaskPlanner(object):
  def __init__(self, env, configs):
    self.env = env

    self.planners = list()
    for i, config in enumerate(configs):
      if config['planner_type'] == 'block_stacking':
        self.planners.append(BlockStackingPlanner(env.envs[i], config))
      elif config['planner_type'] == 'block_adjacent':
        self.planners.append(BlockAdjacentPlanner(env.envs[i], config))
      elif config['planner_type'] == 'play':
        self.planners.append(PlayPlanner(env.envs[i], config))

  def getNextAction(self):
    return self.planners[self.env.active_env_id].getNextAction()

  def getStepsLeft(self):
    return self.planners[self.env.active_env_id].getStepsLeft()

  def getValue(self):
    return self.planners[self.env.active_env_id].getValue()

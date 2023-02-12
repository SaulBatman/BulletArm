import pybullet as pb
import numpy as np

from bulletarm.pybullet.utils import constants
from bulletarm.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from bulletarm.pybullet.utils import transformations
from bulletarm.planners.close_loop_block_stacking_planner import CloseLoopBlockStackingPlanner
from bulletarm.pybullet.utils.constants import NoValidPositionException

import matplotlib.pyplot as plt

class CloseLoopBlockStackingEnv(CloseLoopEnv):
  '''Close loop block stacking task.

  The robot needs to stack all N cubic blocks. The number of blocks N is set by the config.

  Args:
    config (dict): Intialization arguments for the env
  '''
  def __init__(self, config):
    if 'num_objects' not in config:
      config['num_objects'] = 2
    super().__init__(config)
    assert self.num_obj >= 2

  def reset(self):
    while True:
      self.resetPybulletWorkspace()
      self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
      try:
        self._generateShapes(constants.CUBE, self.num_obj, random_orientation=self.random_orientation)
      except NoValidPositionException as e:
        continue
      else:
        break
    return self._getObservation()

  def _getValidOrientation(self, random_orientation):
    if random_orientation:
      orientation = pb.getQuaternionFromEuler([0., 0., np.pi/2 * (np.random.random_sample() - 0.5)])
    else:
      orientation = pb.getQuaternionFromEuler([0., 0., 0.])
    return orientation

  def _checkTermination(self):
    return not self._isHolding() and self._checkStack(self.objects)

  

def createCloseLoopBlockStackingEnv(config):
  return CloseLoopBlockStackingEnv(config)




from bulletarm.planners.close_loop_block_stacking_planner import CloseLoopBlockStackingPlanner
if __name__ == '__main__':
  env = CloseLoopBlockStackingEnv({'seed': 1, 'num_objects': 3, 'view_type': 'camera_center_xyz_height', 'robot':'kuka', 'seed': 1, 'view_scale': 1.5, 'workspace': np.array([[0.25, 0.65], [-0.2, 0.2], [0.01, 0.25]]), 'render': True})
  planner = CloseLoopBlockStackingPlanner(env, {})
  _, _, obs = env.reset()
  step_num = 0
  while True:
    action = planner.getNextAction()
    # global_obs, goal_obs, goal_bbox, all_bbox = planner.getNextGoal()
    step_num += 1
    global_obs, goal_bbox, all_bbox = planner.getNextTemporal()
    if global_obs is not None:
      plt.imshow(global_obs[-1][0])
    print(global_obs)
    # plt.figure(1)
    # plt.imshow(goal_obs)
    # plt.figure(2)
    # plt.imshow(global_obs)
    (state, obs, in_hands), reward, done = env.step(action)
    # obs1, obs2, obs3 = planner.obscrop(global_obs, all_bbox)
    # fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    # axs[0].imshow(obs1)
    # axs[1].imshow(obs2)
    # axs[2].imshow(obs3)
    # print(1)
    # if done:
    #   env.reset()
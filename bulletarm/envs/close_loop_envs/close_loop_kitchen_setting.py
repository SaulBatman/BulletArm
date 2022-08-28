import pybullet as pb
import numpy as np

from bulletarm.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from bulletarm.pybullet.utils import constants
from bulletarm.planners.close_loop_shoe_packing_planner import CloseLoopShoePackPlanner
from bulletarm.pybullet.utils.constants import NoValidPositionException
from bulletarm.pybullet.objects.shoe_rack_short import ShoeRack

class CloseLoopKitchenSettingEnv(CloseLoopEnv):
    '''Close loop shoe packing task.

  The robot needs to pick up the shoes and place on the shoe rack in pairs.

  Args:
    config (dict): Initialization arguments for the env
  '''

    def __init__(self, config):
      super().__init__(config)

    def reset(self):
      while True:
        self.resetPybulletWorkspace()
        try:
          self._generateShapes(constants.KITCHEN_PLATE, 1, scale=0.5, random_orientation=self.random_orientation)
          self._generateShapes(constants.KITCHEN_KNIFE, 1, scale=0.8, random_orientation=self.random_orientation)
          self._generateShapes(constants.KITCHEN_FORK, 1, scale=1, random_orientation=self.random_orientation)

        except NoValidPositionException as e:
          continue
        else:
          break
      return self._getObservation()

    def _checkTermination(self):

      shoe_left_pos = self.objects[1].getPosition()
      shoe_right_pos = self.objects[2].getPosition()

      shoe_rack_pos_left = ShoeRack.getLeftPose(self.objects[0])[0]
      shoe_rack_pos_right = ShoeRack.getRightPose(self.objects[0])[0]

      stage = [0, 0]
      finish = False
      change = False
      if np.linalg.norm(np.array(shoe_left_pos) - np.array(shoe_rack_pos_left)) < 0.03:
        stage[0] = 1
      if np.linalg.norm(np.array(shoe_right_pos) - np.array(shoe_rack_pos_right)) < 0.03:
        stage[1] = 1

      if stage[0] == 1 and stage[1] == 1 and \
        self._checkObjUpright(self.objects[1]) and self._checkObjUpright(self.objects[2]):
        finish = True
        change = stage[0] and self.previous_stage[0]
      self.previous_stage = stage

      return False


    def isSimValid(self):
      for obj in self.objects:
        p = obj.getPosition()

        if self._isObjectHeld(obj):
          continue
        if not self.workspace[0][0] - 0.05 < p[0] < self.workspace[0][1] + 0.05 and \
          self.workspace[1][0] - 0.05 < p[1] < self.workspace[1][1] + 0.05 and \
          self.workspace[2][0] < p[2] < self.workspace[2][1]:
          return False
        return True


def createCloseLoopKitchenSettingEnv(config):
  return CloseLoopKitchenSettingEnv(config)


if __name__ == '__main__':
  env = CloseLoopKitchenSettingEnv({'seed': 0, 'render': True})
  # planner = CloseLoopShoePackPlanner(env, {})
  env.reset()
  # count = 0
  # while True:
  #   action = planner.getNextAction()
  #   (state, obs, in_hands), reward, done = env.step(action)
  #   # import time
  #   # time.sleep(0.1)
  #   if done:
  #     # count += 1
  #     env.reset()
  #     # if count == 6:
  #       # print(count)
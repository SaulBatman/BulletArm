import pybullet as pb
import numpy as np

from bulletarm.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from bulletarm.pybullet.utils import constants
from bulletarm.planners.close_loop_shoe_packing_planner import CloseLoopShoePackingPlanner
from bulletarm.pybullet.utils.constants import NoValidPositionException
from bulletarm.pybullet.objects.shoe_rack_short import ShoeRackShort
from bulletarm.pybullet.utils import transformations

class CloseLoopShoePacking(CloseLoopEnv):
    '''Close loop shoe packing task.

  The robot needs to pick up the shoes and place on the shoe rack in pairs.

  Args:
    config (dict): Initialization arguments for the env
  '''

    def __init__(self, config):
      super().__init__(config)
      self.previous_stage = (0, 0)

    def reset(self):
      while True:
        self.resetPybulletWorkspace()
        self.previous_stage = (0, 0)
        try:
          self._generateShapes(constants.SHOE_RACK_SHORT, 1, scale=0.7, random_orientation=self.random_orientation)
          self._generateShapes(constants.SHOE_LEFT, 1, scale=0.7, random_orientation=self.random_orientation)
          self._generateShapes(constants.SHOE_RIGHT, 1, scale=0.7, random_orientation=self.random_orientation)

        except NoValidPositionException as e:
          continue
        else:
          break
      return self._getObservation()

    def _checkTermination(self):
      # the shoe size is (x = 0.052, y = 0.082, z = 0.052)
      # the shoe rack short size is (x = 0.16, y = 0.284, z = 0.135)

      shoe_left_pos = self.objects[1].getPosition()
      shoe_right_pos = self.objects[2].getPosition()

      shoe_rack_pos_left = ShoeRackShort.getLeftPose(self.objects[0])[0]
      shoe_rack_pos_right = ShoeRackShort.getRightPose(self.objects[0])[0]

      stage = [0, 0]
      finish = False
      change = False
      if abs(np.array(shoe_left_pos)[0] - np.array(shoe_rack_pos_left)[0]) < 0.04 and \
        abs(np.array(shoe_left_pos)[1] - np.array(shoe_rack_pos_left)[1]) < 0.02 and \
        abs(np.array(shoe_left_pos)[2] - np.array(shoe_rack_pos_left)[2]) < 0.01:
        stage[0] = 1
      if abs(np.array(shoe_right_pos)[0] - np.array(shoe_rack_pos_right)[0]) < 0.04 and \
        abs(np.array(shoe_right_pos)[1] - np.array(shoe_rack_pos_right)[1]) < 0.02 and \
        abs(np.array(shoe_right_pos)[2] - np.array(shoe_rack_pos_right)[2]) < 0.01:
        stage[1] = 1

      if stage[0] == 1 and stage[1] == 1 and \
        self._checkObjUpright(self.objects[1]) and self._checkObjUpright(self.objects[2]):
        finish = True
        change = stage[0] and self.previous_stage[0]
      self.previous_stage = stage

      return finish and change


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


def createCloseLoopShoePackingEnv(config):
  return CloseLoopShoePacking(config)

#
# if __name__ == '__main__':
#   env = CloseLoopShoePacking({'seed': 2, 'workspace': np.array([[0.2, 0.6], [-0.2, 0.2], [0, 1]]), 'render': True})
#   planner = CloseLoopShoePackingPlanner(env, {})
#   env.reset()
#   # count = 0
#   while True:
#     action = planner.getNextAction()
#     (state, obs, in_hands), reward, done = env.step(action)
#     # import time
#     # time.sleep(0.1)
#     if done:
#       # count += 1
#       env.reset()
#       # if count == 6:
#         # print(count)
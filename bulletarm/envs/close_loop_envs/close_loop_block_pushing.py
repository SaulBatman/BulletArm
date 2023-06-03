import pybullet as pb
import numpy as np

from bulletarm.pybullet.utils import constants
from bulletarm.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from bulletarm.pybullet.utils import transformations
from bulletarm.planners.close_loop_block_pushing_planner import CloseLoopBlockPushingPlanner
from bulletarm.pybullet.equipments.tray import Tray
from bulletarm.pybullet.utils.constants import NoValidPositionException

class CloseLoopBlockPushingEnv(CloseLoopEnv):
  ''' Close loop block pushing task.

  The robot needs to push a block into a goal area.

  Args:
    config (dict): Intialization arguments for the env
  '''
  def __init__(self, config):
    super().__init__(config)
    self.goal_pos = self.workspace.mean(1)[:2]
    self.goal_id = None
    self.goal_grid_size_half = 11
    self.goal_size = self.goal_grid_size_half * 2 * self.obs_size_m / self.heightmap_size

  def getGoalPixel(self, gripper_pos=None):
    if gripper_pos is None:
      gripper_pos = self.robot._getEndEffectorPosition()
    goal_pixel_x = (self.goal_pos[0] - gripper_pos[0]) / self.heightmap_resolution + self.heightmap_size // 2
    goal_pixel_y = (self.goal_pos[1] - gripper_pos[1]) / self.heightmap_resolution + self.heightmap_size // 2
    return round(goal_pixel_x), round(goal_pixel_y)

  def reset(self):
    while True:
      self.resetPybulletWorkspace()
      try:
        if not self.random_orientation:
          padding = 0.08+0.05
          min_distance = 0.09
          x = np.random.random() * (self.workspace_size - padding) + self.workspace[0][0] + padding / 2
          while True:
            y1 = np.random.random() * (self.workspace_size - padding) + self.workspace[1][0] + padding / 2
            y2 = np.random.random() * (self.workspace_size - padding) + self.workspace[1][0] + padding / 2
            if max(y1, y2) - min(y1, y2) > min_distance:
              break
          self._generateShapes(constants.CUBE, 1, pos=[[x, y1, self.object_init_z]], random_orientation=False)
          goal_pos = [x, y2]
        else:
          self._generateShapes(constants.CUBE, 1, random_orientation=self.random_orientation)
          # pb.changeDynamics(self.objects[0].object_id, -1, lateralFriction=0.1)
          goal_pos = self._getValidPositions(0.08+0.05, 0.09, self.getObjectPositions()[:, :2].tolist(), 1)[0]
        self.goal_pos = goal_pos
      except NoValidPositionException as e:
        continue
      else:
        break
    if self.goal_id is not None:
      pb.removeBody(self.goal_id)
    goal_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[self.goal_size/2, self.goal_size/2, 0.0025], rgbaColor=[0, 0, 1, 1])
    self.goal_id = pb.createMultiBody(baseMass=0,
                                      baseVisualShapeIndex=goal_visual,
                                      basePosition=[*self.goal_pos, 0],
                                      baseOrientation=transformations.quaternion_from_euler(0, 0, 0), )

    return self._getObservation()

  def _getHeightmap(self, gripper_pos=None, gripper_rz=None):
    heightmap = super()._getHeightmap(gripper_pos, gripper_rz)
    goal_x, goal_y = self.getGoalPixel(gripper_pos)
    # heightmap[max(goal_x-self.goal_grid_size, 0):min(goal_x+self.goal_grid_size, self.heightmap_size-1), max(goal_y-self.goal_grid_size, 0):min(goal_y+self.goal_grid_size, self.heightmap_size-1)] += 0.025
    test_x = np.arange(goal_x - self.goal_grid_size_half, goal_x + self.goal_grid_size_half, 1)
    test_x = test_x[(0 <= test_x) & (test_x < 128)]
    test_y = np.arange(goal_y - self.goal_grid_size_half, goal_y + self.goal_grid_size_half, 1)
    test_y = test_y[(0 <= test_y) & (test_y < 128)]
    # heightmap[test_x, test_y] += 0.025
    X2D, Y2D = np.meshgrid(test_x, test_y)
    out = np.column_stack((X2D.ravel(), Y2D.ravel())).astype(int)
    heightmap[out[:, 0].reshape(-1), out[:, 1].reshape(-1)] += 0.02
    return heightmap

  def _checkTermination(self):
    obj_pos = self.objects[0].getPosition()[:2]
    return np.linalg.norm(np.array(self.goal_pos) - np.array(obj_pos)) < 0.05

  def isSimValid(self):
    for obj in self.objects:
      p = obj.getPosition()
      if self._isObjectHeld(obj):
        continue
      if not self.workspace[0][0]-0.05 < p[0] < self.workspace[0][1]+0.05 and \
          self.workspace[1][0]-0.05 < p[1] < self.workspace[1][1]+0.05 and \
          self.workspace[2][0] < p[2] < self.workspace[2][1]:
        return False
    return True

def createCloseLoopBlockPushingEnv(config):
  return CloseLoopBlockPushingEnv(config)

from bulletarm.planners.close_loop_block_pushing_planner import CloseLoopBlockPushingPlanner
if __name__ == '__main__':
  # env = CloseLoopBlockStackingEnv({'seed': 1, 'num_objects': 3, 'time_horizon': 2, 'view_type': 'camera_side_viola_rgbd_custom_2', 'robot':'kuka', 'seed': 1, 'view_scale': 1.5, 'workspace': np.array([[0.25, 0.65], [-0.2, 0.2], [0.01, 0.25]]), 'render': True})
  env = CloseLoopBlockPushingEnv({'num_objects':2,'seed': 1, 'robot':'kuka', 'time_horizon': 2, 'obs_type': 'state_tr2', 'seed': 1, 'view_scale': 1.5, 'workspace': np.array([[0.25, 0.65], [-0.2, 0.2], [0.01, 0.25]]), 'render': True})

  
  planner = CloseLoopBlockPushingPlanner(env, {})
  _, _, obs = env.reset()
  
  step_num = 0
  while True:
    action = planner.getNextAction()
    # global_obs, goal_obs, goal_bbox, all_bbox = planner.getNextGoal()
    step_num += 1
    
    # if global_obs is not None:
    #   plt.imshow(global_obs[-1][0])
    # print(global_obs)
    # plt.figure(1)
    # plt.imshow(goal_obs)
    # plt.figure(2)
    # plt.imshow(global_obs)
    # print(env.getCurrentTimeStep())
    (state, in_hands, obs), reward, done = env.step(action)
    # global_obs, in_hand, goal_bbox, all_bbox, ee_pos, sub_traj_id, high_level_info = planner.getObsTemporal(done)
    # if high_level_info.max() >1 :
    #   print(high_level_info)
    # print(1)
    # obs1, obs2, obs3 = planner.obscrop(global_obs, all_bbox)
    # fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    # axs[0].imshow(obs1)
    # axs[1].imshow(obs2)
    # axs[2].imshow(obs3)
    # print(1)
    if done:
      env.reset()
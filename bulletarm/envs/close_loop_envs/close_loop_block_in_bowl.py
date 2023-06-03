import pybullet as pb
import numpy as np
import matplotlib.pyplot as plt

from bulletarm.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from bulletarm.pybullet.utils import constants
from bulletarm.planners.close_loop_block_in_bowl_planner import CloseLoopBlockInBowlPlanner
from bulletarm.pybullet.utils.constants import NoValidPositionException
from bulletarm.pybullet.equipments.tray import Tray

from bulletarm.visualization import visualizeObs

class CloseLoopBlockInBowlEnv(CloseLoopEnv):
  '''Close loop block in bowl task.

  The robot needs to pick up a block and place it inside a bowl.

  Args:
    config (dict): Intialization arguments for the env
  '''
  def __init__(self, config):
    if 'num_objects' not in config:
      config['num_objects'] = 2
    super().__init__(config)

  def reset_vis(self):
    # while True:
    #   self.resetPybulletWorkspace()
    #   try:
    #     self._generateShapes(constants.CUBE, 1, random_orientation=self.random_orientation)
    #     self._generateShapes(constants.BOWL, 1, scale=0.76, random_orientation=self.random_orientation)
    #   except NoValidPositionException as e:
    #     continue
    #   else:
    #     break
    # return self._getObservation()
    
    while True:
      self.resetPybulletWorkspace()
      try:
        self._generateShapes(constants.CUBE, 1, random_orientation=self.random_orientation)
        self._generateShapes(constants.BOWL, 1, scale=0.76, random_orientation=self.random_orientation)
      except NoValidPositionException as e:
        continue
      else:
        break
    # pb.removeBody(self.table_id)

    # simulate arm pos
    self.robot.moveTo([self.workspace[0].mean()-0.06, self.workspace[1].mean()-0.04, 0.2],
                      pb.getQuaternionFromEuler((0, 0, np.pi / 2)))
    joint_pos0 = list(zip(*pb.getJointStates(self.robot.id, list(np.arange(15)))))[0]
    

    # move from pos
    self.robot.moveTo([self.workspace[0].mean()+0.1, self.workspace[1].mean() - 0.1, 0.3],
                      pb.getQuaternionFromEuler((0, 0, np.pi / 2)))
    joint_pos1 = list(zip(*pb.getJointStates(self.robot.id, list(np.arange(15)))))[0]

    # move to pos
    self.robot.moveTo([self.workspace[0].mean()-0.03, self.workspace[1].mean()+0.06, 0.1],
                      pb.getQuaternionFromEuler((0, 0, np.pi / 2)))
    joint_pos2 = list(zip(*pb.getJointStates(self.robot.id, list(np.arange(15)))))[0]
    # plt.imshow(self._getObservation()[2][0], cmap='Greys', vmin=0.2)
    # plt.show()
  

    from bulletarm.pybullet.robots.kuka import Kuka
    self.robot1 = Kuka()
    self.robot1.initialize()
    self.robot2 = Kuka()
    self.robot2.initialize()

    [pb.resetJointState(self.robot.id, idx, joint_pos0[idx]) for idx in range(len(joint_pos0))]
    [pb.resetJointState(self.robot1.id, idx, joint_pos1[idx]) for idx in range(len(joint_pos1))]
    [pb.resetJointState(self.robot2.id, idx, joint_pos2[idx]) for idx in range(len(joint_pos2))]

    # change color
    # for i in range(-1, 15):
    #   pb.changeVisualShape(self.robot.id, i, rgbaColor=[1, 1, 1, 0.5])
    for i in range(-1, 5):
        pb.changeVisualShape(self.robot.id, i, rgbaColor=[1, 165 / 255, 0, 0])
    for i in range(-1, 5):
        pb.changeVisualShape(self.robot1.id, i, rgbaColor=[0, 1, 0, 0])
    # for i in range(-1, 15):
    #     pb.changeVisualShape(self.robot2.id, i, rgbaColor=[0, 1, 0, 0])
    for i in range(5, 15):
      pb.changeVisualShape(self.robot.id, i, rgbaColor=[1, 165 / 255, 0, 0.5])

    for i in range(5, 15):
      pb.changeVisualShape(self.robot1.id, i, rgbaColor=[0, 1, 0, 0.5])

    # for i in range(5, 15):
    #   pb.changeVisualShape(self.robot2.id, i, rgbaColor=[0, 1, 0, 0.5])

    return self._getObservation()


  def reset(self):
    while True:
      self.resetPybulletWorkspace()
      try:
        self._generateShapes(constants.CUBE, 1, random_orientation=self.random_orientation)
        self._generateShapes(constants.BOWL, 1, scale=0.76, random_orientation=self.random_orientation)
      except NoValidPositionException as e:
        continue
      else:
        break
    return self._getObservation()
    
    
  def _checkTermination(self):
    # check if bowl is upright
    if not self._checkObjUpright(self.objects[1]):
      return False
    # check if bowl and block is touching each other
    if not self.objects[0].isTouching(self.objects[1]):
      return False
    block_pos = self.objects[0].getPosition()[:2]
    bowl_pos = self.objects[1].getPosition()[:2]
    return np.linalg.norm(np.array(block_pos) - np.array(bowl_pos)) < 0.03

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

def createCloseLoopBlockInBowlEnv(config):
  return CloseLoopBlockInBowlEnv(config)

from bulletarm.planners.close_loop_block_in_bowl_planner import CloseLoopBlockInBowlPlanner
if __name__ == '__main__':
  # env = CloseLoopBlockInBowlEnv({'seed': 1, 'robot':'kuka', 'time_horizon': 2, 'view_type': 'camera_side_viola_rgbd_custom_2', 'seed': 1, 'view_scale': 1.5, 'workspace': np.array([[0.25, 0.65], [-0.2, 0.2], [0.01, 0.25]]), 'render': True})
  env = CloseLoopBlockInBowlEnv({'seed': 1, 'robot':'kuka', 'time_horizon': 2, 'obs_type': 'state_tr2', 'seed': 1, 'view_scale': 1.5, 'workspace': np.array([[0.25, 0.65], [-0.2, 0.2], [0.01, 0.25]]), 'render': True})
  planner = CloseLoopBlockInBowlPlanner(env, {})
  _, _, obs = env.reset()
  done = [False]
  while True:
    action = planner.getNextAction()
    # action[4] = 0
    global_obs, in_hand, goal_bbox, all_bbox, ee_pos, _, _ = planner.getObsTemporal(done)
    print(env.current_episode_steps)
    (state, in_hands, obs), reward, done = env.step(action)
    
    print(1)
    if done:
      print(1)
      # env.reset()
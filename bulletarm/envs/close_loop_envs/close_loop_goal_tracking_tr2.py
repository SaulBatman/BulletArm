import pybullet as pb
import numpy as np

from bulletarm.pybullet.utils import constants
from bulletarm.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from bulletarm.pybullet.utils import transformations
from bulletarm.planners.close_loop_block_stacking_planner import CloseLoopBlockStackingPlanner
from bulletarm.pybullet.utils.constants import NoValidPositionException
import copy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class CloseLoopBlockStackingTR2Env(CloseLoopEnv):
  '''Close loop block stacking task.

  The robot needs to stack all N cubic blocks. The number of blocks N is set by the config.

  Args:
    config (dict): Intialization arguments for the env
  '''
  def __init__(self, config):
    # if 'num_objects' not in config:
    #   config['num_objects'] = np.random.randint(1,4)
    #   print(config['num_objects'])
    super().__init__(config)
    self.goal_id = None

  def reset(self):
    self.done=0
    # self.num_obj = np.random.randint(1,4)
    self.num_obj = 2
    # print(self.num_obj)
    while True:
      self.resetPybulletWorkspace()
      self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
      try:
        shape_handles = self._generateShapes(constants.CUBE, 1, random_orientation=self.random_orientation)
        self.goal_pos = self._getValidPositions(0.08+0.05, 0.09, self.getObjectPositions()[:, :2].tolist(), 1)[0]
      except NoValidPositionException as e:
        continue
      else:
        break
    if self.goal_id is not None:
      pb.removeBody(goal.goal_id)
    rot = shape_handles[0].getRotation()
    self.goal_pos.append(0.025)
    goal = self._generateShapes(constants.VOID_GOAL, 1, pos=[self.goal_pos], rot=[rot])
    
    self.obj_poss=[]
    self.obj_rots=[]
    for obj in self.objects:
      self.obj_poss.append(copy.copy(obj.getPosition()))
      self.obj_rots.append(copy.copy(obj.getRotation()))
    return self._getObservation()
  
  def recover(self):
    self.done=0
    self.resetPybulletWorkspace()
    self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
    sorted_inds = np.flip(np.argsort(np.array(self.obj_poss)[:,2], axis=0))
    if self.num_obj == 1:
      self._generateShapes(constants.CUBE, 1, pos=[self.obj_poss[0]], rot=[self.obj_rots[0]])
      shape_handles = self._generateShapes(constants.VOID_GOAL, 1, pos=[self.obj_poss[1]], rot=[self.obj_rots[1]])
      # self.end_goal = shape_handles[0]
    else:
      for i in sorted_inds:
        try:
          shape_handles = self._generateShapes(constants.CUBE, 1, pos=[self.obj_poss[i]], rot=[self.obj_rots[i]])
        except NoValidPositionException as e:
          continue
      # self.end_goal = shape_handles[0]



  def _getValidOrientation(self, random_orientation):
    if random_orientation:
      orientation = pb.getQuaternionFromEuler([0., 0., np.pi/2 * (np.random.random_sample() - 0.5)])
    else:
      orientation = pb.getQuaternionFromEuler([0., 0., 0.])
    return orientation

  def _checkTermination(self):
    if self.num_obj == 1:
      obj_pos = self.objects[0].getPosition()
      return not self._isHolding() and np.linalg.norm(np.array(self.goal_pos) - np.array(obj_pos)) < 0.05
    else:
      return not self._isHolding() and self._checkStack(self.objects)
  

  

def createCloseLoopBlockStackingTR2Env(config):
  return CloseLoopBlockStackingTR2Env(config)




from bulletarm.planners.close_loop_block_stacking_tr2_planner import CloseLoopBlockStackingTR2Planner
if __name__ == '__main__':
  # env = CloseLoopBlockStackingEnv({'seed': 1, 'num_objects': 3, 'time_horizon': 2, 'view_type': 'camera_side_viola_rgbd_custom_2', 'robot':'kuka', 'seed': 1, 'view_scale': 1.5, 'workspace': np.array([[0.25, 0.65], [-0.2, 0.2], [0.01, 0.25]]), 'render': True})
  env = CloseLoopBlockStackingTR2Env({'robot':'kuka', 
                                      'time_horizon': 1, 
                                      'obs_type': 'state', 
                                      'view_type': 'vector_goal_distance',
                                      'seed': 1, 
                                      'view_scale': 1.5, 
                                      'workspace': np.array([[0.25, 0.65], [-0.2, 0.2], [0.01, 0.25]]), 
                                      'max_steps': 50,
                                      'dpos':0.2,
                                      'render': True})

  
  planner = CloseLoopBlockStackingTR2Planner(env, {'max_teacher_length': 50})
  _, _, obs = env.reset()
  # global_obs, in_hand, goal_bbox, all_bbox, ee_pos, sub_traj_id, high_level_info = planner.getObsTemporal(False)
  step_num = 0
  teacher_traj, mask, time_steps = planner.getHighLevelTraj()
  while True:

    state, global_obs, in_hand, ee_pos = planner.getObsTemporal()
    # print("obs:", global_obs)
    action = planner.getNextAction()
    action_q = planner.getNextTemporalAction(action)
    # print("action:", action)
    # print("action_q:", action_q)
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
    mask, steps = env.getCurrentTimeStep()
    print(steps)
    (state, in_hands, obs), reward, done = env.step(action)
    # print(planner.target_obj)
    # print(planner.last_target_obj)
    
    
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
      print('done')
      # env.recover()
      env.reset()
      teacher_traj, mask, time_steps = planner.getHighLevelTraj()
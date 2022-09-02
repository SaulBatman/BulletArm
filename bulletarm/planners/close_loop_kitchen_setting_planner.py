import numpy as np
from bulletarm.planners.close_loop_planner import CloseLoopPlanner
from bulletarm.pybullet.utils import transformations
# from bulletarm.pybullet.objects.shoe_rack_short import ShoeRackShort

class CloseLoopKitchenSettingPlanner(CloseLoopPlanner):
  def __init__(self, env, config):
    super().__init__(env, config)

    self.current_target = None
    self.stage = 0

    self.pre_grasp_pos_fork = self.env.workspace.mean(1)
    self.grasp_pos_fork = self.env.workspace.mean(1)
    self.post_grasp_pos_fork = self.env.workspace.mean(1)
    self.release_pos_fork = self.env.workspace.mean(1)
    self.pick_rot_fork = 0
    self.place_rot_fork = 0

    # self.pre_grasp_pos_knife = self.env.workspace.mean(1)
    # self.grasp_pos_knife = self.env.workspace.mean(1)
    # self.post_grasp_pos_knife = self.env.workspace.mean(1)
    # self.release_pos_knife = self.env.workspace.mean(1)
    # self.pick_rot_knife = 0
    # self.place_rot_knife = 0

  def getNextActionToCurrentTarget(self):
    x, y, z, r = self.getActionByGoalPose(self.current_target[0], self.current_target[1])
    if np.all(np.abs([x, y, z]) < self.dpos) and (not self.random_orientation or np.abs(r) < self.drot):
      p = self.current_target[3]
      self.current_target = None
    else:
      p = self.current_target[2]
    return self.env._encodeAction(p, x, y, z, r)

  def setWaypoints(self):
    plate_pos = self.env.objects[0].getPosition()
    fork_pos = self.env.objects[1].getPosition()
    # knife_pos = self.env.objects[2].getPosition()
    plate_rot = transformations.euler_from_quaternion(self.env.objects[0].getRotation())[2]
    fork_rot = transformations.euler_from_quaternion(self.env.objects[1].getRotation())[2]
    # knife_rot = transformations.euler_from_quaternion(self.env.objects[2].getRotation())[2]

    self.pick_rot_fork = fork_rot + np.pi / 2
    # self.pick_rot_knife = knife_rot
    self.place_rot_fork = plate_rot
    # self.place_rot_knife = plate_rot

    # rotate_fork = abs(self.pick_rot_fork - self.place_rot_fork)
    # rotate_knife = abs(self.pick_rot_knife - self.place_rot_knife)
    # if rotate_fork > np.pi:
    #   # print('----------------------')
    #   if self.place_rot_fork >= 0:
    #     self.place_rot_fork -= 2 * np.pi
    #   else:
    #     self.place_rot_fork += 2 * np.pi
    # if rotate_knife > np.pi:
    #   # print('~~~~~~~~~~~~~~~~~~~~~~')
    #   if self.place_rot_knife >= 0:
    #     self.place_rot_knife -= 2 * np.pi
    #   else:
    #     self.place_rot_knife += 2 * np.pi

    self.pre_grasp_pos_fork = [fork_pos[0], fork_pos[1], fork_pos[2] + 0.05]
    self.grasp_pos_fork = [fork_pos[0], fork_pos[1], fork_pos[2]]
    self.post_grasp_pos_fork = [fork_pos[0], fork_pos[1], fork_pos[2] + 0.05]
    self.release_pos_fork = [plate_pos[0], plate_pos[1], plate_pos[2] + 0.06]

    # self.pre_grasp_pos_knife = [knife_pos[0], knife_pos[1], knife_pos[2] + 0.15]
    # self.grasp_pos_knife = [knife_pos[0], knife_pos[1], knife_pos[2] + 0.005]
    # self.post_grasp_pos_knife = [knife_pos[0], knife_pos[1], knife_pos[2] + 0.15]
    # self.release_pos_knife = [plate_pos[0], plate_pos[1], plate_pos[2] + 0.06]

  def setNewTarget(self):
    if self.stage == 0:     #arrive fork
      self.setWaypoints()
      self.current_target = (self.pre_grasp_pos_fork, self.pick_rot_fork, 1, 1)
      self.stage = 1
    elif self.stage == 1:   #grasp fork
      self.current_target = (self.grasp_pos_fork, self.pick_rot_fork, 1, 0)
      self.stage = 2
    elif self.stage == 2:   #lift fork
      self.current_target = (self.post_grasp_pos_fork, self.pick_rot_fork, 0, 0)
      self.stage = 3
    elif self.stage == 3:   #before release fork
      self.current_target = (self.release_pos_fork, self.place_rot_fork, 0, 0)
      self.stage = 4
    elif self.stage == 4:   #release fork
      self.current_target = (self.release_pos_fork, self.place_rot_fork, 0, 1)
      self.stage = 0
    # elif self.stage == 5:   #arrive knife
    #   self.current_target = (self.pre_grasp_pos_knife, self.pick_rot_knife, 1, 1)
    #   self.stage = 6
    # elif self.stage == 6:   #grasp knife
    #   self.current_target = (self.grasp_pos_knife, self.pick_rot_knife, 1, 0)
    #   self.stage = 7
    # elif self.stage == 7:   #lift knife
    #   self.current_target = (self.post_grasp_pos_knife, self.pick_rot_knife, 0, 0)
    #   self.stage = 8
    # elif self.stage == 8:   #before release knife
    #   self.current_target = (self.release_pos_knife, self.place_rot_knife, 0, 0)
    #   self.stage = 9
    # elif self.stage == 9:   #release knife
    #   self.current_target = (self.release_pos_knife, self.place_rot_knife, 0, 1)
    #   self.stage = 0

  def getNextAction(self):
    if self.env.current_episode_steps == 1:
      self.current_target = None
      self.stage = 0
    if self.current_target is not None:
      return self.getNextActionToCurrentTarget()
    else:
      self.setNewTarget()
      return self.getNextActionToCurrentTarget()

  def getStepsLeft(self):
    return 100

  def getActionByGoalPose(self, goal_pos, goal_rot):
    current_pos = self.env.robot._getEndEffectorPosition()
    current_rot = transformations.euler_from_quaternion(self.env.robot._getEndEffectorRotation())
    pos_diff = goal_pos - current_pos
    rot_diff = np.array(goal_rot) - current_rot
    # TODO: should this be 2pi?
    if rot_diff[2] > np.pi:
      rot_diff[2] -= 2 * np.pi
      # print(1)
    elif rot_diff[2] < -np.pi:
      rot_diff[2] += 2 * np.pi
      # print(1)

    # R = np.array([[np.cos(-current_rot[-1]), -np.sin(-current_rot[-1])],
    #               [np.sin(-current_rot[-1]), np.cos(-current_rot[-1])]])
    # pos_diff[:2] = R.dot(pos_diff[:2])

    pos_diff[pos_diff // self.dpos > 0] = self.dpos
    pos_diff[pos_diff // -self.dpos > 0] = -self.dpos

    rot_diff[rot_diff // self.drot > 0] = self.drot
    rot_diff[rot_diff // -self.drot > 0] = -self.drot

    x, y, z, r = pos_diff[0], pos_diff[1], pos_diff[2], rot_diff[2]

    return x, y, z, r

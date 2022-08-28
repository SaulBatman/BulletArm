import numpy as np
from bulletarm.planners.close_loop_planner import CloseLoopPlanner
from bulletarm.pybullet.utils import transformations
from bulletarm.pybullet.objects.shoe_rack_short import ShoeRackShort

class CloseLoopShoePackingPlanner(CloseLoopPlanner):
  def __init__(self, env, config):
    super().__init__(env, config)

    self.current_target = None
    self.stage = 0

    self.pre_grasp_pos_left = self.env.workspace.mean(1)
    self.grasp_pos_left = self.env.workspace.mean(1)
    self.post_grasp_pos_left = self.env.workspace.mean(1)
    self.release_pos_left = self.env.workspace.mean(1)
    self.pick_rot_left = 0
    self.place_rot_left = 0

    self.pre_grasp_pos_right = self.env.workspace.mean(1)
    self.grasp_pos_right = self.env.workspace.mean(1)
    self.post_grasp_pos_right = self.env.workspace.mean(1)
    self.release_pos_right = self.env.workspace.mean(1)
    self.pick_rot_right = 0
    self.place_rot_right = 0

  def getNextActionToCurrentTarget(self):
    x, y, z, r = self.getActionByGoalPose(self.current_target[0], self.current_target[1])
    if np.all(np.abs([x, y, z]) < self.dpos) and (not self.random_orientation or np.abs(r) < self.drot):
      p = self.current_target[3]
      self.current_target = None
    else:
      p = self.current_target[2]
    return self.env._encodeAction(p, x, y, z, r)

  def setWaypoints(self):
    shoe_left_pos = self.env.objects[1].getPosition()
    shoe_right_pos = self.env.objects[2].getPosition()
    shoe_left_rot = transformations.euler_from_quaternion(self.env.objects[1].getRotation())[2]
    shoe_right_rot = transformations.euler_from_quaternion(self.env.objects[2].getRotation())[2]

    shoe_rack_pos_left = ShoeRackShort.getLeftPose(self.env.objects[0])[0]
    shoe_rack_pos_right = ShoeRackShort.getRightPose(self.env.objects[0])[0]
    shoe_rack_rot = transformations.euler_from_quaternion(self.env.objects[0].getRotation())[2]

    self.pick_rot_left = shoe_left_rot
    self.pick_rot_right = shoe_right_rot
    self.place_rot_left = shoe_rack_rot - np.pi / 2
    self.place_rot_right = shoe_rack_rot - np.pi / 2

    rotate_left = abs(self.pick_rot_left - self.place_rot_left)
    rotate_right = abs(self.pick_rot_right - self.place_rot_right)
    if rotate_left > np.pi:
      # print('----------------------')
      if self.place_rot_left >= 0:
        self.place_rot_left -= 2 * np.pi
      else:
        self.place_rot_left += 2 * np.pi
    if rotate_right > np.pi:
      # print('~~~~~~~~~~~~~~~~~~~~~~')
      if self.place_rot_right >= 0:
        self.place_rot_right -= 2 * np.pi
      else:
        self.place_rot_right += 2 * np.pi

    self.pre_grasp_pos_left = [shoe_left_pos[0], shoe_left_pos[1], shoe_left_pos[2] + 0.15]
    self.grasp_pos_left = [shoe_left_pos[0], shoe_left_pos[1], shoe_left_pos[2] + 0.005]
    self.post_grasp_pos_left = [shoe_left_pos[0], shoe_left_pos[1], shoe_left_pos[2] + 0.15]
    self.release_pos_left = [shoe_rack_pos_left[0], shoe_rack_pos_left[1], shoe_rack_pos_left[2] + 0.06]

    self.pre_grasp_pos_right = [shoe_right_pos[0], shoe_right_pos[1], shoe_right_pos[2] + 0.15]
    self.grasp_pos_right = [shoe_right_pos[0], shoe_right_pos[1], shoe_right_pos[2] + 0.005]
    self.post_grasp_pos_right = [shoe_right_pos[0], shoe_right_pos[1], shoe_right_pos[2] + 0.15]
    self.release_pos_right = [shoe_rack_pos_right[0], shoe_rack_pos_right[1], shoe_rack_pos_right[2] + 0.06]

  def setNewTarget(self):
    if self.stage == 0:   #arrive left shoe
      self.setWaypoints()
      self.current_target = (self.pre_grasp_pos_left, self.pick_rot_left, 1, 1)
      self.stage = 1
    elif self.stage == 1:   #grasp left shoe
      self.current_target = (self.grasp_pos_left, self.pick_rot_left, 1, 0)
      self.stage = 2
    elif self.stage == 2:   #lift left shoe
      self.current_target = (self.post_grasp_pos_left, self.pick_rot_left, 0, 0)
      self.stage = 3
    elif self.stage == 3:   #before release left shoe
      self.current_target = (self.release_pos_left, self.place_rot_left, 0, 0)
      self.stage = 4
    elif self.stage == 4:   #release left shoe
      self.current_target = (self.release_pos_left, self.place_rot_left, 0, 1)
      self.stage = 5
    elif self.stage == 5:   #arrive right shoe
      self.current_target = (self.pre_grasp_pos_right, self.pick_rot_right, 1, 1)
      self.stage = 6
    elif self.stage == 6:   #grasp right shoe
      self.current_target = (self.grasp_pos_right, self.pick_rot_right, 1, 0)
      self.stage = 7
    elif self.stage == 7:   #lift right shoe
      self.current_target = (self.post_grasp_pos_right, self.pick_rot_right, 0, 0)
      self.stage = 8
    elif self.stage == 8:   #before release right shoe
      self.current_target = (self.release_pos_right, self.place_rot_right, 0, 0)
      self.stage = 9
    elif self.stage == 9:   #release right shoe
      self.current_target = (self.release_pos_right, self.place_rot_right, 0, 1)
      self.stage = 0

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

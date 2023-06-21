import numpy as np

from bulletarm.pybullet.utils import constants
from bulletarm.planners.base_planner import BasePlanner
from bulletarm.pybullet.utils import transformations

import queue

class CloseLoopPlanner(BasePlanner):
  def __init__(self, env, config):
    super().__init__(env, config)
    self.dpos = config['dpos'] if 'dpos' in config else 0.05
    self.drot = config['drot'] if 'drot' in config else np.pi / 4

    self.high_traj_type = config['high_traj_type']

    self.time_horizon = self.env.time_horizon
    self.pick_place_stage = 0
    # self.post_pose_reached = False
    # pos, rot, primitive
    self.current_target = None
    self.target_obj = None

    self.last_target_obj = None
    self.global_obs = None
    self.goal_obs = None
    self.init_q = False
    self.global_obs_q = None
    self.goal_bbox_q = None
    self.all_bbox_q = None
    self.in_hand = None
    self.in_hand_q = None
    self.ee_pos_q = None
    # self.all_obs = None # (N, x1, y1, x2, y2)

    self.max_teacher_length = config['max_teacher_length']
    self.init_a = False
    self.last_target_obj = None
    # self.traj_type = config['obs_type'] if 'dpos' in config else 'state_tr2'

  def getActionByGoalPose(self, goal_pos, goal_rot):
    current_pos = self.env.robot._getEndEffectorPosition()
    current_rot = transformations.euler_from_quaternion(self.env.robot._getEndEffectorRotation())
    pos_diff = goal_pos - current_pos
    rot_diff = np.array(goal_rot) - current_rot

    # R = np.array([[np.cos(-current_rot[-1]), -np.sin(-current_rot[-1])],
    #               [np.sin(-current_rot[-1]), np.cos(-current_rot[-1])]])
    # pos_diff[:2] = R.dot(pos_diff[:2])

    pos_diff[pos_diff // self.dpos > 0] = self.dpos
    pos_diff[pos_diff // -self.dpos > 0] = -self.dpos

    rot_diff[rot_diff // self.drot > 0] = self.drot
    rot_diff[rot_diff // -self.drot > 0] = -self.drot

    x, y, z, r = pos_diff[0], pos_diff[1], pos_diff[2], rot_diff[2]

    return x, y, z, r


  def getAllAnchors(self):
    # (N, x, y) in global_obs image space
    object_poses = self.env.getObjectPoses(self.env.objects)
    return self.space2img(object_poses[:,:2])

  def getAllAnchorsWorld(self):
    # (N, x, y) in global_obs image space
    object_poses = self.env.getObjectPoses(self.env.objects)
    gripper_pos = self.env.robot._getEndEffectorPosition().reshape(1,3)
    poses = np.concatenate([object_poses[:,:3], gripper_pos], axis=0)
    return poses

  def centercrop(self, img, size=14):
    # crop around center
    img_x, img_y = img.shape
    return img[img_x//2-size:img_x//2+size, img_y//2-size:img_y//2+size]

  def getbbox(self, anchor, half_size=6):
    if len(anchor.shape) == 1:
      # (x1, y1, x2, y2)
      return np.array([anchor[0]-half_size, anchor[1]-half_size, anchor[0]+half_size, anchor[1]+half_size])
    elif len(anchor.shape) > 1:
      # (N, x1, y1, x2, y2)
      return np.vstack((anchor[:,0]-half_size, anchor[:,1]-half_size, anchor[:,0]+half_size, anchor[:,1]+half_size)).T
    else:
      NotImplementedError

  def obscrop(self, global_obs, bbox):
    # crop around anchor
    if len(bbox.shape) == 1:
      return global_obs[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    elif len(bbox.shape) > 1:
      bbox_obs = []
      for i in range(bbox.shape[0]):
        bbox_obs.append(global_obs[bbox[i,0]:bbox[i,2], bbox[i,1]:bbox[i,3]])
      return bbox_obs
    else:
      NotImplementedError

  def space2img(self, anchor):
    view_scale = self.env.view_scale
    work_space = self.env.workspace
    workspace_size =  work_space[0,1] - work_space[0,0]
    center_shift = np.array([128,128])/2 - np.array([128,128])/2/view_scale
    if len(anchor.shape) == 1:
      # Convert coordinates in workspace (x,y) to img indices. (only for center scaling)
      x_in_ws, y_in_ws = anchor
      x_y_idx = np.array([(x_in_ws-work_space[0,0])/workspace_size*128, (y_in_ws-work_space[1,0])/workspace_size*128])
      xy_in_img = (x_y_idx/view_scale + center_shift).astype(int)
      return xy_in_img
    elif len(anchor.shape) > 1:
      # Convert coordinates in workspace (N, x, y) to img indices. (only for center scaling)
      xs = (anchor[:,0]-work_space[0,0])/workspace_size*128
      ys = (anchor[:,1]-work_space[1,0])/workspace_size*128
      xys_in_img = (np.array([xs, ys]).T/view_scale + center_shift).astype(int)
      return xys_in_img
    else:
      NotImplementedError

  def scaleXYZR(self, XYZR):
    if not (-1<self.env._scaleRz(XYZR[2])<1):
      print(XYZR)
    return np.array([self.env._scaleX(XYZR[0]), self.env._scaleY(XYZR[1]), self.env._scaleZ(XYZR[2]), self.env._scaleRz(XYZR[3])])


  def getNextGoal(self):
    # print(self.target_obj)
    half_size = 10
    if self.target_obj is None:
      object_x, object_y, object_z = self.last_target_obj.getPosition()[:3]
      # object_x, object_y, object_z = self.end_goal.getPosition()[:3]
      object_rot = list(transformations.euler_from_quaternion(self.last_target_obj.getRotation()))[2]
    else:
      object_x, object_y, object_z = self.target_obj.getPosition()[:3]
      object_rot = list(transformations.euler_from_quaternion(self.target_obj.getRotation()))[2]
    ee_x, ee_y, ee_z = self.env.robot._getEndEffectorPosition()
    # object_rot = list(transformations.euler_from_quaternion(self.target_obj.getRotation()))
    # gripper_rz = transformations.euler_from_quaternion(self.env.robot._getEndEffectorRotation())[2]
    # self.goal_obs = self.centercrop(self.env._getHeightmap(gripper_pos=[object_x, object_y, ee_z], gripper_rz=None))
    # self.global_obs = self.env._getHeightmap(gripper_pos=[self.env.workspace[0].mean(), self.env.workspace[1].mean(), 0.2], gripper_rz=None).reshape(1,128,128)
    state, self.in_hand, self.global_obs = self.env._getObservation()
    ee_rot = transformations.euler_from_quaternion(self.env.robot._getEndEffectorRotation())
    # print(object_x, object_y)
    self.last_target_obj = self.target_obj if self.target_obj is not None else self.last_target_obj
    if self.env.obs_type == 'state':
      
      goal_bbox = self.scaleXYZR([object_x, object_y, object_z, object_rot])
      all_bbox = self.env.getObjectPoses(self.env.objects)
      # if self.env.view_type == 'vector_tr2':
      #   self.global_obs = np.concatenate([self.global_obs, goal_bbox])
      # print(self.env.view_type.find('goal'))
      if self.env.view_type.find('goal') > -1 or self.env.view_type.find('tr2') > -1:
        if self.env.view_type in ['vector_goal', 'vector_tr2']:
          self.global_obs = np.concatenate([self.global_obs, goal_bbox]) # (35+4) obs
        elif self.env.view_type == 'vector_goal_distance':
          scaled_ee = self.scaleXYZR([ee_x, ee_y, ee_z, ee_rot[2]])
          goal_distance = np.abs(goal_bbox-scaled_ee)
          self.global_obs = np.concatenate([self.global_obs, goal_distance]) # (35+4) obs

      if self.env.high_traj_type == 'goal_ee':
        high_level_info = np.concatenate([self.scaleXYZR([ee_x, ee_y, ee_z, ee_rot[2]])])
      elif self.env.high_traj_type == 'goal_obj':
        high_level_info = np.concatenate([goal_bbox])
      else:
        high_level_info = np.concatenate([self.scaleXYZR([ee_x, ee_y, ee_z, ee_rot[2]]), goal_bbox])
      # high_level_info = self.global_obs
    else:
      # get goal anchor and obs
      if self.env.view_type.find('side') == -1:
        goal_img_xy = self.space2img(np.array([object_x, object_y, object_z]).reshape(1, -1))
        goal_bbox = self.getbbox(goal_img_xy) # bounding box (x1, y1, x2, y2) 
        anchors = self.getAllAnchors()
      else:
        # self.env.sensor.getPointCloud(128)
        goal_img_xy = self.env.sensor.world2cam(np.array([object_x, object_y, object_z]).reshape(1, -1))
        goal_bbox = self.getbbox(goal_img_xy, half_size=half_size) # bounding box (x1, y1, x2, y2) 
        anchors = self.getAllAnchorsWorld()
        anchors = self.env.sensor.world2cam(anchors)
      all_bbox = self.getbbox(anchors, half_size=half_size)
      
    joint_angles = self.env.robot.getJointAngles()[:7]
    # ee_pos = np.array([ee_x, ee_y, ee_z, *ee_rot, *joint_angles])
    # print(state)
    ee_pos = np.concatenate([np.array([state]), self.scaleXYZR([ee_x, ee_y, ee_z, ee_rot[2]])])
    sub_traj_id = self.getSubTrajID()

    # high_level_info = self.getHighLevelInfo()

    return state, self.global_obs, self.in_hand, goal_bbox, all_bbox, ee_pos, sub_traj_id, high_level_info

  def getSubTrajID(self):
    # need to be implemented in subclass
    return None
  
  def getHighLevelInfo(self):
    # need to be implemented in subclass
    return None

  def initQ(self, time_horizon):
    self.state_q = queue.Queue(time_horizon)
    self.global_obs_q = queue.Queue(time_horizon)
    self.goal_bbox_q = queue.Queue(time_horizon)
    self.all_bbox_q = queue.Queue(time_horizon)
    self.in_hand_q = queue.Queue(time_horizon)
    self.ee_pos_q = queue.Queue(time_horizon)
  
  def initA(self, time_horizon):
    self.action_q = queue.Queue(time_horizon)

  def insertQ(self, state, global_obs, in_hand, goal_bbox, all_bbox, ee_pos):
    if self.global_obs_q.full():
      self.state_q.get()
      self.global_obs_q.get()
      self.goal_bbox_q.get()
      self.all_bbox_q.get()
      self.in_hand_q.get()
      self.ee_pos_q.get()
      self.state_q.put(state)
      self.global_obs_q.put(global_obs)
      self.goal_bbox_q.put(goal_bbox)
      self.all_bbox_q.put(all_bbox)
      self.in_hand_q.put(in_hand)
      self.ee_pos_q.put(ee_pos)
    else:
      self.state_q.put(state)
      self.global_obs_q.put(global_obs)
      self.goal_bbox_q.put(goal_bbox)
      self.all_bbox_q.put(all_bbox)
      self.in_hand_q.put(in_hand)
      self.ee_pos_q.put(ee_pos)

  def insertA(self, action):
    if self.action_q.full():
      self.action_q.get()
      self.action_q.put(action)
    else:
      self.action_q.put(action)

  def Q2Array(self, Q):
    return np.stack(list(Q.queue))

  def populateQ(self, state, global_obs, in_hand, goal_bbox, all_bbox, ee_pos):
    while not self.global_obs_q.full():
      self.state_q.put(state)
      self.global_obs_q.put(global_obs)
      self.in_hand_q.put(in_hand)
      self.goal_bbox_q.put(goal_bbox)
      self.all_bbox_q.put(all_bbox)
      self.ee_pos_q.put(ee_pos)

  def populateA(self, action):
    while not self.action_q.full():
      self.action_q.put(action)
    

  def getObsTemporalBBox(self, done):
      # global_obs is (c, 128, 128)
      state, global_obs, in_hand, goal_bbox, all_bbox, ee_pos, sub_traj_id, high_level_info = self.getNextGoal()
      if not self.init_q:
        self.init_q = True
        self.initQ(self.time_horizon)
        self.populateQ(state, global_obs, in_hand, goal_bbox, all_bbox, ee_pos)
      else:
        if done:
          self.init_q = False
          return self.Q2Array(self.state_q), self.Q2Array(self.global_obs_q), self.Q2Array(self.in_hand_q), self.Q2Array(self.goal_bbox_q), self.Q2Array(self.all_bbox_q), self.Q2Array(self.ee_pos_q), sub_traj_id, high_level_info
        self.insertQ(state, global_obs, in_hand, goal_bbox, all_bbox, ee_pos)
      
      return self.Q2Array(self.state_q), self.Q2Array(self.global_obs_q), self.Q2Array(self.in_hand_q), self.Q2Array(self.goal_bbox_q), self.Q2Array(self.all_bbox_q), self.Q2Array(self.ee_pos_q), sub_traj_id, high_level_info

  def getObsTemporal(self):
      # global_obs is (c, 128, 128)
      state, global_obs, in_hand, all_bbox, goal_bbox, ee_pos, sub_traj_id, high_level_info = self.getNextGoal()
      if not self.init_q:
        self.init_q = True
        self.initQ(self.time_horizon)
        self.populateQ(state, global_obs, in_hand, all_bbox, goal_bbox, ee_pos)
      else:
        if self.env.done:
          self.init_q = False
          return self.Q2Array(self.state_q), self.Q2Array(self.global_obs_q), self.Q2Array(self.in_hand_q), self.Q2Array(self.ee_pos_q)
        self.insertQ(state, global_obs, in_hand, all_bbox, goal_bbox, ee_pos)
      
      return self.Q2Array(self.state_q), self.Q2Array(self.global_obs_q), self.Q2Array(self.in_hand_q), self.Q2Array(self.ee_pos_q)
  
  def getNextTemporalAction(self, action=None):
    if action.any() == None:
      action = self.getNextAction()

    if not self.init_a:
        self.init_a = True
        self.initA(self.time_horizon)
        self.populateA(action)
    else:
      if self.env.done:
        self.init_a = False
        self.insertA(action)
        return self.Q2Array(self.action_q)
      self.insertA(action)
      

    return self.Q2Array(self.action_q)



  def getHighMaskStep(self, trajectory, teacher_obs_dim=8):
    standardized_teacher_traj = np.zeros([self.max_teacher_length, teacher_obs_dim])
    standardized_teacher_traj[:len(trajectory), :] = trajectory
    
    teacher_traj_mask = np.zeros([self.max_teacher_length])
    if self.high_traj_type == 'entire':
      teacher_traj_mask[:len(trajectory)] = 1
    elif self.high_traj_type.find('goal') > -1:
      teacher_traj_mask[len(trajectory)-1] = 1
    elif self.high_traj_type == 'None':
      pass

    teacher_time_steps = np.zeros([self.max_teacher_length])
    teacher_time_steps[:len(trajectory)] = np.arange(len(trajectory))
    return standardized_teacher_traj, teacher_traj_mask, teacher_time_steps
  
  def getHighLevelTraj(self):
    self.env.reward=0
    self.target_obj=None
    done = False
    teacher_traj = []
    while not done:
      action = self.getNextAction()
      state, global_obs, in_hand, goal_bbox, all_bbox, ee_pos, sub_traj_id, high_level_info = self.getNextGoal()
      teacher_traj.append(high_level_info)
      
      _, _, done = self.env.step(action)

    teacher_traj = np.array(teacher_traj)
    teacher_obs_dim = teacher_traj.shape[-1]
    standardized_teacher_traj, mask, time_steps = self.getHighMaskStep(teacher_traj, teacher_obs_dim)
    self.env.recover()
    return standardized_teacher_traj, mask, time_steps
  # def getObsTemporal(self):
  #   # global_obs is (c, 128, 128)
  #   global_obs, _, goal_bbox, all_bbox = self.getNextGoal()
  #   if not self.init_q:
  #     self.init_q = True
  #     self.initQ(self.time_horizon)
  #     self.insertQ(global_obs, goal_bbox, all_bbox)
  #   else:
  #     self.insertQ(global_obs, goal_bbox, all_bbox)
  #   if self.global_obs_q.full():
  #     return self.Q2Array(self.global_obs_q), self.Q2Array(self.goal_bbox_q), self.Q2Array(self.all_bbox_q)
  #   else:
  #     return None, None, None

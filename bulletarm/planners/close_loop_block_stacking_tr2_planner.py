import numpy as np

from bulletarm.pybullet.utils import constants
from bulletarm.planners.close_loop_planner import CloseLoopPlanner
from bulletarm.pybullet.utils import transformations
import queue

class CloseLoopBlockStackingTR2Planner(CloseLoopPlanner):
  def __init__(self, env, config):
    super().__init__(env, config)
    # self.time_horizon = self.env.time_horizon
    # self.pick_place_stage = 0
    # # self.post_pose_reached = False
    # # pos, rot, primitive
    # self.current_target = None
    # self.target_obj = None

    # self.last_target_obj = None
    # self.global_obs = None
    # self.goal_obs = None
    # self.init_q = False
    # self.global_obs_q = None
    # self.goal_bbox_q = None
    # self.all_bbox_q = None
    # self.in_hand = None
    # self.in_hand_q = None
    # self.ee_pos_q = None
    # # self.all_obs = None # (N, x1, y1, x2, y2)


  def getNextActionToCurrentTarget(self):
    # print(self.target_obj)
    x, y, z, r = self.getActionByGoalPose(self.current_target[0], self.current_target[1])
    if np.all(np.abs([x, y, z]) < self.dpos) and np.abs(r) < self.drot:
      primitive = constants.PICK_PRIMATIVE if self.current_target[2] is constants.PICK_PRIMATIVE else constants.PLACE_PRIMATIVE
      self.current_target = None
    else:
      primitive = constants.PICK_PRIMATIVE if self.isHolding() else constants.PLACE_PRIMATIVE
    return self.env._encodeAction(primitive, x, y, z, r)

  def setNewTarget(self):
    objects = np.array(list(filter(lambda x: not self.isObjectHeld(x) and self.isObjOnTop(x), self.env.objects)))
    object_poses = self.env.getObjectPoses(objects)
    sorted_inds = np.flip(np.argsort(object_poses[:,2], axis=0))
    objects = objects[sorted_inds]

    if self.env.current_episode_steps == 1:
      self.pick_place_stage = 0

    if self.pick_place_stage in [0, 1, 2]:
      if self.target_obj is None:
        self.target_obj = objects[1] if len(objects) > 1 else objects[0]
        # self.goal_obj = self.target_obj
      object_pos = self.target_obj.getPosition()
      object_rot = list(transformations.euler_from_quaternion(self.target_obj.getRotation()))
      gripper_rz = transformations.euler_from_quaternion(self.env.robot._getEndEffectorRotation())[2]
      while object_rot[2] - gripper_rz > np.pi / 4:
        object_rot[2] -= np.pi / 2
      while object_rot[2] - gripper_rz < -np.pi / 4:
        object_rot[2] += np.pi / 2
      pre_pick_pos = object_pos[0], object_pos[1], 0.15
      if self.pick_place_stage == 0:
        self.pick_place_stage = 1
        self.current_target = (pre_pick_pos, object_rot, constants.PLACE_PRIMATIVE)
      elif self.pick_place_stage == 1:
        self.pick_place_stage = 2
        self.current_target = (object_pos, object_rot, constants.PICK_PRIMATIVE)
      else:
        self.pick_place_stage = 3
        self.target_obj = None
        self.current_target = (pre_pick_pos, object_rot, constants.PICK_PRIMATIVE)

    else:
      if self.target_obj is None:
        self.target_obj = objects[0]
      # self.last_target_obj = self.target_obj
      object_pos = self.target_obj.getPosition()
      object_rot = list(transformations.euler_from_quaternion(self.target_obj.getRotation()))
      gripper_rz = transformations.euler_from_quaternion(self.env.robot._getEndEffectorRotation())[2]
      while object_rot[2] - gripper_rz > np.pi / 4:
        object_rot[2] -= np.pi / 2
      while object_rot[2] - gripper_rz < -np.pi / 4:
        object_rot[2] += np.pi / 2
      
      pre_place_pos = object_pos[0], object_pos[1], object_pos[2] + 0.1
      if self.pick_place_stage == 3:
        self.pick_place_stage = 4
        self.current_target = (pre_place_pos, object_rot, constants.PICK_PRIMATIVE)
      elif self.pick_place_stage == 4:
        self.pick_place_stage = 5
        if self.target_obj.object_type_id == 31:
          place_pos = object_pos[0], object_pos[1], object_pos[2]
        else:
          place_pos = object_pos[0], object_pos[1], object_pos[2] + self.getMaxBlockSize() * 1.2
        self.current_target = (place_pos, object_rot, constants.PLACE_PRIMATIVE)
      else:
        self.pick_place_stage = 0
        self.target_obj = None
        self.current_target = (pre_place_pos, object_rot, constants.PLACE_PRIMATIVE)

  def getNextAction(self):
    
    # print(self.current_target)
    if self.env.current_episode_steps == 1:
      self.pick_place_stage = 0
      self.current_target = None
    if self.current_target is not None:
      return self.getNextActionToCurrentTarget()
    else:
      self.setNewTarget()
      return self.getNextActionToCurrentTarget()
    
  def getSubTrajID(self):
    return 0
  
  def getHighLevelInfo(self, info_type='tr2'):
    if info_type == 'tr2':
       ee_xyz = self.env.robot._getEndEffectorPosition()
       goal_xyz = self.target_obj.getPosition()
       high_level_info = np.concatenate([ee_xyz, goal_xyz])
    else:
      NotImplementedError
    return high_level_info
  
  



    # if not self.env._isHolding():
    #   self.pre_pose_reached = False
    #   block_pos = self.env.objects[0].getPosition()
    #   block_rot = transformations.euler_from_quaternion(self.env.objects[0].getRotation())
    #
    #   x, y, z, r = self.getActionByGoalPose(block_pos, block_rot)
    #
    #   if np.all(np.abs([x, y, z]) < self.dpos) and np.abs(r) < self.drot:
    #     primitive = constants.PICK_PRIMATIVE
    #   else:
    #     primitive = constants.PLACE_PRIMATIVE
    #
    # else:
    #   block_pos = self.env.objects[1].getPosition()
    #   block_rot = transformations.euler_from_quaternion(self.env.objects[1].getRotation())
    #
    #   pre_place_pos = block_pos[0], block_pos[1], 0.1
    #   x, y, z, r = self.getActionByGoalPose(pre_place_pos, block_rot)
    #   primitive = constants.PICK_PRIMATIVE
    #   if np.all(np.abs([x, y, z]) < 0.005) and np.abs(r) < np.pi/12:
    #     self.pre_pose_reached = True
    #
    #   if self.pre_pose_reached:
    #     place_pos = block_pos[0], block_pos[1], block_pos[2] + self.getMaxBlockSize()
    #     x, y, z, r = self.getActionByGoalPose(place_pos, block_rot)
    #     if np.all(np.abs([x, y, z]) < 0.005) and np.abs(r) < np.pi / 12:
    #       primitive = constants.PLACE_PRIMATIVE
    #
    #
    #   # if np.all(np.abs([x, y, z]) < 0.005) and np.abs(r) < np.pi/12:
    #   #   self.pre_pose_reached = True
    #   #   place_pos = block_pos[0], block_pos[1], block_pos[2] + self.getMaxBlockSize()/2
    #   #   x, y, z, r = self.getActionByGoalPose(place_pos, block_rot)
    #   #   if np.all(np.abs([x, y, z]) < 0.005) and np.abs(r) < np.pi/12:
    #   #     primitive = constants.PLACE_PRIMATIVE
    # return self.env._encodeAction(primitive, x, y, z, r)

  def getStepsLeft(self):
    return 100

    
  # def getAllAnchors(self):
  #   # (N, x, y) in global_obs image space
  #   object_poses = self.env.getObjectPoses(self.env.objects)
  #   return self.space2img(object_poses[:,:2])

  # def getAllAnchorsWorld(self):
  #   # (N, x, y) in global_obs image space
  #   object_poses = self.env.getObjectPoses(self.env.objects)
  #   gripper_pos = self.env.robot._getEndEffectorPosition().reshape(1,3)
  #   poses = np.concatenate([object_poses[:,:3], gripper_pos], axis=0)
  #   return poses

  # def centercrop(self, img, size=14):
  #   # crop around center
  #   img_x, img_y = img.shape
  #   return img[img_x//2-size:img_x//2+size, img_y//2-size:img_y//2+size]

  # def getbbox(self, anchor, half_size=6):
  #   if len(anchor.shape) == 1:
  #     # (x1, y1, x2, y2)
  #     return np.array([anchor[0]-half_size, anchor[1]-half_size, anchor[0]+half_size, anchor[1]+half_size])
  #   elif len(anchor.shape) > 1:
  #     # (N, x1, y1, x2, y2)
  #     return np.vstack((anchor[:,0]-half_size, anchor[:,1]-half_size, anchor[:,0]+half_size, anchor[:,1]+half_size)).T
  #   else:
  #     NotImplementedError

  # def obscrop(self, global_obs, bbox):
  #   # crop around anchor
  #   if len(bbox.shape) == 1:
  #     return global_obs[bbox[0]:bbox[2], bbox[1]:bbox[3]]
  #   elif len(bbox.shape) > 1:
  #     bbox_obs = []
  #     for i in range(bbox.shape[0]):
  #       bbox_obs.append(global_obs[bbox[i,0]:bbox[i,2], bbox[i,1]:bbox[i,3]])
  #     return bbox_obs
  #   else:
  #     NotImplementedError

  # def space2img(self, anchor):
  #   view_scale = self.env.view_scale
  #   work_space = self.env.workspace
  #   workspace_size =  work_space[0,1] - work_space[0,0]
  #   center_shift = np.array([128,128])/2 - np.array([128,128])/2/view_scale
  #   if len(anchor.shape) == 1:
  #     # Convert coordinates in workspace (x,y) to img indices. (only for center scaling)
  #     x_in_ws, y_in_ws = anchor
  #     x_y_idx = np.array([(x_in_ws-work_space[0,0])/workspace_size*128, (y_in_ws-work_space[1,0])/workspace_size*128])
  #     xy_in_img = (x_y_idx/view_scale + center_shift).astype(int)
  #     return xy_in_img
  #   elif len(anchor.shape) > 1:
  #     # Convert coordinates in workspace (N, x, y) to img indices. (only for center scaling)
  #     xs = (anchor[:,0]-work_space[0,0])/workspace_size*128
  #     ys = (anchor[:,1]-work_space[1,0])/workspace_size*128
  #     xys_in_img = (np.array([xs, ys]).T/view_scale + center_shift).astype(int)
  #     return xys_in_img
  #   else:
  #     NotImplementedError


  # def getNextGoal(self):
  #   # print(self.target_obj)
  #   half_size = 10
  #   if self.target_obj is None:
  #     object_x, object_y, object_z = self.last_target_obj.getPosition()[:3]
  #   else:
  #     object_x, object_y, object_z = self.target_obj.getPosition()[:3]
  #   ee_x, ee_y, ee_z = self.env.robot._getEndEffectorPosition()
  #   # object_rot = list(transformations.euler_from_quaternion(self.target_obj.getRotation()))
  #   # gripper_rz = transformations.euler_from_quaternion(self.env.robot._getEndEffectorRotation())[2]
  #   # self.goal_obs = self.centercrop(self.env._getHeightmap(gripper_pos=[object_x, object_y, ee_z], gripper_rz=None))
  #   # self.global_obs = self.env._getHeightmap(gripper_pos=[self.env.workspace[0].mean(), self.env.workspace[1].mean(), 0.2], gripper_rz=None).reshape(1,128,128)
  #   _, self.in_hand, self.global_obs = self.env._getObservation()
  #   ee_rot = transformations.euler_from_quaternion(self.env.robot._getEndEffectorRotation())
  #   # print(object_x, object_y)
  #   self.last_target_obj = self.target_obj if self.target_obj is not None else self.last_target_obj
  #   # get goal anchor and obs
  #   if self.env.view_type.find('side') == -1:
  #     goal_img_xy = self.space2img(np.array([object_x, object_y, object_z]).reshape(1, -1))
  #     goal_bbox = self.getbbox(goal_img_xy) # bounding box (x1, y1, x2, y2) 
  #     anchors = self.getAllAnchors()
  #   else:
  #     # self.env.sensor.getPointCloud(128)
  #     goal_img_xy = self.env.sensor.world2cam(np.array([object_x, object_y, object_z]).reshape(1, -1))
  #     goal_bbox = self.getbbox(goal_img_xy, half_size=half_size) # bounding box (x1, y1, x2, y2) 
  #     anchors = self.getAllAnchorsWorld()
  #     anchors = self.env.sensor.world2cam(anchors)
  #   # get all anchors and obs
    
  #   all_bbox = self.getbbox(anchors, half_size=half_size)
  #   joint_angles = self.env.getJointAngles()
  #   ee_pos = np.array([ee_x, ee_y, ee_z, *ee_rot, *joint_angles])
  #   return self.global_obs, self.in_hand, goal_bbox, all_bbox, ee_pos

  # def initQ(self, time_horizon):
  #   self.global_obs_q = queue.Queue(time_horizon)
  #   self.goal_bbox_q = queue.Queue(time_horizon)
  #   self.all_bbox_q = queue.Queue(time_horizon)
  #   self.in_hand_q = queue.Queue(time_horizon)
  #   self.ee_pos_q = queue.Queue(time_horizon)

  # def insertQ(self, global_obs, in_hand, goal_bbox, all_bbox, ee_pos):
  #   if self.global_obs_q.full():
  #     self.global_obs_q.get()
  #     self.goal_bbox_q.get()
  #     self.all_bbox_q.get()
  #     self.in_hand_q.get()
  #     self.ee_pos_q.get()
  #     self.global_obs_q.put(global_obs)
  #     self.goal_bbox_q.put(goal_bbox)
  #     self.all_bbox_q.put(all_bbox)
  #     self.in_hand_q.put(in_hand)
  #     self.ee_pos_q.put(ee_pos)
  #   else:
  #     self.global_obs_q.put(global_obs)
  #     self.goal_bbox_q.put(goal_bbox)
  #     self.all_bbox_q.put(all_bbox)
  #     self.in_hand_q.put(in_hand)
  #     self.ee_pos_q.put(ee_pos)

  # def Q2Array(self, Q):
  #   return np.stack(list(Q.queue))

  # def populateQ(self, global_obs, in_hand, goal_bbox, all_bbox, ee_pos):
  #   while not self.global_obs_q.full():
  #     self.global_obs_q.put(global_obs)
  #     self.in_hand_q.put(in_hand)
  #     self.goal_bbox_q.put(goal_bbox)
  #     self.all_bbox_q.put(all_bbox)
  #     self.ee_pos_q.put(ee_pos)
    

  # def getObsTemporal(self, done):
  #     # global_obs is (c, 128, 128)
  #     global_obs, in_hand, goal_bbox, all_bbox, ee_pos = self.getNextGoal()
  #     if not self.init_q:
  #       self.init_q = True
  #       self.initQ(self.time_horizon)
  #       self.populateQ(global_obs, in_hand, goal_bbox, all_bbox, ee_pos)
  #     else:
  #       if done:
  #         self.init_q = False
  #         return self.Q2Array(self.global_obs_q), self.Q2Array(self.in_hand_q), self.Q2Array(self.goal_bbox_q), self.Q2Array(self.all_bbox_q), self.Q2Array(self.ee_pos_q)
  #       self.insertQ(global_obs, in_hand, goal_bbox, all_bbox, ee_pos)
      
  #     return self.Q2Array(self.global_obs_q), self.Q2Array(self.in_hand_q), self.Q2Array(self.goal_bbox_q), self.Q2Array(self.all_bbox_q), self.Q2Array(self.ee_pos_q)

  # # def getObsTemporal(self):
  # #   # global_obs is (c, 128, 128)
  # #   global_obs, _, goal_bbox, all_bbox = self.getNextGoal()
  # #   if not self.init_q:
  # #     self.init_q = True
  # #     self.initQ(self.time_horizon)
  # #     self.insertQ(global_obs, goal_bbox, all_bbox)
  # #   else:
  # #     self.insertQ(global_obs, goal_bbox, all_bbox)
  # #   if self.global_obs_q.full():
  # #     return self.Q2Array(self.global_obs_q), self.Q2Array(self.goal_bbox_q), self.Q2Array(self.all_bbox_q)
  # #   else:
  # #     return None, None, None



  
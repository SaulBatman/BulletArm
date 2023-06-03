import pybullet as pb
import numpy as np
from scipy import stats
from bulletarm.envs.base_env import BaseEnv
from bulletarm.pybullet.utils import transformations
from bulletarm.pybullet.utils.renderer import Renderer
from bulletarm.pybullet.utils.ortho_sensor import OrthographicSensor
from bulletarm.pybullet.utils.sensor import Sensor
from bulletarm.pybullet.equipments.tray import Tray
from scipy.ndimage import rotate
import cv2

class CloseLoopEnv(BaseEnv):
  def __init__(self, config):
    if 'workspace' not in config:
      config['workspace'] = np.asarray([[0.3, 0.6],
                                        [-0.15, 0.15],
                                        [0.01, 0.25]])
    if 'robot' not in config:
      config['robot'] = 'kuka'
    if 'action_sequence' not in config:
      config['action_sequence'] = 'pxyzr'
    if 'max_steps' not in config:
      config['max_steps'] = 50
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [1, 1]
    if 'view_type' not in config:
      config['view_type'] = 'camera_center_xyz'
    if 'obs_type' not in config:
      config['obs_type'] = 'pixel'
    if 'view_scale' not in config:
      config['view_scale'] = 1.5
    if 'close_loop_tray' not in config:
      config['close_loop_tray'] = False
    super().__init__(config)
    self.view_type = config['view_type']
    self.obs_type = config['obs_type']
    assert self.view_type in ['render_center', 'render_center_height', 'render_fix', 'camera_center_xyzr', 'camera_center_xyr',
                              'camera_center_xyz', 'camera_center_xy', 'camera_fix', 'camera_center_xyr_height',
                              'camera_center_xyz_height', 'camera_center_xy_height', 'camera_fix_height',
                              'camera_center_z', 'camera_center_z_height', 'pers_center_xyz', 'camera_side',
                              'camera_side_rgbd', 'camera_side_height', 'camera_side_viola', 'camera_side_viola_rgbd',
                              "camera_side_viola_rgbd_custom_1", "camera_side_viola_rgbd_custom_2",
                               "ws_center_fix", "ws_center_fix_height"]
    self.view_scale = config['view_scale']
    self.robot_type = config['robot']
    if config['robot'] == 'kuka':
      self.robot.home_positions = [-0.4446, 0.0837, -2.6123, 1.8883, -0.0457, -1.1810, 0.0699, 0., 0., 0., 0., 0., 0., 0., 0.]
      self.robot.home_positions_joint = self.robot.home_positions[:7]

    self.has_tray = config['close_loop_tray']
    self.bin_size = self.workspace_size - 0.05
    self.tray = None
    if self.has_tray:
      self.tray = Tray()
    
    self.workspace = config['workspace']
    # if self.view_type.find('center') > -1:
    #   self.ws_size *= 1.5

    self.renderer = None
    self.pers_sensor = None
    self.obs_size_m = self.workspace_size * self.view_scale
    self.initSensor()

    self.simulate_z_threshold = self.workspace[2][0] + 0.07

    self.simulate_pos = None
    self.simulate_rot = None

    self.time_horizon = config['time_horizon']
    # self.cloud = None

  def initialize(self):
    super().initialize()
    ws_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[self.workspace_size / 2, self.workspace_size / 2, 0.001],
                                     rgbaColor=[0.2, 0.2, 0.2, 1])
    ws_id = pb.createMultiBody(baseMass=0,
                               baseVisualShapeIndex=ws_visual,
                               basePosition=[self.workspace[0].mean(), self.workspace[1].mean(), 0],
                               baseOrientation=[0, 0, 0, 1])
    if self.has_tray:
      self.tray.initialize(pos=[self.workspace[0].mean(), self.workspace[1].mean(), 0],
                           size=[self.bin_size, self.bin_size, 0.1])

  def initSensor(self):
    cam_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 0.29]
    target_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 0]
    cam_up_vector = [-1, 0, 0]
    self.sensor = OrthographicSensor(cam_pos, cam_up_vector, target_pos, self.obs_size_m, 0.1, 1)
    self.sensor.setCamMatrix(cam_pos, cam_up_vector, target_pos)
    self.renderer = Renderer(self.workspace)
    self.pers_sensor = Sensor(cam_pos, cam_up_vector, target_pos, self.obs_size_m, cam_pos[2] - 1, cam_pos[2])

  def _getValidOrientation(self, random_orientation):
    if random_orientation:
      orientation = pb.getQuaternionFromEuler([0., 0., np.pi * (np.random.random_sample() - 0.5)])
    else:
      orientation = pb.getQuaternionFromEuler([0., 0., 0.])
    return orientation

  def resetPybulletWorkspace(self):
    self.renderer.clearPoints()
    super().resetPybulletWorkspace()
    self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
    self.simulate_pos = self.robot._getEndEffectorPosition()
    self.simulate_rot = transformations.euler_from_quaternion(self.robot._getEndEffectorRotation())

  def step(self, action):
    p, x, y, z, rot = self._decodeAction(action)
    current_pos = self.robot._getEndEffectorPosition()
    current_rot = list(transformations.euler_from_quaternion(self.robot._getEndEffectorRotation()))
    if self.action_sequence.count('r') == 1:
      current_rot[0] = 0
      current_rot[1] = 0

    # bTg = transformations.euler_matrix(0, 0, current_rot[-1])
    # bTg[:3, 3] = current_pos
    # gTt = np.eye(4)
    # gTt[:3, 3] = [x, y, z]
    # bTt = bTg.dot(gTt)
    # pos = bTt[:3, 3]

    pos = np.array(current_pos) + np.array([x, y, z])
    rot = np.array(current_rot) + np.array(rot)
    rot_q = pb.getQuaternionFromEuler(rot)
    pos[0] = np.clip(pos[0], self.workspace[0, 0], self.workspace[0, 1])
    pos[1] = np.clip(pos[1], self.workspace[1, 0], self.workspace[1, 1])
    pos[2] = np.clip(pos[2], self.workspace[2, 0], self.workspace[2, 1])
    self.robot.moveTo(pos, rot_q, dynamic=True)
    self.robot.controlGripper(p)
    self.robot.adjustGripperCommand()
    self.setRobotHoldingObj()
    self.renderer.clearPoints()
    obs = self._getObservation(action)
    valid = self.isSimValid()
    if valid:
      done = self._checkTermination()
      reward = 1.0 if done else 0.0
    else:
      done = True
      reward = 0
    if not done:
      done = self.current_episode_steps >= self.max_steps
    self.current_episode_steps += 1

    self.simulate_pos = pos
    self.simulate_rot = rot
    return obs, reward, done

  # def getJointAngles(self):
  #   # p, x, y, z, rot = self._decodeAction(action)
  #   current_pos = self.robot._getEndEffectorPosition()
  #   current_rot = transformations.euler_from_quaternion(self.robot._getEndEffectorRotation())
  #   # if self.action_sequence.count('r') == 1:
  #   #   current_rot[0] = 0
  #   #   current_rot[1] = 0

  #   return self.robot._calculateIK(current_pos, current_rot)


  def setRobotHoldingObj(self):
    self.robot.holding_obj = self.robot.getPickedObj(self.objects)

  def setRobotHoldingObjWithRotConstraint(self):
    self.robot.holding_obj = None
    for obj in self.objects:
      obj_rot = transformations.euler_from_quaternion(obj.getRotation())[-1]
      gripper_rot = transformations.euler_from_quaternion(self.robot._getEndEffectorRotation())[-1]
      angle_diff = abs(gripper_rot - obj_rot)
      angle_diff = min(angle_diff, abs(angle_diff - np.pi))
      angle_diff = min(angle_diff, abs(angle_diff - np.pi / 2))
      if len(pb.getContactPoints(self.robot.id, obj.object_id)) >= 2 and angle_diff < np.pi / 12:
        self.robot.holding_obj = obj
        break

  def _scaleX(self, x):
    scaled = 2 * (x - self.workspace[0, 0]) / (self.workspace[0, 1] - self.workspace[0, 0]) - 1
    return np.clip(scaled, -1, 1)

  def _scaleY(self, y):
    scaled = 2 * (y - self.workspace[1, 0]) / (self.workspace[1, 1] - self.workspace[1, 0]) - 1
    return np.clip(scaled, -1, 1)

  def _scaleZ(self, z):
    scaled = 2 * (z - self.workspace[2, 0]) / (self.workspace[2, 1] - self.workspace[2, 0]) - 1
    return np.clip(scaled, -1, 1)

  def _scaleRz(self, rz):
    while rz < -np.pi:
      rz += 2*np.pi
    while rz > np.pi:
      rz -= 2*np.pi
    scaled = 2 * (rz - -np.pi) / (2*np.pi) - 1
    return np.clip(scaled, -1, 1)

  def _scalePos(self, pos):
    return np.array([self._scaleX(pos[0]), self._scaleY(pos[1]), self._scaleZ(pos[2])])

  def getObjectPoses(self, objects=None):
    if objects is None: objects = self.objects

    obj_poses = list()
    for obj in objects:
      pos, rot = obj.getPose()
      rot = self.convertQuaternionToEuler(rot)

      obj_poses.append(pos + rot)
    return np.array(obj_poses)

  def _getVecObservation(self):
    '''
    get the observation in vector form. The observation has a size of (1+4+4*n), where the first value is the gripper
    state, the following 4-vector is the gripper's (x, y, z, rz), and the n 4-vectors afterwards are the (x, y, z, rz)s
    of the objects in the scene
    :return: the observation vector in np.array
    '''
    gripper_pos = self.robot._getEndEffectorPosition()
    scaled_gripper_pos = self._scalePos(gripper_pos)
    gripper_rz = transformations.euler_from_quaternion(self.robot._getEndEffectorRotation())[2]
    scaled_gripper_rz = self._scaleRz(gripper_rz)
    obj_poses = self.getObjectPoses()
    obj_poses = np.stack((obj_poses[:, 0], obj_poses[:, 1], obj_poses[:, 2], obj_poses[:, 5]), 1)
    scaled_obj_poses = []
    for i in range(obj_poses.shape[0]):
      scaled_obj_poses.append(
        np.concatenate([self._scalePos(obj_poses[i, :3]), np.array([self._scaleRz(obj_poses[i, 3])])]))
    scaled_obj_poses = np.concatenate(scaled_obj_poses)
    gripper_state = self.robot.getGripperOpenRatio()
    gripper_state = gripper_state * 2 - 1
    gripper_state = np.clip(gripper_state, -1, 1)
    # gripper_state = 1 if self._isHolding() else -1
    if self.obs_type == 'state_tr2':
      # returns (15+15+1+3+1)=35 vector obs
      # normalize angles and vel with max_angle=+-10, max_vel=+-10 constrain
      joint_angles = self.robot.getJointAngles()/10
      joint_vels = self.robot.getJointVels()/10
      gripper_state = 1 if self._isHolding() else -1
      obs = np.concatenate([joint_angles, joint_vels, np.array([gripper_state]), scaled_gripper_pos, np.array([scaled_gripper_rz])])
      return obs
    else:
      obs = np.concatenate(
        [np.array([gripper_state]), scaled_gripper_pos, np.array([scaled_gripper_rz]), scaled_obj_poses])
      return obs

  def world2pixel(self, gripper_pos, offset=[0,0]):
    workspace_size = self.workspace[0][1]-self.workspace[0][0]
    x = (gripper_pos[0] - self.workspace[0][0])/workspace_size
    y = (gripper_pos[1] - self.workspace[1][0])/workspace_size
    return np.array([x*128+offset[0],y*128+offset[1]]).astype(int)


  
  def _getObservation(self, action=None):
    ''''''
    if self.obs_type == 'pixel':
      self.heightmap = self._getHeightmap()
      heightmap = self.heightmap
      # draw gripper if view is centered at the gripper
      if self.view_type in ['camera_center_xyz', 'camera_center_xyz_height', 'render_center', 'render_center_height']:
        gripper_img = self.getGripperImg()
        if self.view_type.find('height') > -1:
          gripper_pos = self.robot._getEndEffectorPosition()
          heightmap[gripper_img == 1] = gripper_pos[2]
        else:
          heightmap[gripper_img == 1] = 0
      # add channel dimension if view is depth only
      

      
      # gripper_img = self.getGripperImg()
      # in_hand_img = self._getHeightmapInHand()
      gripper_pos = self.robot._getEndEffectorPosition()
      # gripper_img = self.getGripperImg()
      # heightmap[gripper_img == 1] = 0

      in_hand_img = self._getHeightmapInHand(gripper_pos)
      # in_hand_img = [cv2.resize(img, dsize=(128, 128), interpolation = cv2.INTER_LINEAR) for img in in_hand_img]
      # in_hand_img = np.stack(in_hand_img)
      # self.sensor.getPointCloud(512)

      # gripper_in_image_coor = self.sensor.world2pixel(gripper_pos)
      
      # workspace_size = 
      # in_hand_img = heightmap.copy()
      # in_hand_img[gripper_img == 1] = 0
      # crop_size_half = 16

      # vals,counts = np.unique(in_hand_img, return_counts=True)
      # index = np.argmax(counts)
      # base_img = np.ones([1, 160, 160])*vals[index]
      # base_img[:, 80-64:80+64, 80-64:80+64] = in_hand_img
      # gripper_in_image_coor = self.world2pixel(gripper_pos) # +16 is because of the padding
      # in_hand_img = in_hand_img[:, gripper_in_image_coor[0]-crop_size_half:gripper_in_image_coor[0]+crop_size_half, gripper_in_image_coor[1]-crop_size_half:gripper_in_image_coor[1]+crop_size_half]# (1, 32, 32)


      if self.view_type.find('rgb') == -1:
        heightmap = heightmap.reshape([1, self.heightmap_size, self.heightmap_size])
      return self._isHolding(), in_hand_img, heightmap
    else:
      obs = self._getVecObservation()
      in_hand = 0
      return self._isHolding(), in_hand, obs

  def _getHeightmapInHand(self, gripper_pos=None):
    # assume "camera_center_xyz"
    def add_gripper(heightmap, gripper_img, channel, img_type='depth'):
      if img_type != 'depth':
        gripper_pos = self.robot._getEndEffectorPosition()
        heightmap[channel][gripper_img == 1] = gripper_pos[2]
      else:
        heightmap[channel][gripper_img == 1] = 0
      return heightmap
    crop_size = 45
    heightmap_size = self.heightmap_size
    gripper_z_offset = 0.04 # panda
    if self.robot_type == 'kuka':
      gripper_z_offset = 0.12
      # gripper_z_offset = 0 # for visualization
    elif self.robot_type == 'ur5':
      gripper_z_offset = 0.06
    if gripper_pos is None:
      gripper_pos = self.robot._getEndEffectorPosition()
    # if self.view_type.find('side') == -1:
    gripper_pos[2] += gripper_z_offset
    # gripper_pos[2] = self.workspace[2].max()
    target_pos = [gripper_pos[0], gripper_pos[1], 0]
    cam_up_vector = [-1, 0, 0]
    cam_pos = [gripper_pos[0], gripper_pos[1], 0.29]
    view_scale=0.6
    obs_size_m = self.workspace_size * view_scale
    tem_sensor = OrthographicSensor(cam_pos, cam_up_vector, target_pos, obs_size_m, 0.1, 1)
    tem_sensor.setCamMatrix(gripper_pos, cam_up_vector, target_pos)
    gripper_img = self.getGripperImg(obs_size_m=obs_size_m)
    if self.view_type.find('rgbd') > -1:
      rgb_img = tem_sensor.getRGBImg(heightmap_size)
      depth_img = tem_sensor.getDepthImg(heightmap_size).reshape(1, heightmap_size, heightmap_size)
      depth = np.concatenate([rgb_img, depth_img])
      depth = add_gripper(depth, gripper_img, channel=3)
    elif self.view_type.find('rgb') > -1:
      depth = tem_sensor.getDepthImg(heightmap_size).reshape([-1, heightmap_size, heightmap_size])
      # depth = add_gripper(heightmap, gripper_img, channel)
    else: # only depth
      depth = tem_sensor.getDepthImg(heightmap_size).reshape([-1, heightmap_size, heightmap_size])
      depth = add_gripper(depth, gripper_img, channel=0)
    # gripper_pos = self.robot._getEndEffectorPosition()

    # gripper_img = self.getGripperImg(heightmap_size=160)
    # if self.view_type.find('height') > -1:
    #     depth = -heightmap + gripper_pos[2]
    #     # depth[gripper_img == 1] = 0
    #   else:
    #     depth = heightmap
    #     return depth
    

    
    
    # height_center = heightmap_size//2
    # depth = depth[:, height_center-crop_size:height_center+crop_size, height_center-crop_size:height_center+crop_size]

    # else:
    #   if self.view_type in ['camera_side_viola', 'camera_side_viola_rgbd', 'camera_side_viola_height']:
    #     cam_pos = [-0.7, self.workspace[1].mean()-1.2, 0.9]
    #     target_pos = [self.workspace[0].mean()-0.2, self.workspace[1].mean(), 0.3]
    #     cam_up_vector = [0, 0, 3]
    #     self.sensor = Sensor(cam_pos, cam_up_vector, target_pos, 0.7, 0.3, 3)
    #     self.sensor.fov = 35
    #     self.sensor.proj_matrix = pb.computeProjectionMatrixFOV(self.sensor.fov, 1, self.sensor.near, self.sensor.far)
    #     if self.view_type == 'camera_side_viola':
    #       depth = self.sensor.getDepthImg(self.heightmap_size)
    #     elif self.view_type == 'camera_side_viola_rgbd':
    #       rgb_img = self.sensor.getRGBImg(self.heightmap_size)
    #       depth_img = self.sensor.getDepthImg(self.heightmap_size).reshape(1, self.heightmap_size, self.heightmap_size)
    #       depth = np.concatenate([rgb_img, depth_img])
    #     else:
    #       depth = self.sensor.getHeightmap(self.heightmap_size)
          
        
      # else:
      #   NotImplementedError
    
    # if self.view_type.find('rgb') == -1:
    #     depth = depth.reshape([1, crop_size*2, crop_size*2])
    # else:
    #     depth = depth.reshape([1, crop_size*2, crop_size*2])

    return depth

  def simulate(self, action):
    flag = True
    p, dx, dy, dz, r = self._decodeAction(action)
    dtheta = r[2]
    # pos = list(self.robot._getEndEffectorPosition())
    # gripper_rz = transformations.euler_from_quaternion(self.robot._getEndEffectorRotation())[2]
    pos = self.simulate_pos
    gripper_rz = self.simulate_rot[2]
    pos[0] += dx
    pos[1] += dy
    pos[2] += dz
    temp1, temp2, temp3 = pos[0].copy(), pos[1].copy(), pos[2].copy()
    pos[0] = np.clip(pos[0], self.workspace[0, 0], self.workspace[0, 1])
    pos[1] = np.clip(pos[1], self.workspace[1, 0], self.workspace[1, 1])
    pos[2] = np.clip(pos[2], self.simulate_z_threshold, self.workspace[2, 1])
    if (temp1!=pos[0]) or (temp2!=pos[1]) or (temp3!=pos[2]):
      flag=False
    gripper_rz += dtheta
    self.simulate_pos = pos
    self.simulate_rot = [0, 0, gripper_rz]
    # obs = self.renderer.getTopDownDepth(self.obs_size_m, self.heightmap_size, pos, 0)
    obs = self._getHeightmap(gripper_pos=self.simulate_pos, gripper_rz=gripper_rz)
    gripper_img = self.getGripperImg(p, gripper_rz)
    if self.view_type.find('height') > -1:
      obs[gripper_img == 1] = self.simulate_pos[2]
    else:
      obs[gripper_img == 1] = 0
    # gripper_img = gripper_img.reshape([1, self.heightmap_size, self.heightmap_size])
    # obs[gripper_img==1] = 0
    obs = obs.reshape([1, self.heightmap_size, self.heightmap_size])

    return self._isHolding(), None, obs, flag

  def resetSimPose(self):
    self.simulate_pos = np.array(self.robot._getEndEffectorPosition())
    self.simulate_rot = np.array(transformations.euler_from_quaternion(self.robot._getEndEffectorRotation()))

  def canSimulate(self):
    # pos = list(self.robot._getEndEffectorPosition())
    return not self._isHolding() and self.simulate_pos[2] > self.simulate_z_threshold

  def getGripperImg(self, gripper_state=None, gripper_rz=None, heightmap_size=None, obs_size_m=None):
    if obs_size_m is None:
      obs_size_m = self.obs_size_m
    if heightmap_size is None:
      heightmap_size = self.heightmap_size
    if gripper_state is None:
      gripper_state = self.robot.getGripperOpenRatio()
    if gripper_rz is None:
      gripper_rz = transformations.euler_from_quaternion(self.robot._getEndEffectorRotation())[2]
    im = np.zeros((heightmap_size, heightmap_size))
    gripper_half_size = 5 * self.workspace_size / obs_size_m
    gripper_half_size = round(gripper_half_size/128*heightmap_size)
    if self.robot_type in ['panda', 'ur5', 'ur5_robotiq']:
      gripper_max_open = 42 * self.workspace_size / obs_size_m
    elif self.robot_type == 'kuka':
      gripper_max_open = 45 * self.workspace_size / obs_size_m
    else:
      raise NotImplementedError
    d = int(gripper_max_open/128*heightmap_size * gripper_state)
    square_gripper = False
    anchor = heightmap_size//2
    if square_gripper:
      
      im[int(anchor - d // 2 - gripper_half_size):int(anchor - d // 2 + gripper_half_size), int(anchor - gripper_half_size):int(anchor + gripper_half_size)] = 1
      im[int(anchor + d // 2 - gripper_half_size):int(anchor + d // 2 + gripper_half_size), int(anchor - gripper_half_size):int(anchor + gripper_half_size)] = 1
    else:
      l = 1.2*gripper_half_size
      w = 0.9*gripper_half_size
      im[int(anchor-d//2-w):int(anchor-d//2+w), int(anchor-l):int(anchor+l)] = 1
      im[int(anchor+d//2-w):int(anchor+d//2+w), int(anchor-l):int(anchor+l)] = 1
    im = rotate(im, np.rad2deg(gripper_rz), reshape=False, order=0)
    return im

  def _getHeightmap(self, gripper_pos=None, gripper_rz=None):
    gripper_z_offset = 0.04 # panda
    if self.robot_type == 'kuka':
      gripper_z_offset = 0.12
      # gripper_z_offset = 0 # for visualization
    elif self.robot_type == 'ur5':
      gripper_z_offset = 0.06
    if gripper_pos is None:
      gripper_pos = self.robot._getEndEffectorPosition()
    if gripper_rz is None:
      gripper_rz = transformations.euler_from_quaternion(self.robot._getEndEffectorRotation())[2]
    if self.view_type == 'render_center':
      return self.renderer.getTopDownDepth(self.obs_size_m, self.heightmap_size, gripper_pos, 0)
    elif self.view_type == 'render_center_height':
      depth = self.renderer.getTopDownDepth(self.obs_size_m, self.heightmap_size, gripper_pos, 0)
      heightmap = gripper_pos[2] - depth
      return heightmap
    elif self.view_type == 'render_fix':
      return self.renderer.getTopDownHeightmap(self.heightmap_size)

    elif self.view_type == 'camera_center_xyzr':
      # xyz centered, alighed
      gripper_pos[2] += gripper_z_offset
      target_pos = [gripper_pos[0], gripper_pos[1], 0]
      T = transformations.euler_matrix(0, 0, gripper_rz)
      cam_up_vector = T.dot(np.array([-1, 0, 0, 1]))[:3]
      self.sensor.setCamMatrix(gripper_pos, cam_up_vector, target_pos)
      heightmap = self.sensor.getHeightmap(self.heightmap_size)
      depth = -heightmap + gripper_pos[2]
      return depth
    elif self.view_type in ['camera_center_xyr', 'camera_center_xyr_height']:
      # xy centered, aligned
      target_pos = [gripper_pos[0], gripper_pos[1], 0]
      T = transformations.euler_matrix(0, 0, gripper_rz)
      cam_up_vector = T.dot(np.array([-1, 0, 0, 1]))[:3]
      cam_pos = [gripper_pos[0], gripper_pos[1], 0.29]
      self.sensor.setCamMatrix(cam_pos, cam_up_vector, target_pos)
      heightmap = self.sensor.getHeightmap(self.heightmap_size)
      if self.view_type == 'camera_center_xyr':
        depth = -heightmap + gripper_pos[2]
      else:
        depth = heightmap
      return depth
    elif self.view_type in ['camera_center_xyz', 'camera_center_xyz_height']:
      # xyz centered, gripper will be visible
      gripper_pos[2] += gripper_z_offset
      target_pos = [gripper_pos[0], gripper_pos[1], 0]
      cam_up_vector = [-1, 0, 0]
      self.sensor.setCamMatrix(gripper_pos, cam_up_vector, target_pos)
      heightmap = self.sensor.getHeightmap(self.heightmap_size)
      if self.view_type == 'camera_center_xyz':
        depth = -heightmap + gripper_pos[2]
      else:
        depth = heightmap
      return depth
    elif self.view_type in ['pers_center_xyz']:
      # xyz centered, gripper will be visible
      gripper_pos[2] += gripper_z_offset
      target_pos = [gripper_pos[0], gripper_pos[1], 0]
      cam_up_vector = [-1, 0, 0]
      self.pers_sensor.setCamMatrix(gripper_pos, cam_up_vector, target_pos)
      heightmap = self.pers_sensor.getHeightmap(self.heightmap_size)
      depth = -heightmap + gripper_pos[2]
      return depth
    elif self.view_type in ['camera_center_xy', 'camera_center_xy_height']:
      # xy centered
      target_pos = [gripper_pos[0], gripper_pos[1], 0]
      cam_up_vector = [-1, 0, 0]
      cam_pos = [gripper_pos[0], gripper_pos[1], 0.29]
      self.sensor.setCamMatrix(cam_pos, cam_up_vector, target_pos)
      heightmap = self.sensor.getHeightmap(self.heightmap_size)
      if self.view_type == 'camera_center_xy':
        depth = -heightmap + gripper_pos[2]
      else:
        depth = heightmap
      return depth
    elif self.view_type in ['camera_center_z', 'camera_center_z_height']:
      gripper_pos[2] += gripper_z_offset
      cam_pos = [self.workspace[0].mean(), self.workspace[1].mean(), gripper_pos[2]]
      target_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 0]
      cam_up_vector = [-1, 0, 0]
      self.sensor.setCamMatrix(cam_pos, cam_up_vector, target_pos)
      heightmap = self.sensor.getHeightmap(self.heightmap_size)
      if self.view_type == 'camera_center_z':
        depth = -heightmap + gripper_pos[2]
      else:
        depth = heightmap
      return depth
    elif self.view_type in ['ws_center_fix', 'ws_center_fix_height']:
      gripper_pos[2] += gripper_z_offset
      cam_pos = [self.workspace[0].mean(), self.workspace[1].mean(), self.workspace[2].max()]
      target_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 0]
      cam_up_vector = [-1, 0, 0]
      height_bias = self.workspace[2].max() - gripper_pos[2]# make sure 1. not see gripper 2. a fixed view from fixed ws top
      self.sensor.setCamMatrix(cam_pos, cam_up_vector, target_pos)
      heightmap = self.sensor.getHeightmap(self.heightmap_size) 
      if self.view_type == 'ws_center_fix':
        depth = -heightmap + self.workspace[2].max()
      else:
        depth = heightmap
      return depth
    elif self.view_type in ['camera_fix', 'camera_fix_height']:
      heightmap = self.sensor.getHeightmap(self.heightmap_size)
      if self.view_type == 'camera_fix':
        depth = -heightmap + gripper_pos[2]
      else:
        depth = heightmap
      return depth
    elif self.view_type in ['camera_side', 'camera_side_rgbd', 'camera_side_height']:
      cam_pos = [1, self.workspace[1].mean(), 0.6]
      target_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 0]
      cam_up_vector = [-1, 0, 0]
      self.sensor = Sensor(cam_pos, cam_up_vector, target_pos, 0.7, 0.1, 3)
      self.sensor.fov = 40
      self.sensor.proj_matrix = pb.computeProjectionMatrixFOV(self.sensor.fov, 1, self.sensor.near, self.sensor.far)
      if self.view_type == 'camera_side':
        depth = self.sensor.getDepthImg(self.heightmap_size)
      elif self.view_type == 'camera_side_rgbd':
        rgb_img = self.sensor.getRGBImg(self.heightmap_size)
        depth_img = self.sensor.getDepthImg(self.heightmap_size).reshape(1, self.heightmap_size, self.heightmap_size)
        depth = np.concatenate([rgb_img, depth_img])
      else:
        depth = self.sensor.getHeightmap(self.heightmap_size)
      return depth
    elif self.view_type in ['camera_side_viola', 'camera_side_viola_rgbd', 'camera_side_viola_height']:
      cam_pos = [-0.7, self.workspace[1].mean()-1.2, 0.9]
      target_pos = [self.workspace[0].mean()-0.2, self.workspace[1].mean(), 0.3]
      cam_up_vector = [0, 0, 3]
      self.sensor = Sensor(cam_pos, cam_up_vector, target_pos, 0.7, 0.3, 3)
      self.sensor.fov = 35
      self.sensor.proj_matrix = pb.computeProjectionMatrixFOV(self.sensor.fov, 1, self.sensor.near, self.sensor.far)
      if self.view_type == 'camera_side_viola':
        depth = self.sensor.getDepthImg(self.heightmap_size)
      elif self.view_type == 'camera_side_viola_rgbd':
        rgb_img = self.sensor.getRGBImg(self.heightmap_size)
        depth_img = self.sensor.getDepthImg(self.heightmap_size).reshape(1, self.heightmap_size, self.heightmap_size)
        depth = np.concatenate([rgb_img, depth_img])
      else:
        depth = self.sensor.getHeightmap(self.heightmap_size)
      return depth
    elif self.view_type in ['camera_side_viola_custom_1', 'camera_side_viola_rgbd_custom_1', 'camera_side_viola_height_custom_1']:
      # 1
      cam_pos = [0, self.workspace[1].mean()-1.2, 0.9]
      target_pos = [self.workspace[0].mean()-0.2, self.workspace[1].mean(), 0.3]
      cam_up_vector = [0, 0, 3]
      self.sensor = Sensor(cam_pos, cam_up_vector, target_pos, 0.7, 0.3, 3)
      self.sensor.fov = 35
      self.sensor.proj_matrix = pb.computeProjectionMatrixFOV(self.sensor.fov, 1, self.sensor.near, self.sensor.far)
      if self.view_type == 'camera_side_viola_custom_1':
        depth = self.sensor.getDepthImg(self.heightmap_size)
      elif self.view_type == 'camera_side_viola_rgbd_custom_1':
        rgb_img = self.sensor.getRGBImg(self.heightmap_size)
        depth_img = self.sensor.getDepthImg(self.heightmap_size).reshape(1, self.heightmap_size, self.heightmap_size)
        depth = np.concatenate([rgb_img, depth_img])
      else:
        depth = self.sensor.getHeightmap(self.heightmap_size)
      return depth
    elif self.view_type in ['camera_side_viola_custom_2', 'camera_side_viola_rgbd_custom_2', 'camera_side_viola_height_custom_2']:
      # 2
      cam_pos = [1.2, self.workspace[1].mean(), 1]
      target_pos = [self.workspace[0].mean()-0.2, self.workspace[1].mean(), 0]
      cam_up_vector = [0, 0, 3]
      self.sensor = Sensor(cam_pos, cam_up_vector, target_pos, 0.7, 0.3, 3)
      self.sensor.fov = 35
      self.sensor.proj_matrix = pb.computeProjectionMatrixFOV(self.sensor.fov, 1, self.sensor.near, self.sensor.far)
      if self.view_type == 'camera_side_viola_custom_2':
        depth = self.sensor.getDepthImg(self.heightmap_size)
      elif self.view_type == 'camera_side_viola_rgbd_custom_2':
        rgb_img = self.sensor.getRGBImg(self.heightmap_size)
        depth_img = self.sensor.getDepthImg(self.heightmap_size).reshape(1, self.heightmap_size, self.heightmap_size)
        depth = np.concatenate([rgb_img, depth_img])
      else:
        depth = self.sensor.getHeightmap(self.heightmap_size)
      return depth
    else:
      raise NotImplementedError

  def _encodeAction(self, primitive, x, y, z, r):
    if hasattr(r, '__len__'):
      assert len(r) in [1, 2, 3]
      if len(r) == 1:
        rz = r[0]
        ry = 0
        rx = 0
      elif len(r) == 2:
        rz = r[0]
        ry = 0
        rx = r[1]
      else:
        rz = r[0]
        ry = r[1]
        rx = r[2]
    else:
      rz = r
      ry = 0
      rx = 0

    primitive_idx, x_idx, y_idx, z_idx, rot_idx = map(lambda a: self.action_sequence.find(a),
                                                      ['p', 'x', 'y', 'z', 'r'])
    action = np.zeros(len(self.action_sequence), dtype=float)
    if primitive_idx != -1:
      action[primitive_idx] = primitive
    if x_idx != -1:
      action[x_idx] = x
    if y_idx != -1:
      action[y_idx] = y
    if z_idx != -1:
      action[z_idx] = z
    if rot_idx != -1:
      if self.action_sequence.count('r') == 1:
        action[rot_idx] = rz
      elif self.action_sequence.count('r') == 2:
        action[rot_idx] = rz
        action[rot_idx+1] = rx
      elif self.action_sequence.count('r') == 3:
        action[rot_idx] = rz
        action[rot_idx+1] = ry
        action[rot_idx+2] = rx

    return action

  # def isSimValid(self):
  #   all_upright = np.all(list(map(lambda o: self._checkObjUpright(o, threshold=np.deg2rad(10)), self.objects)))
  #   return all_upright and super().isSimValid()

  def _checkStack(self, objects=None):
    # 2-step checking
    if super()._checkStack(objects):
      self.wait(100)
      return super()._checkStack(objects)
    return False

  def getPointCloud(self):
    self.renderer.getNewPointCloud()
    return self.renderer.points

  def getEndEffectorPose(self):
    # get 4Dof pose: x, y, z, theta
    gripper_width = self.robot.getGripperOpenRatio()
    pos = self.robot._getEndEffectorPosition()
    rot = transformations.euler_from_quaternion(self.robot._getEndEffectorRotation())
    transform = np.append(pos, rot[2])
    
    return np.append(gripper_width, transform) # [p, x, y, z, theta]
  
  def getCurrentTimeStep(self):
    # return self.current_episode_steps - 1
    current_time_steps = self.current_episode_steps - 1
    current_time_steps = current_time_steps.numpy()
    mask = np.zeros(self.time_horizon)
    quetient = current_time_steps // self.time_horizon
    mask_steps = current_time_steps % self.time_horizon
    time_steps = np.zeros(self.time_horizon)
    if quetient < 1:
        mask[self.time_horizon-mask_steps:self.time_horizon] = np.ones(mask_steps)
        time_steps[self.time_horizon-mask_steps:self.time_horizon] = np.arange(0, mask_steps, 1)
    else:
        mask = np.ones(self.time_horizon).astype(bool)
        time_steps = np.arange(0, self.time_horizon, 1)
        
    return mask.astype(bool), time_steps.astype(int)
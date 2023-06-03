import sys
sys.path.append('..')

import pybullet as pb
import numpy as np
import os

import bulletarm
from bulletarm.pybullet.objects.pybullet_object import PybulletObject
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.utils import transformations

class VoidGoal(PybulletObject):
  def __init__(self, pos, rot, scale):
    goal_visual = pb.createVisualShape(pb.GEOM_SPHERE, 
                                       radius=0.025, 
                                       rgbaColor=[0, 0, 1, 1])
    self.object_id = pb.createMultiBody(baseMass=0,
                                      baseVisualShapeIndex=goal_visual,
                                      basePosition=pos,
                                      baseOrientation=rot )
    
    super(VoidGoal, self).__init__(constants.VOID_GOAL, self.object_id)

    self.original_size = 0.025
    self.size = 0.01 * scale

  def getHeight(self):
    return self.size

  def getRotation(self):
    pos, rot = self.getPose()
    return rot
  
  # def getPosition(self):
  #   pos, _ = pb.getBasePositionAndOrientation(self.object_id)
  #   pos[-1] = 0
  #   return list(pos)

  def getPose(self):
    pos, rot = pb.getBasePositionAndOrientation(self.object_id)
    T = transformations.quaternion_matrix(rot)
    t = 0
    while T[2, 2] < 0.5 and t < 4:
      T = T.dot(transformations.euler_matrix(np.pi/2, 0, 0))
      t += 1

    t = 0
    while T[2, 2] < 0.5 and t < 4:
      T = T.dot(transformations.euler_matrix(0, np.pi/2, 0))
      t += 1

    rot = transformations.quaternion_from_matrix(T)
    return list(pos), list(rot)

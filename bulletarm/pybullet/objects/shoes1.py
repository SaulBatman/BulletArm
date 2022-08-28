import sys

sys.path.append('..')

import pybullet as pb
import numpy as np
import os

import bulletarm
from bulletarm.pybullet.objects.pybullet_object import PybulletObject
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.utils import transformations


class ShoeRight(PybulletObject):
  def __init__(self, pos, rot, scale):
    root_dir = os.path.dirname(bulletarm.__file__)
    sdf_filepath = os.path.join(root_dir, constants.OBJECTS_PATH, 'shoes/shoe_right.sdf')
    object_id = pb.loadSDF(sdf_filepath, globalScaling=scale)[0]
    pb.resetBasePositionAndOrientation(object_id, np.array(pos), np.array(rot))
    pb.changeDynamics(object_id, -1, mass=0.4, lateralFriction=0.6)

    super(ShoeRight, self).__init__(constants.SHOE_RIGHT, object_id)


class ShoeLeft(PybulletObject):
  def __init__(self, pos, rot, scale):
    root_dir = os.path.dirname(bulletarm.__file__)
    sdf_filepath = os.path.join(root_dir, constants.OBJECTS_PATH, 'shoes/shoe_left.sdf')
    object_id = pb.loadSDF(sdf_filepath, globalScaling=scale)[0]
    pb.resetBasePositionAndOrientation(object_id, np.array(pos), np.array(rot))
    pb.changeDynamics(object_id, -1, mass=0.4, lateralFriction=0.6)

    super(ShoeLeft, self).__init__(constants.SHOE_LEFT, object_id)

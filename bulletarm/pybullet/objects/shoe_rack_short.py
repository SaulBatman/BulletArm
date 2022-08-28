import sys
sys.path.append('..')

import pybullet as pb
import numpy as np
import os

import bulletarm
from bulletarm.pybullet.objects.pybullet_object import PybulletObject
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.utils import transformations

class ShoeRackShort(PybulletObject):
  def __init__(self, pos, rot, scale):
    root_dir = os.path.dirname(bulletarm.__file__)
    sdf_filepath = os.path.join(root_dir, constants.OBJECTS_PATH, 'shoe_rack/shoe_rack_short.sdf')
    object_id = pb.loadSDF(sdf_filepath, globalScaling=scale)[0]
    pos = (pos[0], pos[1], 0)
    pb.resetBasePositionAndOrientation(object_id, np.array(pos), np.array(rot))
    super(ShoeRackShort, self).__init__(constants.SHOE_RACK_SHORT, object_id)

  def getLeftPose(self):
    return pb.getLinkState(self.object_id, 0)

  def getRightPose(self):
    return pb.getLinkState(self.object_id, 1)
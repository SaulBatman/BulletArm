import unittest
import time
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from bulletarm import env_factory

class TestBulletBlockStacking(unittest.TestCase):
  env_config = {'num_objects': 2, 'seed': 1, 'robot':'kuka', 'seed': 1, 'view_scale': 1.5, 'workspace': np.array([[0.25, 0.65], [-0.2, 0.2], [0.01, 0.25]]), 'render': True}

  planner_config = {'random_orientation': True}

  def testPlanner(self):
    self.env_config['render'] = True
    env = env_factory.createEnvs(1, 'block_stacking', self.env_config, self.planner_config)
    env.reset()
    for i in range(5, -1, -1):
      action = env.getNextAction()
      (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
      self.assertEqual(env.getStepsLeft(), i)
    self.assertEqual(rewards, 1)
    self.assertEqual(dones, 1)
    env.close()

  def testPlanner2(self):
    self.env_config['render'] = False
    self.env_config['seed'] = 0
    num_processes = 3
    env = env_factory.createEnvs(num_processes,  'block_stacking', self.env_config, self.planner_config)
    # planner = CloseLoopBlockStackingPlanner(env, {})
    total = 0
    s = 0
    step_times = []
    (states, hand_obs, obs) = env.reset()
    pbar = tqdm(total=500)
    while total < 500:
      t0 = time.time()
      action = env.getNextAction()
      t_plan = time.time() - t0
      (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset='step_auto_reset')
      env.getObsTemporalBBox(dones)
      s += rewards.sum()
      total += dones.sum()
      t_action = time.time() - t0 - t_plan
      t = time.time() - t0
      step_times.append(t)

      pbar.set_description(
        '{}/{}, SR: {:.3f}, plan time: {:.2f}, action time: {:.2f}, avg step time: {:.2f}'
          .format(s, total, float(s) / total if total != 0 else 0, t_plan, t_action, np.mean(step_times))
      )
      pbar.update(dones.sum())
    env.close()

if __name__ == "__main__":
  TestBulletBlockStacking().testPlanner2()
import unittest
import time
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from bulletarm import env_factory

class TestBulletCloseLoopBlockStackingTR2(unittest.TestCase):
  env_config = {'robot':'kuka', 'time_horizon': 2, 'obs_type': 'state_tr2', 'seed': 1, 'view_scale': 1.5, 'workspace': np.array([[0.25, 0.65], [-0.2, 0.2], [0.01, 0.25]]), 'render': True}

  planner_config = {'random_orientation': True, 'dpos': 0.05, 'drot': np.pi / 4, 'max_teacher_length': 50}

  def testPlanner2(self):
    self.env_config['render'] = True
    self.env_config['seed'] = 0
    num_processes = 2
    env = env_factory.createEnvs(num_processes,  'close_loop_block_stacking_tr2', self.env_config, self.planner_config)
    total = 0
    s = 0
    step_times = []
    states, hand_obs, obs = env.reset()
    teacher_traj, mask, time_steps = env.getHighLevelTraj()
    pbar = tqdm(total=500)
    while total < 500:
      t0 = time.time()
      action = env.getNextAction()
      t_plan = time.time() - t0
      # print(env.getCurrentTimeStep())
      (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset='None')
      tem_states, global_obs, in_hand, ee_pos = env.getObsTemporal()
      # plt.imshow(obs_[0, 0])
      # plt.colorbar()
      # plt.show()
      # print(states_)
      s += rewards.sum()
      total += dones.sum()
      t_action = time.time() - t0 - t_plan
      t = time.time() - t0
      step_times.append(t)

      pbar.set_description(
        '{:.3f}/{}, SR: {:.3f}, plan time: {:.2f}, action time: {:.2f}, avg step time: {:.2f}'
          .format(s, total, float(s) / total if total != 0 else 0, t_plan, t_action, np.mean(step_times))
      )

      if dones.all():
        states, hand_obs, obs = env.reset()
        teacher_traj, mask, time_steps = env.getHighLevelTraj()

    env.close()

if __name__ == "__main__":
  TestBulletCloseLoopBlockStackingTR2().testPlanner2()
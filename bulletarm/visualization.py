import matplotlib.pyplot as plt
import matplotlib
import torch
import numpy as np
import torchvision

def visualizeObs(global_obs, in_hand, goal_bbox, all_bbox, horizon=10):
    bbox_size=10*2
    fig, axes = plt.subplots(2, horizon)
    for i in range(horizon):
        axes[0, i].imshow(global_obs[i][0])
        # axes[i].arrow(x=64, y=64, dx=sim_actions1_star[2]/view_obs_m*128, dy=sim_actions1_star[1]/view_obs_m*128, width=.5)
        # visualize bbox
        # axes[0, i].set_title(f"state:{transition.state[i]}, done:{transition.done}")
        for bbox in all_bbox[i]:
            axes[0, i].add_patch(matplotlib.patches.Rectangle((bbox[0],bbox[1]),bbox_size,bbox_size,
                        edgecolor='red',
                        facecolor='none',
                        lw=4))
        # visualize goal bbox
        goal_bbox = goal_bbox[i]
        axes[0, i].add_patch(matplotlib.patches.Rectangle((goal_bbox[0][0],goal_bbox[0][1]),bbox_size,bbox_size,
                        edgecolor='green',
                        facecolor='none',
                        lw=2))
        # # visualize goal bbox after roi_align
        # roi_align = torchvision.ops.RoIAlign(output_size=20, spatial_scale=1.0, sampling_ratio=-1, aligned=True)
        # a=roi_align(torch.from_numpy(global_obs[i][0]).unsqueeze(0).unsqueeze(0), [torch.from_numpy(goal_bbox).float()])
        # axes[2, i].imshow(a[0,0])
        # visualize in_hand
        axes[1, i].imshow(in_hand[i][-1])
    plt.show()

# def visualizeVIOLABuffer(transition: GoalExpertTransition, horizon=10):
#     bbox_size=10*2
#     fig, axes = plt.subplots(3, horizon)
#     for i in range(horizon):
#         axes[0, i].imshow(transition.obs[i][0])
#         axes[i].arrow(x=64, y=64, dx=sim_actions1_star[2]/view_obs_m*128, dy=sim_actions1_star[1]/view_obs_m*128, width=.5)
#         # visualize bbox
#         axes[0, i].set_title(f"state:{transition.state[i]}, done:{transition.done}")
#         for bbox in transition.all_bbox[i]:
#             axes[0, i].add_patch(matplotlib.patches.Rectangle((bbox[0],bbox[1]),bbox_size,bbox_size,
#                         edgecolor='red',
#                         facecolor='none',
#                         lw=4))
#         # visualize goal bbox
#         goal_bbox = transition.goal_bbox[i]
#         axes[0, i].add_patch(matplotlib.patches.Rectangle((goal_bbox[0][0],goal_bbox[0][1]),bbox_size,bbox_size,
#                         edgecolor='green',
#                         facecolor='none',
#                         lw=2))
#         # visualize goal bbox after roi_align
#         roi_align = torchvision.ops.RoIAlign(output_size=20, spatial_scale=1.0, sampling_ratio=-1, aligned=True)
#         a=roi_align(torch.from_numpy(transition.obs[i][0]).unsqueeze(0).unsqueeze(0), [torch.from_numpy(goal_bbox).float()])
#         axes[2, i].imshow(a[0,0])
#         # visualize in_hand
#         axes[1, i].imshow(transition.in_hand[i][-1])
#     plt.show()
#!/usr/bin/env python3

# Copyright 2022 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from email.errors import ObsoleteHeaderDefect
from time import time
import warnings
import os
import cv2
import sys
sys.path.append("..")

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from dm_env import specs

import PatchAIL.utils as utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader, make_expert_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import pickle
from tqdm import tqdm

import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

warnings.filterwarnings('ignore', category=DeprecationWarning)
torch.backends.cudnn.benchmark = True

def _make_dir(filename=None, folder_name=None):
    folder = folder_name
    if folder is None:
        folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)
        
def logits_to_pixels(rewards, pixel_to_output_dict, image_height=84, image_width=84):
    pixel_rewards = np.zeros((image_height, image_width))
    for h in range(image_height):
        for w in range(image_width):
            related_logits = pixel_to_output_dict["{}-{}".format(h, w)]
            sum_logits = []
            for logit in related_logits:
                sum_logits.append(rewards[logit[0], logit[1]])
            mean_logits = np.mean(sum_logits)
            pixel_rewards[h][w] = mean_logits
    
    return pixel_rewards

class Patch_aggregator(nn.Module):
    def __init__(self, patchD):
        super().__init__()
        self.patchD = patchD
        

    def forward(self, obs):
        #obs = obs / 255.0 - 0.5
        logits = self.patchD(obs)

        logits = logits.view(logits.shape[0],-1)
        expert_scalar = torch.mean(logits, axis=1)
        # return expert_scalar.view(-1,1)
                
        # normalized_expert_scalar = (expert_scalar + 10)/20
        probs = torch.cat([torch.sigmoid(expert_scalar.view(-1,1)), 1-torch.sigmoid(expert_scalar.view(-1,1))], axis=1)
        #torch.chunk(torch.cat([torch.unsqueeze(normalized_expert_scalar, 0), torch.unsqueeze(1-normalized_expert_scalar, 0)], axis=-2), chunks=normalized_expert_scalar.shape[0], dim=1)
        return probs

class Encoder_feature_mapper(nn.Module):
    def __init__(self, encoder, encoder_type):
        super().__init__()
        self.encoder = encoder
        self.encoder_type = encoder_type

    def forward(self, obs):
        print(self.encoder_type)
        feature_map = self.encoder.forward_convnet(obs)
        return feature_map


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec[cfg.obs_type].shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)

def _make_dir(filename=None, folder_name=None):
    folder = folder_name
    if folder is None:
        folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)
        
# def save_three_images(video_frames, filedir, obs_policy, task_name, extra_info=None, format="png"):
    
#     _make_dir(folder_name=filedir)
#     import matplotlib.pyplot as plt
#     from PIL import Image, ImageDraw, ImageFont
#     import scipy.misc
#     from scipy import ndimage
#     if extra_info:
#         extra_info = list(extra_info)

#     for i, frame in tqdm(enumerate(video_frames)):
#         imgs = [frame[0:3,...], frame[3:6,...], frame[6:9,...]]
#         for j, img in enumerate(imgs):
#             im = Image.fromarray(np.uint8(np.rollaxis(img, 0,3)))
#             # rotated_img = ndimage.rotate(np.uint8(np.rollaxis(img, 0,3)), 360)
#             # plt.imshow(rotated_img)
#             # plt.savefig("{}/img_{}_{}_plt_lower.{}".format(filedir, i, j, format))
#             d = ImageDraw.Draw(im)

#             # draw text, half opacity
#             if extra_info:
#                 d.text(
#                     (1, 10), "Max logits: {}".format(extra_info[i][1]), fill=(255, 255, 204)
#                 )
#             im.save("{}/{}_img_{}_{}_{}.{}".format(filedir, task_name, obs_policy, i, j, format))

def save_three_images(video_frames, filedir, obs_policy, task_name, extra_info=None, format="png"):
    
    _make_dir(folder_name=filedir)
    import matplotlib.pyplot as plt
    plt.tight_layout()
    fig = plt.figure()
    plt.tight_layout()
    from PIL import Image, ImageDraw, ImageFont
    import scipy.misc
    from scipy import ndimage
    import matplotlib.animation as animation
    if extra_info:
        extra_info = list(extra_info)

    original_videos = []
    for i, frame in tqdm(enumerate(video_frames)):
        if frame.shape[0] == 9: #dmc 3*3frames
            imgs = [frame[0:3,...], frame[3:6,...], frame[6:9,...]]
            for j, img in enumerate(imgs):
                
                im = Image.fromarray(np.uint8(np.rollaxis(img, 0,3)))
                # rotated_img = ndimage.rotate(np.uint8(np.rollaxis(img, 0,3)), 360)
                # plt.imshow(rotated_img)
                # plt.savefig("{}/img_{}_{}_plt_lower.{}".format(filedir, i, j, format))
                d = ImageDraw.Draw(im)

                # draw text, half opacity
                if extra_info:
                    d.text(
                        (1, 10), "Max logits: {}".format(extra_info[i][1]), fill=(255, 255, 204)
                    )
                im.save("{}/{}_img_{}_{}_{}.{}".format(filedir, task_name, obs_policy, i, j, format))

                if j == 2:
                    plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
                    plt.axis('off')
                    result0 = plt.imshow(np.uint8(np.rollaxis(img, 0,3)), origin='upper', animated=True)
                    original_videos.append([result0])
        elif frame.shape[0] == 4: #atari 4*1frame grayscale
            imgs = [frame[0:1,...], frame[1:2,...], frame[2:3,...], frame[3:4,...]]
            for j, img in enumerate(imgs):
                
                rgb_im = cv2.cvtColor(img[0], cv2.COLOR_GRAY2RGB)
                
                im = Image.fromarray(rgb_im)
                # rotated_img = ndimage.rotate(np.uint8(np.rollaxis(img, 0,3)), 360)
                # plt.imshow(rotated_img)
                # plt.savefig("{}/img_{}_{}_plt_lower.{}".format(filedir, i, j, format))
                d = ImageDraw.Draw(im)

                # draw text, half opacity
                if extra_info:
                    d.text(
                        (1, 10), "Max logits: {}".format(extra_info[i][1]), fill=(255, 255, 204)
                    )
                im.save("{}/{}_img_{}_{}_{}.{}".format(filedir, task_name, obs_policy, i, j, format))

                if j == 2:
                    plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
                    plt.axis('off')
                    result0 = plt.imshow(np.uint8(np.rollaxis(img, 0,3)), origin='upper', animated=True)
                    original_videos.append([result0])      
        else:
            raise NotImplementedError
    ani = animation.ArtistAnimation(fig, original_videos, interval=500, blit=False, repeat_delay=1000)
    ani.save(os.path.join(filedir, '{}_original_video_{}_{}frames.mp4'.format(task_name,obs_policy,len(video_frames))),fps=25, extra_args=['-vcodec', 'libx264']) # 


def logits_to_pixels(rewards, pixel_to_output_dict, image_height=84, image_width=84):
    pixel_rewards = np.zeros((image_height, image_width))
    for h in range(image_height):
        for w in range(image_width):
            related_logits = pixel_to_output_dict["{}-{}".format(h, w)]
            sum_logits = []
            for logit in related_logits:
                sum_logits.append(rewards[logit[0], logit[1]])
            mean_logits = np.mean(sum_logits)
            pixel_rewards[h][w] = mean_logits
    

    return pixel_rewards

def save_heatmap_on_images(rewards, video_frames, filedir, mapping_dict, obs_policy, task_name, vmin=-4, vmax=4, extra_info=None, format="png", image_height=100, image_width=100):
    _make_dir(folder_name=filedir)
    heatmaps = []

    import matplotlib.pyplot as plt
    plt.tight_layout()

    pixel_to_output_dict = mapping_dict

    import matplotlib.animation as animation
    fig = plt.figure()
    plt_0 = plt.subplot2grid((5,5), (0,0), colspan=5, rowspan=5)
    ax_0 = fig.add_subplot(plt_0)
    # from mpl_toolkits.axes_grid1 import make_axes_locatable
    # I like to position my colorbars this way, but you don't have to
    #div = make_axes_locatable(ax_0)
    #cax = div.append_axes('left', '5%', '5%')
    
    # plt_1 = plt.subplot2grid((5,5), (0,4), colspan=1, rowspan=5)
    # fig.add_subplot(plt_1)
    cb = None
    input_height = 84
    input_width = 84

    logit_max = np.max(rewards)
    logit_min = np.min(rewards)

    for i, frame in tqdm(enumerate(video_frames)):
        imgs = [frame[0:3,...], frame[3:6,...], frame[6:9,...]]
        for j, img in enumerate(imgs):
 
            result0 = plt_0.imshow(np.uint8(np.rollaxis(img, 0,3)), origin='upper', animated=True)
            pixel_rewards = logits_to_pixels(rewards[i], pixel_to_output_dict, image_height=input_height, image_width=input_width)
           
            normalized_pixel_rewards = (np.clip(pixel_rewards, vmin, vmax)+abs(vmin))/(vmax-vmin)
            result1 = plt_0.imshow(np.exp(normalized_pixel_rewards), origin='upper', alpha=0.25, cmap='jet', vmin=np.exp(0), vmax=np.exp(1), animated=True)
            #result1 = None
            if i == 0 and j == 0:
                cb = plt.colorbar(result1)

            plt_0.grid(None)
            #plt_1.grid(None)
            plt_0.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
            plt.grid(None)
            #heatmaps.append([result0, result1] + list(reward_bar))
            heatmaps.append([result0, result1])
            plt.savefig("{}/{}_img_heat_{}_{}_{}.{}".format(filedir, task_name, obs_policy, i, j, format))
            #plt.clf()


def save_gradcam_images(obs_list, next_obs_list, discriminator, device, obs_policy, reward_list, task_name, suite_name):
    obs_list_tensor = torch.tensor(obs_list).to(device)
    next_obs_list_tensor = torch.tensor(next_obs_list).to(device)
    
    input_for_D =  torch.cat([obs_list_tensor, next_obs_list_tensor], dim=1)
    gradcam_model = Patch_aggregator(discriminator)
    with torch.no_grad():
        with utils.eval_mode(discriminator):
            logits = discriminator(input_for_D)
            probs = gradcam_model(input_for_D)

    target_layers_for_gradcam = [gradcam_model.patchD.convnet[-3]]
    cam = GradCAM(model=gradcam_model, target_layers=target_layers_for_gradcam, use_cuda=True)
    # cam.batch_size = 1

    targets = [] # 0 == expert
    for i in range(input_for_D.shape[0]):
        targets.append(ClassifierOutputTarget(0))
    grayscale_cam = cam(input_tensor=input_for_D, targets=targets)

    
    obs_for_grad_cam_0 = []
    obs_for_grad_cam_1 = []
    obs_for_grad_cam_2 = []
    obs_for_grad_cam_3 = []
    
    for i, frame in tqdm(enumerate(np.array(obs_list)/255)):
        if suite_name == 'dmc':
            imgs = [frame[0:3,...], frame[3:6,...], frame[6:9,...]]

        elif suite_name == 'atari':
            imgs = [frame[0:1,...], frame[1:2,...], frame[2:3,...], frame[3:4,...]]
            
            obs_for_grad_cam_3.append(cv2.cvtColor((imgs[3][0]*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)/255)

        obs_for_grad_cam_0.append(cv2.cvtColor((imgs[0][0]*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)/255)
        obs_for_grad_cam_1.append(cv2.cvtColor((imgs[1][0]*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)/255)
        obs_for_grad_cam_2.append(cv2.cvtColor((imgs[2][0]*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)/255)

    obs_for_grad_cam_0 = np.array(obs_for_grad_cam_0)
    obs_for_grad_cam_1 = np.array(obs_for_grad_cam_1)
    obs_for_grad_cam_2 = np.array(obs_for_grad_cam_2)
    if suite_name == 'atari':
        obs_for_grad_cam_3 = np.array(obs_for_grad_cam_3)
    
    share_encoder_string = "PatchD"

    visualization_0 = [show_cam_on_image(img, grayscale, use_rgb=True, image_weight=0.00) for img, grayscale in zip(obs_for_grad_cam_0, grayscale_cam)]
    for i, grad_cam_img in enumerate(visualization_0):
        plt.imshow(grad_cam_img, origin='upper', animated=True)
        plt.text(x=1.5, y=6,  s='Expert Pred Prob: {:.3f}'.format(probs[i][0]), color='white', fontsize=18)
        plt.savefig('{}_{}_gradcam_{}_{}_{}_label-expert_truereward={}.pdf'.format(task_name, share_encoder_string, obs_policy, i, 'frame0', reward_list[i]), bbox_inches='tight')
        plt.clf()

    visualization_1 = [show_cam_on_image(img, grayscale, use_rgb=True, image_weight=0.00) for img, grayscale in zip(obs_for_grad_cam_1, grayscale_cam)]
    for i, grad_cam_img in enumerate(visualization_1):
        plt.imshow(grad_cam_img, origin='upper', animated=True)
        plt.text(x=1.5, y=6,  s='Expert Pred Prob: {:.3f}'.format(probs[i][0]), color='white', fontsize=18)
        plt.savefig('{}_{}_gradcam_{}_{}_{}_label-expert_truereward={}.pdf'.format(task_name, share_encoder_string, obs_policy, i, 'frame1', reward_list[i]), bbox_inches='tight')
        plt.clf()

    visualization_2 = [show_cam_on_image(img, grayscale, use_rgb=True, image_weight=0.00) for img, grayscale in zip(obs_for_grad_cam_2, grayscale_cam)]
    for i, grad_cam_img in enumerate(visualization_2):
        plt.imshow(grad_cam_img, origin='upper', animated=True)
        plt.text(x=1.5, y=6,  s='Expert Pred Prob: {:.3f}'.format(probs[i][0]), color='white', fontsize=18)
        plt.savefig('{}_{}_gradcam_{}_{}_{}_label-expert_truereward={}.pdf'.format(task_name, share_encoder_string, obs_policy, i, 'frame2', reward_list[i]), bbox_inches='tight')
        plt.clf()
    
    if suite_name == 'atari':
        visualization_3 = [show_cam_on_image(img, grayscale, use_rgb=True, image_weight=0.00) for img, grayscale in zip(obs_for_grad_cam_3, grayscale_cam)]
        for i, grad_cam_img in enumerate(visualization_3):
            plt.imshow(grad_cam_img, origin='upper', animated=True)
            plt.text(x=1.5, y=6,  s='Expert Pred Prob: {:.3f}'.format(probs[i][0]), color='white', fontsize=18)
            plt.savefig('{}_{}_gradcam_{}_{}_{}_label-expert_truereward={}.pdf'.format(task_name, share_encoder_string, obs_policy, i, 'frame3', reward_list[i]), bbox_inches='tight')
            plt.clf()

    targets = [] # 0 == expert
    for i in range(input_for_D.shape[0]):
        targets.append(ClassifierOutputTarget(1))
    grayscale_cam = cam(input_tensor=input_for_D, targets=targets)


    visualization_0 = [show_cam_on_image(img, grayscale, use_rgb=True, image_weight=0.00) for img, grayscale in zip(obs_for_grad_cam_0, grayscale_cam)]
    for i, grad_cam_img in enumerate(visualization_0):
        plt.imshow(grad_cam_img, origin='upper', animated=True)
        plt.text(x=1.5, y=6,  s='Non-Expert Pred Prob: {:.3f}'.format(probs[i][1]), color='white', fontsize=18)
        plt.savefig('{}_{}_gradcam_{}_{}_{}_label-noexpert_truereward={}.pdf'.format(task_name, share_encoder_string, obs_policy, i, 'frame0',reward_list[i]), bbox_inches='tight')
        plt.clf()

    visualization_1 = [show_cam_on_image(img, grayscale, use_rgb=True, image_weight=0.00) for img, grayscale in zip(obs_for_grad_cam_1, grayscale_cam)]
    for i, grad_cam_img in enumerate(visualization_1):
        plt.imshow(grad_cam_img, origin='upper', animated=True)
        plt.text(x=1.5, y=6,  s='Non-Expert Pred Prob: {:.3f}'.format(probs[i][1]), color='white', fontsize=18)
        plt.savefig('{}_{}_gradcam_{}_{}_{}_label-noexpert_truereward={}.pdf'.format(task_name, share_encoder_string, obs_policy, i, 'frame1',reward_list[i]), bbox_inches='tight')
        plt.clf()

    visualization_2 = [show_cam_on_image(img, grayscale, use_rgb=True, image_weight=0.00) for img, grayscale in zip(obs_for_grad_cam_2, grayscale_cam)]
    for i, grad_cam_img in enumerate(visualization_2):
        plt.imshow(grad_cam_img, origin='upper', animated=True)
        plt.text(x=1.5, y=6,  s='Non-Expert Pred Prob: {:.3f}'.format(probs[i][1]), color='white', fontsize=18)
        plt.savefig('{}_{}_gradcam_{}_{}_{}_label-noexpert_truereward={}.pdf'.format(task_name, share_encoder_string, obs_policy, i, 'frame2',reward_list[i]), bbox_inches='tight')
        plt.clf()

    if suite_name == 'atari':
        visualization_3 = [show_cam_on_image(img, grayscale, use_rgb=True, image_weight=0.00) for img, grayscale in zip(obs_for_grad_cam_3, grayscale_cam)]
        for i, grad_cam_img in enumerate(visualization_3):
            plt.imshow(grad_cam_img, origin='upper', animated=True)
            plt.text(x=1.5, y=6,  s='Non-Expert Pred Prob: {:.3f}'.format(probs[i][1]), color='white', fontsize=18)
            plt.savefig('{}_{}_gradcam_{}_{}_{}_label-noexpert_truereward={}.pdf'.format(task_name, share_encoder_string, obs_policy, i, 'frame3',reward_list[i]), bbox_inches='tight')
            plt.clf()


def save_feature_map_images(obs_list, next_obs_list, patchD, encoder, encoder_type, device, obs_policy, reward_list, task_name, reward_logits_list, pixel_to_output_dict, vmin, vmax, filedir, vmin_coef, vmax_coef, visualization_type):

    _make_dir(folder_name=filedir)  
    obs_list_tensor = torch.tensor(obs_list).to(device)
    next_obs_list_tensor = torch.tensor(next_obs_list).to(device)
    feature_map_model = Encoder_feature_mapper(encoder, encoder_type)

    if encoder_type == 'PatchAIL' and patchD:
        input_for_patchD = torch.cat([obs_list_tensor, next_obs_list_tensor], dim=1)
        feature_map = feature_map_model(input_for_patchD)
    else:
        feature_map = feature_map_model(obs_list_tensor)
    
    spatial_feature_map = []
    for i in range(len(feature_map)): # for each layer
        
        feature_map_layer_i = feature_map[i]
        feature_rows = feature_map_layer_i.shape[2]
        feature_cols = feature_map_layer_i.shape[3]
        
        mean_pooling_abs_value = torch.mean(abs(feature_map_layer_i), dim=1)
        mean_pooling_abs_value_flatten = mean_pooling_abs_value.view(mean_pooling_abs_value.shape[0], -1)
        # 2D softmax
        softmax_mean_pooling_flatten =  torch.nn.functional.softmax(mean_pooling_abs_value_flatten, dim=1)
        softmax_mean_pooling = softmax_mean_pooling_flatten.view(mean_pooling_abs_value_flatten.shape[0], feature_rows, feature_cols)
        spatial_feature_map.append(softmax_mean_pooling)


    obs_for_feature_map_0 = []
    obs_for_feature_map_1 = []
    obs_for_feature_map_2 = []
    for i, frame in tqdm(enumerate(np.array(obs_list)/255)):
        imgs = [frame[0:3,...], frame[3:6,...], frame[6:9,...]]
        obs_for_feature_map_0.append(np.rollaxis(imgs[0], 0,3))
        obs_for_feature_map_1.append(np.rollaxis(imgs[1], 0,3))
        obs_for_feature_map_2.append(np.rollaxis(imgs[2], 0,3))
    
    obs_for_feature_map_0 = np.array(obs_for_feature_map_0)
    obs_for_feature_map_1 = np.array(obs_for_feature_map_1)
    obs_for_feature_map_2 = np.array(obs_for_feature_map_2)
    
    input_height = 84
    input_width = 84
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    plt.rcParams['savefig.bbox'] = 'tight' 
    plt.tight_layout()
    fig = plt.figure()
    plt.tight_layout()
    plt_0 = plt.subplot2grid((5,5), (0,0), colspan=5, rowspan=5)
    ax_0 = fig.add_subplot(plt_0)

    animation_list = []

    convlayer_idx = 4
    for n, img in tqdm(enumerate(obs_for_feature_map_2)):
        pixel_rewards = logits_to_pixels(reward_logits_list[n], pixel_to_output_dict, image_height=input_height, image_width=input_width)
        resized_feature_map = cv2.resize(spatial_feature_map[convlayer_idx][n].cpu().detach().numpy(), (img.shape[0], img.shape[1]), interpolation = cv2.INTER_AREA) 
        
        weighted_patch_rewards = pixel_rewards * resized_feature_map
        
        if visualization_type == 'feature_map':
            to_draw = resized_feature_map
            result_2 = plt_0.imshow(to_draw, origin='upper', alpha=1, cmap='jet', vmin=np.min(resized_feature_map), vmax=np.max(resized_feature_map), animated=True)
        elif visualization_type == 'patch_rewards':
            to_draw = pixel_rewards
            result_2 = plt_0.imshow(to_draw, origin='upper', alpha=1, cmap='jet', vmin=vmin, vmax=vmax, animated=True)
        elif visualization_type == 'feature_weighted_patch_rewards':
            to_draw = weighted_patch_rewards
            result_2 = plt_0.imshow(to_draw, origin='upper', alpha=1, cmap='jet', vmin=vmin*np.mean(resized_feature_map)*vmin_coef, vmax=vmax*np.mean(resized_feature_map)*vmax_coef, animated=True)

        plt_0.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
        plt.axis('off')
        if visualization_type == 'patch_rewards':
            plt.savefig('{}_{}_{}_{}_{}_{}_truereward={}.pdf'.format(task_name, encoder_type, visualization_type, obs_policy, n, 'frame2', reward_list[n]), bbox_inches='tight')
        else:
            plt.savefig('{}_{}_{}_convlayer-{}_{}_{}_{}_truereward={}.pdf'.format(task_name, encoder_type, visualization_type, convlayer_idx, obs_policy, n, 'frame2', reward_list[n]), bbox_inches='tight')
        
        animation_list.append([result_2])

    ani = animation.ArtistAnimation(fig, animation_list, interval=500, blit=False, repeat_delay=1000)
    ani.save(os.path.join(filedir, '{}_{}_featuremap_patchreward-weighted_convlayer-{}_video_{}_{}frames.mp4'.format(task_name, encoder_type, 4, obs_policy,len(animation_list))),fps=25, extra_args=['-vcodec', 'libx264']) # 

class WorkspaceVis:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()
        if cfg.suite.name == 'atari':
            cfg.agent.n_actions = self.train_env.action_spec().num_values

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(), cfg.agent)
            
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        with open(self.cfg.expert_dataset, 'rb') as f:
            if self.cfg.obs_type == 'pixels':
                self.expert_demo, _, _, self.expert_reward = pickle.load(f)
            elif self.cfg.obs_type == 'features':
                _, self.expert_demo, _, self.expert_reward = pickle.load(f)
        self.expert_demo = self.expert_demo[:self.cfg.num_demos]
        self.expert_demo_reward = self.expert_reward
        self.expert_reward = np.mean([np.mean(_) for _ in self.expert_reward[:self.cfg.num_demos]])
        
    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        self.eval_env = hydra.utils.call(self.cfg.suite.task_make_fn)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.suite.action_repeat

    def load_snapshot(self, snapshot):
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        
        agent_payload = {}
        for k, v in payload.items():
            if k not in self.__dict__:
                agent_payload[k] = v
        
        self.agent.load_snapshot(agent_payload)

    def eval_heatmap(self):
        vminmax_dict = {'finger_spin': [-2 ,6, 2, 2], 'cartpole_swingup': [-2, 2, 2, 2], 'cheetah_run': [-2, 4, 0.75, 2.25], 'hopper_stand': [-4, 4, 1.5, 1.5], 'quadruped_run': [-4, 4, 2, 4], 'walker_stand': [-2 ,3, 2, 1.5], 'walker_walk': [-2, 3, 2.25, 1.25],
                        'pong': [-0.5, 0.5, 1, 1], 'freeway': [-0.5, 0.5, 1, 1], 'krull':[-0.5, 0.5, 1, 1], 'boxing': [-0.5, 0.5, 1, 1]}
        random_expert_index_dict = {'finger_spin': [36 ,35], 'cartpole_swingup': [59, 103], 'cheetah_run': [26, 60], 'hopper_stand': [172, 27], 'quadruped_run': [43, 102], 'walker_stand': [3, 136], 'walker_walk': [5, 27]}
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.suite.num_eval_episodes)
        #mapping_file = None
        mapping_file_exist = True
        if not mapping_file_exist:
            # ====================
            def get_filters_mapping_list(height, width, kernel_size, stride, padding, dilation=[1,1]):
                assert dilation[0]==1 and dilation[1] == 1
                output_height = int((height + 2 * padding[0] - dilation[0]*(kernel_size[0]-1)-1)/stride[0] + 1)
                output_width = int((width + 2 * padding[1] - dilation[1]*(kernel_size[1]-1)-1)/stride[1] + 1)
                filters_mapping_list = []
                for h in range(output_height):
                    for w in range(output_width):
                        start_pixel_height = 0 - padding[0] + h * stride[0]
                        start_pixel_width = 0 - padding[1] + w * stride[1]
                        pixel_list = []
                        for k_h in range(kernel_size[0]):
                            for k_w in range(kernel_size[1]):
                                pixel_list.append([start_pixel_height+k_h, start_pixel_width+k_w])
                        output_location = [h, w]
                        filters_mapping_list.append([pixel_list, output_location])
                return filters_mapping_list
            def find_related_output(intersted_locations, all_mappings):
                related_output = []
                for location in intersted_locations:
                    for mapping in all_mappings:
                        if location in mapping[0]:
                            related_output.append(mapping[1])
                return related_output
            def get_pixel_to_output_dict(conv2d_mapping_list, image_height=84, image_width=84):
                print("Getting pixel_to_output_dict")
                result = {}  # [ "i,j" , [ [a,b], .... [e,f] ] ]
                for h in tqdm(range(image_height)):
                    for w in range(image_width):
                        interested_locations = [[h,w]]
                        for mapping_layer in conv2d_mapping_list:
                            related_output = find_related_output(interested_locations, mapping_layer)
                            interested_locations = related_output
                        result["{}-{}".format(h,w)] = related_output
                        
                return result

            # ====================
            image_height = 84
            image_width = 84
            input_height = 84
            input_width = 84

            conv2d_mapping_list = []
            
            for layer in self.agent.discriminator.convnet:
                if 'kernel_size' in dir(layer) and 'stride' in dir(layer) and 'padding' in dir(layer):
                    conv2d_mapping_list.append(get_filters_mapping_list(height=input_height, width=input_width, kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding, dilation=layer.dilation))
                    input_height=int((input_height + 2 * layer.padding[0] - layer.dilation[0]*(layer.kernel_size[0]-1)-1)/layer.stride[0] + 1)
                    input_width = int((input_width + 2 * layer.padding[1] - layer.dilation[1]*(layer.kernel_size[1]-1)-1)/layer.stride[1] + 1)
                    print([input_height, input_width])
            
            pixel_to_output_dict = get_pixel_to_output_dict(conv2d_mapping_list, image_height, image_width)
            
            import pickle
            file_name='./default_pixel_to_output_dict_{}.pkl'.format(self.agent.suite_name)
            f = open(file_name,'wb')
            pickle.dump(pixel_to_output_dict,f)
            f.close()

            exit(0)
        else:
            import pickle
            print("Loading mapping file (pixel to logits)")
            
            with open(Path(self.cfg.mapping_file), 'rb') as f:
                pixel_to_output_dict = pickle.load(f)
            
        if self.cfg.suite.name == 'openaigym' or self.cfg.suite.name == 'metaworld':
            paths = []
        
        obs_list = []
        next_obs_list = []
        reward_list = []
        reward_logits_list = []

        while eval_until_episode(episode):
            if self.cfg.suite.name == 'metaworld':
                path = []

            time_step = self.eval_env.reset()

            if self.video_recorder:
                self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            #while not time_step.last():
            
            # =========================== START random running in environment ===========================
            # ==--====--====--====--====--====--====--====--====--====--====--====--====--====--====--====--====--====--==
            if self.cfg.snapshot_epoch == '_scalarD_share':
                encoder_type = 'share'
                #save_feature_map_images(obs_list, next_obs_list, False, self.agent.encoder, encoder_type, self.device, 'random', reward_list, self.cfg.task_name, reward_logits_list, pixel_to_output_dict)
            elif self.cfg.snapshot_epoch == '_scalarD_noshare':
                encoder_type = 'shareNO'
                #save_feature_map_images(obs_list, next_obs_list, False, self.agent.disc_encoder, encoder_type, self.device, 'random', reward_list, self.cfg.task_name, reward_logits_list, pixel_to_output_dict)
            else:
                encoder_type = 'PatchAIL'
                #save_feature_map_images(obs_list, next_obs_list, True, self.agent.discriminator, encoder_type, self.device, 'random', reward_list, self.cfg.task_name, reward_logits_list, pixel_to_output_dict)
           
            
            for i in tqdm(range(100)):
                with torch.no_grad(), utils.eval_mode(self.agent):
                    obs = time_step.observation[self.cfg.obs_type]
                    action = self.agent.act(time_step.observation[self.cfg.obs_type],
                                            self.global_step,
                                            eval_mode=True)
                    
                    if self.agent.suite_name == 'atari':
                        random_action = np.random.randint(self.cfg.agent.n_actions)
                    elif self.agent.suite_name == 'dmc':
                        random_action = np.random.random(action.shape) * 2 -1
                    else:
                        raise NotImplementedError

                    
                time_step = self.eval_env.step(random_action)
                next_obs = time_step.observation[self.cfg.obs_type]

                obs_list.append(obs)
                next_obs_list.append(next_obs)
                reward_list.append(time_step.reward)

                if encoder_type == 'PatchAIL':
                    
                    reward_logits = self.agent.dac_rewarder(np.array([obs]), np.array([action]), np.array([next_obs]), return_logits=True)
                    reward_logits_list.append(reward_logits.squeeze().detach().cpu().numpy())

                # ========== end ==========
            
                if self.cfg.suite.name == 'metaworld':
                    path.append(time_step.observation['goal_achieved'])
                if self.video_recorder:
                    self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1
            # =========================== END running in environment ===========================

            if self.cfg.visualization_type == 'gradcam':
                save_gradcam_images(obs_list, next_obs_list, self.agent.discriminator, self.device, 'random', reward_list, self.cfg.task_name, self.agent.suite_name)
            elif self.cfg.visualization_type == 'patch_rewards' or self.cfg.visualization_type == 'feature_map' or self.cfg.visualization_type == 'feature_weighted_patch_rewards':
                if self.cfg.snapshot_epoch == '_scalarD_share':
                    encoder_type = 'share'
                    save_feature_map_images(obs_list, next_obs_list, False, self.agent.encoder, encoder_type, self.device, 'random', reward_list, self.cfg.task_name, reward_logits_list, pixel_to_output_dict, vmin=vminmax_dict[self.cfg.task_name][0], vmax=vminmax_dict[self.cfg.task_name][1], filedir='.', vmin_coef=vminmax_dict[self.cfg.task_name][2], vmax_coef=vminmax_dict[self.cfg.task_name][3], visualization_type = self.cfg.visualization_type)
                elif self.cfg.snapshot_epoch == '_scalarD_noshare':
                    encoder_type = 'shareNO'
                    save_feature_map_images(obs_list, next_obs_list, False, self.agent.disc_encoder, encoder_type, self.device, 'random', reward_list, self.cfg.task_name, reward_logits_list, pixel_to_output_dict, vmin=vminmax_dict[self.cfg.task_name][0], vmax=vminmax_dict[self.cfg.task_name][1], filedir='.', vmin_coef=vminmax_dict[self.cfg.task_name][2], vmax_coef=vminmax_dict[self.cfg.task_name][3], visualization_type = self.cfg.visualization_type)
                else:
                    encoder_type = 'PatchAIL'
                    save_feature_map_images(obs_list, next_obs_list, True, self.agent.discriminator, encoder_type, self.device, 'random', reward_list, self.cfg.task_name, reward_logits_list, pixel_to_output_dict, vmin=vminmax_dict[self.cfg.task_name][0], vmax=vminmax_dict[self.cfg.task_name][1], filedir='.', vmin_coef=vminmax_dict[self.cfg.task_name][2], vmax_coef=vminmax_dict[self.cfg.task_name][3], visualization_type = self.cfg.visualization_type)
            else:
                raise NotImplementedError

            save_three_images(obs_list, '.', 'random', self.cfg.task_name)
            
            # save heatmaps on images
            save_heatmap_on_images(reward_logits_list, obs_list, '.', pixel_to_output_dict, 'random', self.cfg.task_name, vmin=vminmax_dict[self.cfg.task_name][0], vmax=vminmax_dict[self.cfg.task_name][1])

            obs_list = []
            next_obs_list = []
            reward_list = []
            reward_logits_list = []

            # =========================== START expert demos ===========================
            print("strat expert demos")
            for i in tqdm(range(100)):
                if self.agent.suite_name == 'atari':
                    if self.cfg.task_name == 'pong':
                        traj_idx = 0
                    elif self.cfg.task_name == 'boxing':
                        traj_idx = 3
                    elif self.cfg.task_name == 'krull':
                        traj_idx = 7
                    elif self.cfg.task_name == 'freeway':
                        traj_idx = 1
                    else:
                        raise NotImplementedError
                
                else:
                    traj_idx = 0
                
                obs = self.expert_demo[traj_idx][i]
                next_obs = self.expert_demo[traj_idx][i+1]
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(obs,
                                            self.global_step,
                                            eval_mode=True)

                obs_list.append(obs)
                next_obs_list.append(next_obs)
                reward_list.append(self.expert_demo_reward[traj_idx][i])

                if encoder_type == 'PatchAIL':
                    reward_logits = self.agent.dac_rewarder(np.array([obs]), np.array([action]), np.array([next_obs]), return_logits=True)
                    reward_logits_list.append(reward_logits.squeeze().detach().cpu().numpy())

                # ========== end ==========
            
                if self.cfg.suite.name == 'metaworld':
                    path.append(time_step.observation['goal_achieved'])
                if self.video_recorder:
                    self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            # =========================== END expert demos ===========================

            if self.cfg.visualization_type == 'gradcam':
                save_gradcam_images(obs_list, next_obs_list, self.agent.discriminator, self.device, 'expert', reward_list, self.cfg.task_name, self.agent.suite_name)
            elif self.cfg.visualization_type == 'patch_rewards' or self.cfg.visualization_type == 'feature_map' or self.cfg.visualization_type == 'feature_weighted_patch_rewards':
                if self.cfg.snapshot_epoch == '_scalarD_share':
                    encoder_type = 'share'
                    save_feature_map_images(obs_list, next_obs_list, False, self.agent.encoder, encoder_type, self.device, 'random', reward_list, self.cfg.task_name, reward_logits_list, pixel_to_output_dict, vmin=vminmax_dict[self.cfg.task_name][0], vmax=vminmax_dict[self.cfg.task_name][1], filedir='.', vmin_coef=vminmax_dict[self.cfg.task_name][2], vmax_coef=vminmax_dict[self.cfg.task_name][3], visualization_type = self.cfg.visualization_type)
                elif self.cfg.snapshot_epoch == '_scalarD_noshare':
                    encoder_type = 'shareNO'
                    save_feature_map_images(obs_list, next_obs_list, False, self.agent.disc_encoder, encoder_type, self.device, 'random', reward_list, self.cfg.task_name, reward_logits_list, pixel_to_output_dict, vmin=vminmax_dict[self.cfg.task_name][0], vmax=vminmax_dict[self.cfg.task_name][1], filedir='.', vmin_coef=vminmax_dict[self.cfg.task_name][2], vmax_coef=vminmax_dict[self.cfg.task_name][3], visualization_type = self.cfg.visualization_type)
                else:
                    encoder_type = 'PatchAIL'
                    save_feature_map_images(obs_list, next_obs_list, True, self.agent.discriminator, encoder_type, self.device, 'random', reward_list, self.cfg.task_name, reward_logits_list, pixel_to_output_dict, vmin=vminmax_dict[self.cfg.task_name][0], vmax=vminmax_dict[self.cfg.task_name][1], filedir='.', vmin_coef=vminmax_dict[self.cfg.task_name][2], vmax_coef=vminmax_dict[self.cfg.task_name][3], visualization_type = self.cfg.visualization_type)
            else:
                raise NotImplementedError

            save_three_images(obs_list, '.', 'expert', self.cfg.task_name)
            
            # save heatmaps on images
            save_heatmap_on_images(reward_logits_list, obs_list, '.', pixel_to_output_dict, 'expert', self.cfg.task_name, vmin=vminmax_dict[self.cfg.task_name][0], vmax=vminmax_dict[self.cfg.task_name][1])
        
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.suite.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
            if repr(self.agent) != 'drqv2':
                log('expert_reward', self.expert_reward)
            if self.cfg.suite.name == 'openaigym' or self.cfg.suite.name == 'metaworld':
                log("success_percentage", np.mean(paths))


@hydra.main(config_path='../cfgs', config_name='config_normal')
def main(cfg):
    from patch_visualize import WorkspaceVis as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    
    # Load weights
    if cfg.load_snapshot:
        snapshot = Path(cfg.snapshot_weight)
        print(snapshot)
        
        if snapshot.exists():
            print(f'resuming snapshot: {snapshot}')
            workspace.load_snapshot(snapshot)
    
    workspace.eval_heatmap()


if __name__ == '__main__':
    main()


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
import numpy as np
import torch
from torch import nn, optim, distributions
from torch.nn import functional as F

import utils
from agent.encoder import Encoder, AtariEncoder
from agent.modules import Actor, Critic, DiscreteActor, DiscreteCritic

class BCAgent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, stddev_schedule, stddev_clip, use_tb, augment, suite_name, obs_type, n_actions=None):
        self.device = device
        self.lr = lr
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.use_tb = use_tb
        self.augment = augment
        self.use_encoder = True if (suite_name!="adroit" and obs_type=='pixels') else False
        self.suite_name = suite_name

        # models
        if self.use_encoder:
            if self.suite_name == "atari":
                self.encoder = AtariEncoder(obs_shape).to(device)
            else:
                self.encoder = Encoder(obs_shape).to(device)
            repr_dim = self.encoder.repr_dim
        else:
            repr_dim = obs_shape[0] 

        if suite_name == "atari":
            self.actor = DiscreteActor(repr_dim, n_actions, feature_dim,
                           hidden_dim).to(device)

        else:
            self.actor = Actor(repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)

        # optimizers
        if self.use_encoder:
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # data augmentation
        self.aug = utils.RandomShiftsAug(pad=4)

        self.aug = lambda x:x

        self.train()

    def __repr__(self):
        return "bc"
    
    def train(self, training=True):
        self.training = training
        if self.use_encoder:
            self.encoder.train(training)
        self.actor.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)

        obs = self.encoder(obs.unsqueeze(0)) if self.use_encoder else obs.unsqueeze(0)
        
        stddev = utils.schedule(self.stddev_schedule, step)

        dist = self.actor(obs, std=stddev)

        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
        return action.cpu().numpy()[0]

    def update(self, replay_iter, expert_replay_iter, step, bc_regularize=False, *args, **kwargs):
        metrics = dict()

        batch = next(expert_replay_iter)
        obs, action, next_obs = utils.to_torch(batch, self.device)
        action = action.float()
        
        # augment
        if self.use_encoder and self.augment:
            obs = self.aug(obs.float())
            # encode
            obs = self.encoder(obs)
        else:
            obs = obs.float()
    
        if self.suite_name == "atari":
            prob = self.actor(obs, return_action=True)
            # cross entropy loss
            actor_loss = nn.CrossEntropyLoss()(prob, action.long())
        else:
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(obs, stddev)
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)

            actor_loss = -log_prob.mean()
        
        if self.use_encoder:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.use_encoder:
            self.encoder_opt.step()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            if self.suite_name != "atari":
                metrics['actor_logprob'] = log_prob.mean().item()
                metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def save_snapshot(self):
        keys_to_save = ['actor']
        if self.use_encoder:
            keys_to_save += ['encoder']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        return payload

    def load_snapshot(self, payload):
        for k, v in payload.items():
            self.__dict__[k] = v

        # Update optimizers
        if self.use_encoder:
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

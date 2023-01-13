
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
import torch.nn as nn
import utils


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35
        self.unflatten = False

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h_flat = h.view(h.shape[0], -1)
        if 'unflatten' in dir(self) and self.unflatten:
            return h_flat, h
        return h_flat

class AtariEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.unflatten = False
        
        # CNN modeled off of Mnih et al.
        self.repr_dim = 3136
        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 8, stride=4),
                                     nn.ReLU(), nn.Conv2d(32, 64, 4, stride=2),
                                     nn.ReLU(), nn.Conv2d(64, 64, 3, stride=1),
                                     nn.ReLU())

        # self.repr_dim = 225792
        # self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=1, padding='same'),
        #                              nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1, padding='same'),
        #                              nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1, padding='same'),
        #                              nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1, padding='same'),
        #                              nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h_flat = h.view(h.shape[0], -1)
        if 'unflatten' in dir(self) and self.unflatten:
            return h_flat, h
        return h_flat

class EasyEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 225792

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=1, padding='same'),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1, padding='same'),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1, padding='same'),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1, padding='same'),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        return h
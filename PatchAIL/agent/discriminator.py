
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
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

try:
    from vit_pytorch import SimpleViT
    from vit_pytorch.simple_vit import posemb_sincos_2d
    from einops import rearrange, repeat
except:
    print("cannot import vit_pytorch")

class Discriminator(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(in_dim, hid_dim), nn.ReLU(),
                                   nn.Linear(hid_dim, hid_dim), nn.ReLU(),
                                   nn.Linear(hid_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, x):
        output = self.trunk(x)
        return output

class WeightedFeatureDiscriminator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        repr_dim = 32
        self.weight_head = nn.Sequential(nn.Conv2d(in_dim, 64, 4, stride=1, padding=1),
                    nn.LeakyReLU(0.2, True), nn.Conv2d(64, 1, 4, stride=1, padding=1))

        self.feature_head = nn.Sequential(nn.Conv2d(in_dim, 64, 4, stride=1, padding=1),
                    nn.LeakyReLU(0.2, True), nn.Conv2d(64, repr_dim, 4, stride=1, padding=1))

        self.trunk = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Linear(repr_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, x):
        weight = self.weight_head(x)
        weight = nn.Softmax(dim=-1)(weight.view(weight.shape[0],weight.shape[1],-1))
        feature = self.feature_head(x)
        feature = feature.view(feature.shape[0],feature.shape[1], -1)
        res = weight * feature # (B, repr_dim, H, W)
        res = res.sum(dim=1) # (B, repr_dim)
        res = self.trunk(res) # (B, 1)
        return res

class AtariPatchDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator for Atari games"""
    def __init__(self, in_dim, final_iid=False):
        super().__init__()

        # self.repr_dim = 10 * 10 * 1

        # sequence = [nn.Conv2d(in_dim, 32, 8, stride=4, padding=1),
        #             nn.LeakyReLU(0.2, True), nn.Conv2d(32, 64, 4, stride=2, padding=1),
        #             nn.LeakyReLU(0.2, True), nn.Conv2d(64, 128, 3, stride=1, padding=1),
        #             nn.LeakyReLU(0.2, True), nn.Conv2d(128, 1, 3, stride=1, padding=1)
        #             ]

        sequence = [nn.Conv2d(in_dim, 32, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, True), nn.Conv2d(32, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, True), nn.Conv2d(64, 128, 4, stride=1, padding=1),
                    nn.LeakyReLU(0.2, True), nn.Conv2d(128, 1, 4, stride=1, padding=1)
                    ]

        if final_iid:
            sequence += [nn.LeakyReLU(0.2, True), nn.Conv2d(1, 1, 1, stride=1, padding=0)]

        self.convnet = nn.Sequential(*sequence)
        
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        # h = h.view(h.shape[0], -1)
        return h

class PatchDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""
    """Ref to https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/f13aab8148bd5f15b9eb47b690496df8dadbab0c/models/networks.py#L538"""
    def __init__(self, in_dim, final_iid=False):
        super().__init__()

        self.repr_dim = 32 * 35 * 35

        sequence = [nn.Conv2d(in_dim, 32, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, True), nn.Conv2d(32, 64, 4, stride=1, padding=1),
                    nn.LeakyReLU(0.2, True), nn.Conv2d(64, 128, 4, stride=1, padding=1),
                    nn.LeakyReLU(0.2, True), nn.Conv2d(128, 1, 4, stride=1, padding=1)]

        if final_iid:
            sequence += [nn.LeakyReLU(0.2, True), nn.Conv2d(1, 1, 1, stride=1, padding=0)]

        self.convnet = nn.Sequential(*sequence)
        
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        # h = h.view(h.shape[0], -1)
        return h

class IIDPatchDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator on pixels"""
    def __init__(self, in_dim, final_iid=False):
        super().__init__()

        self.repr_dim = 32 * 35 * 35

        sequence = [nn.Conv2d(in_dim, 32, 1, stride=1, padding=0),
                    nn.LeakyReLU(0.2, True), nn.Conv2d(32, 64, 1, stride=1, padding=0),
                    nn.LeakyReLU(0.2, True), nn.Conv2d(64, 128, 1, stride=1, padding=0),
                    nn.LeakyReLU(0.2, True), nn.Conv2d(128, 1, 1, stride=1, padding=0)]

        if final_iid:
            sequence += [nn.LeakyReLU(0.2, True), nn.Conv2d(1, 1, 1, stride=1, padding=0)]

        self.convnet = nn.Sequential(*sequence)
        
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        # h = h.view(h.shape[0], -1)
        return h

class SmallPatchDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator with Smaller Kernel Size -> Lead to less receptive field and more patches"""
    def __init__(self, in_dim, final_iid=False):
        super().__init__()

        self.repr_dim = 32 * 35 * 35

        sequence = [nn.Conv2d(in_dim, 32, 3, stride=2, padding=1),
                    nn.LeakyReLU(0.2, True), nn.Conv2d(32, 64, 3, stride=1, padding=1),
                    nn.LeakyReLU(0.2, True), nn.Conv2d(64, 128, 3, stride=1, padding=1),
                    nn.LeakyReLU(0.2, True), nn.Conv2d(128, 1, 3, stride=1, padding=1)]

        if final_iid:
            sequence += [nn.LeakyReLU(0.2, True), nn.Conv2d(1, 1, 1, stride=1, padding=0)]

        self.convnet = nn.Sequential(*sequence)
        
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        # h = h.view(h.shape[0], -1)
        return h

class BigPatchDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator with Smaller Kernel Size -> Lead to more receptive field and less patches"""
    def __init__(self, in_dim, final_iid=False):
        super().__init__()

        self.repr_dim = 32 * 35 * 35

        sequence = [nn.Conv2d(in_dim, 32, 5, stride=2, padding=1),
                    nn.LeakyReLU(0.2, True), nn.Conv2d(32, 64, 5, stride=1, padding=1),
                    nn.LeakyReLU(0.2, True), nn.Conv2d(64, 128, 5, stride=1, padding=1),
                    nn.LeakyReLU(0.2, True), nn.Conv2d(128, 1, 5, stride=1, padding=1)]

        if final_iid:
            sequence += [nn.LeakyReLU(0.2, True), nn.Conv2d(1, 1, 1, stride=1, padding=0)]

        self.convnet = nn.Sequential(*sequence)
        
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        # h = h.view(h.shape[0], -1)
        return h

class VitDiscriminator(nn.Module):
    """Defines a ViT discriminator"""
    def __init__(self, in_dim):
        super().__init__()

        self.repr_dim = 32 * 35 * 35

        self.vit = SimpleViT(
            image_size = 84,
            channels = in_dim,
            patch_size = 14,
            num_classes = 1,
            dim = 192,
            depth = 12,
            heads = 3,
            mlp_dim = 192 * 4,
        )
        
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5

        *_, h, w, dtype = *obs.shape, obs.dtype

        x = self.vit.to_patch_embedding(obs)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.vit.transformer(x)

        x = self.vit.to_latent(x)
        return self.vit.linear_head(x).squeeze()


class DiscTrunk(nn.Module):
    def __init__(self, repr_dim, feature_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.apply(utils.weight_init)

    def forward(self, obs):
        h = self.trunk(obs)

        return h
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

import dmc2gym
import os 
import time
from PIL import Image

# os.environ["LD_LIBRARY_PATH"] = "~/.mujoco/mjpro210/bin"
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
# os.environ["MUJOCO_GL"] = "osmesa"

env = dmc2gym.make(domain_name='finger', task_name='spin', from_pixels=True, visualize_reward=False)

t1 = time.time()
done = False
obs = env.reset()
i=0
while not done:
 
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
  # env.render(mode='rgb_array')
  frame = env.render(mode='rgb_array')
  im = Image.fromarray(frame)
  im.save("imgs/img_{}.{}".format(i, 'png'))

  i+=1
  
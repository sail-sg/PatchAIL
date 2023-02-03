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

import os
import gym
import numpy as np
import pickle
from pathlib import Path
import tensorflow_datasets as tfds
from collections import deque

games = ['Asterix'] # , 'Breakout', 'Space-Invaders', 'Seaquest', 'Pong']

num_frames= 4

def _make_dir(filename=None, folder_name=None):
    folder = folder_name
    if folder is None:
        folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

def save_video(video_frames, filename, fps=60, video_format="mp4"):
    assert fps == int(fps), fps
    import skvideo.io

    _make_dir(filename)

    skvideo.io.vwrite(
        filename,
        video_frames,
        inputdict={
            "-r": str(int(fps)),
        },
        outputdict={
            "-f": video_format,
            "-pix_fmt": "yuv420p",  # '-pix_fmt=yuv420p' needed for osx https://github.com/scikit-video/scikit-video/issues/74
        },
    )

def uniformly_subsampled_atari_data(
    dataset_name: str,
    data_percent: int,
    data_dir: str,
):
    ds_builder = tfds.builder(dataset_name)
    data_splits = []
    total_num_episode = 0
    for split, info in list(ds_builder.info.splits.items()):
        # Convert `data_percent` to number of episodes to allow
        # for fractional percentages.
        print(info.num_examples)
        num_episodes = int((data_percent / 100) * info.num_examples)
        total_num_episode += num_episodes
        if num_episodes == 0:
            raise ValueError(f"{data_percent}% leads to 0 episodes in {split}!")
        # Sample first `data_percent` episodes from each of the data split.
        data_splits.append(f"{split}[:{num_episodes}]")
    # Interleave episodes across different splits/checkpoints.
    # Set `shuffle_files=True` to shuffle episodes across files within splits.
    # print(data_splits)
    # print(len(data_splits))
    data_splits = data_splits[47:]
    # print(data_splits)
    read_config = tfds.ReadConfig(
        interleave_cycle_length=len(data_splits),
        shuffle_reshuffle_each_iteration=False,
        enable_ordering_guard=False,
    )

    return tfds.load(
        dataset_name,
        data_dir=data_dir,
        split="+".join(data_splits),
        read_config=read_config,
        shuffle_files=False,
    )

for game in games:
    print("=======================================")
    print("Processing game: ", game)
    
    dataset_name = f"rlu_atari_checkpoints_ordered/{''.join(game.split('-'))}_run_1"

    dataset = uniformly_subsampled_atari_data(dataset_name, 100, '/home/liumh/app/atari')
    
    print(dataset)
    # print(dataset.as_numpy_iterator())
    print(len(dataset))

    observations_list = list()
    actions_list = list()
    rewards_list = list()
    terminal_list = list()

    return_list = list()

    # Sort trajs by reward
    
    # for idx, traj in enumerate(dataset):
    #     # print(idx)
    #     ep_rewards_list = list()
    #     for t, timestep in enumerate(list(traj['steps'].as_numpy_iterator())):
    #         ep_rewards_list.append(timestep['reward'])
    #     return_list.append(np.sum(ep_rewards_list))
    # max_idx = np.argsort(-np.array(return_list))[:50]

    # print("max_idx: ", max_idx)
    # return_list.sort(reverse=True)
    # print("max_reward: ", return_list[:50])

    for idx, traj in enumerate(dataset):
        # if idx not in max_idx:
        #     continue
        if idx >= 100:
            break
        print(idx, len(traj['steps']))
        ep_observations_list = list()
        ep_actions_list = list()
        ep_rewards_list = list()
        ep_terminal_list = list()
        video_list = list()
        frames = deque([], maxlen=num_frames)
        for t, timestep in enumerate(list(traj['steps'].as_numpy_iterator())):
            if t == 0:
                for _ in range(num_frames):
                    frames.append([timestep['observation'].squeeze()])
            else:
                frames.append([timestep['observation'].squeeze()])
            video_list.append(timestep['observation'].squeeze())
            ep_observations_list.append(np.concatenate(list(frames), axis=0))
            ep_actions_list.append(timestep['action'])
            ep_rewards_list.append(timestep['reward'])
            ep_terminal_list.append(timestep['is_terminal'])

        observations_list.append(np.array(ep_observations_list))
        actions_list.append(np.array(ep_actions_list))
        rewards_list.append(np.array(ep_rewards_list))
        terminal_list.append(np.array(ep_terminal_list))

        # save_video(video_list, f'videos/{game}_{idx}.mp4')

        # print(np.array(ep_observations_list).shape)

    # # Make np arrays
    observations_list = np.array(observations_list)
    terminal_list = np.array(terminal_list)
    actions_list = np.array(actions_list)
    rewards_list = np.array(rewards_list)

    print(np.mean([np.sum(_) for _ in rewards_list]), np.std([np.sum(_) for _ in rewards_list]))

    game = [x.lower() for x in game.split("-")]
    game = "-".join(game)

    # Save demo in pickle file
    save_dir = Path("expert_demos/atari/{}".format(game))
    save_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = save_dir / 'expert_demos.pkl'
    payload = [
            observations_list, terminal_list, actions_list, rewards_list
        ]
    
    print("saving data ...")
    with open(str(snapshot_path), 'wb') as f:
        pickle.dump(payload, f)
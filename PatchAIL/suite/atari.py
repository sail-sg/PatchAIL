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

from collections import deque, OrderedDict
from os import truncate
from typing import Any, NamedTuple

import gym
import dm_env
import tree
import numpy as np
from dm_env import specs
from typing import Any, Callable, Dict, Optional, Tuple, Union
from gym.spaces.box import Box
from dm_control.suite.wrappers import pixels
import cv2

from .dmc import ActionRepeatWrapper, ExtendedTimeStepWrapper
from dm_env import specs

class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key # Not used

        pixels_spec = wrapped_obs_spec = env.observation_spec()

        pixels_shape = wrapped_obs_spec.shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = OrderedDict()

        self._obs_spec['pixels'] = specs.BoundedArray(shape=np.concatenate(
                                                    [[num_frames], pixels_spec.shape[:2]],
                                                    axis=0),
                                                    dtype=pixels_spec.dtype,
                                                    minimum=0,
                                                    maximum=255,
                                                    name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = OrderedDict()
        obs['pixels'] = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        try:
            pixels = time_step.observation[self._pixels_key]
        except:
            pixels = time_step.observation
        # remove batch dim
        if len(pixels.shape) == 3:
            pixels = pixels[0]
        return pixels.copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append([pixels])
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append([pixels])
        return self._transform_observation(time_step)

    def action_spec(self):
        return self._env.action_spec()

    def observation_spec(self):
        return self._obs_spec

    def __getattr__(self, name):
        return getattr(self._env, name)


class _AtariDopamineWrapper(dm_env.Environment):
    """Wrapper for Atari Dopamine environmnet."""

    def __init__(self, env: gym.Env, max_episode_steps: int = 108000):
        self._env = env
        self._max_episode_steps = max_episode_steps
        self._episode_steps = 0
        self._reset_next_episode = True
        self._reset_next_step = True

    def reset(self) -> dm_env.TimeStep:
        self._episode_steps = 0
        self._reset_next_step = False
        observation = self._env.reset()
        return dm_env.restart(observation.squeeze(-1))

    def step(self, action) -> dm_env.TimeStep:
        if self._reset_next_step:
            return self.reset()
        if not isinstance(action, int):
            action = action.item()
        observation, reward, terminal, _ = self._env.step(action)
        observation = observation.squeeze(-1)
        discount = 1 - float(terminal)
        self._episode_steps += 1
        if terminal:
            self._reset_next_episode = True
            return dm_env.termination(reward, observation)
        if self._episode_steps == self._max_episode_steps:
            self._reset_next_episode = True
            return dm_env.truncation(reward, observation, discount)
        return dm_env.transition(reward, observation, discount)

    def observation_spec(self) -> specs.Array:
        space = self._env.observation_space
        return specs.Array(space.shape[:-1], space.dtype)

    def action_spec(self) -> specs.DiscreteArray:
        return specs.DiscreteArray(self._env.action_space.n, name='action')

    def render(self) -> Any:
        """Render the environment.
        Returns:
            Any: Rendered result.
        """
        return self._env.render()

class AtariPreprocessing(object):
    """
    Copy from https://github.com/google/dopamine/blob/master/dopamine/discrete_domains/atari_lib.py#L333
    A class implementing image preprocessing for Atari 2600 agents.
    Specifically, this provides the following subset from the JAIR paper
    (Bellemare et al., 2013) and Nature DQN paper (Mnih et al., 2015):
        * Frame skipping (defaults to 4).
        * Terminal signal when a life is lost (off by default).
        * Grayscale and max-pooling of the last two frames.
        * Downsample the screen to a square image (defaults to 84x84).
    More generally, this class follows the preprocessing guidelines set down in
    Machado et al. (2018), "Revisiting the Arcade Learning Environment:
    Evaluation Protocols and Open Problems for General Agents".
    """

    def __init__(self, environment, frame_skip=4, terminal_on_life_loss=False,
                screen_size=84):
        """Constructor for an Atari 2600 preprocessor.
        Args:
        environment: Gym environment whose observations are preprocessed.
        frame_skip: int, the frequency at which the agent experiences the game.
        terminal_on_life_loss: bool, If True, the step() method returns
            is_terminal=True whenever a life is lost. See Mnih et al. 2015.
        screen_size: int, size of a resized Atari 2600 frame.
        Raises:
        ValueError: if frame_skip or screen_size are not strictly positive.
        """
        if frame_skip <= 0:
            raise ValueError('Frame skip should be strictly positive, got {}'.
                        format(frame_skip))
        if screen_size <= 0:
            raise ValueError('Target screen size should be strictly positive, got {}'.
                        format(screen_size))

        self.environment = environment
        self.terminal_on_life_loss = terminal_on_life_loss
        self.frame_skip = frame_skip
        self.screen_size = screen_size

        obs_dims = self.environment.observation_space
        # Stores temporary observations used for pooling over two successive
        # frames.
        self.screen_buffer = [
            np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8),
            np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8)
        ]

        self.game_over = False
        self.lives = 0  # Will need to be set by reset().

    @property
    def observation_space(self):
        # Return the observation space adjusted to match the shape of the processed
        # observations.
        return Box(low=0, high=255, shape=(self.screen_size, self.screen_size, 1),
                dtype=np.uint8)

    @property
    def action_space(self):
        return self.environment.action_space

    @property
    def reward_range(self):
        return self.environment.reward_range

    @property
    def metadata(self):
        return self.environment.metadata

    def close(self):
        return self.environment.close()

    def reset(self):
        """Resets the environment.
        Returns:
        observation: numpy array, the initial observation emitted by the
            environment.
        """
        self.environment.reset()
        self.lives = self.environment.ale.lives()
        self._fetch_grayscale_observation(self.screen_buffer[0])
        self.screen_buffer[1].fill(0)
        return self._pool_and_resize()

    def render(self):
        """Renders the current screen, before preprocessing.
        This calls the Gym API's render() method.
        Returns:
        if mode='rgb_array': numpy array, the most recent screen.
        if mode='human': bool, whether the rendering was successful.
        """
        return self.environment.render()

    def step(self, action):
        """Applies the given action in the environment.
        Remarks:
        * If a terminal state (from life loss or episode end) is reached, this may
            execute fewer than self.frame_skip steps in the environment.
        * Furthermore, in this case the returned observation may not contain valid
            image data and should be ignored.
        Args:
        action: The action to be executed.
        Returns:
        observation: numpy array, the observation following the action.
        reward: float, the reward following the action.
        is_terminal: bool, whether the environment has reached a terminal state.
            This is true when a life is lost and terminal_on_life_loss, or when the
            episode is over.
        info: Gym API's info data structure.
        """
        accumulated_reward = 0.

        for time_step in range(self.frame_skip):
            # We bypass the Gym observation altogether and directly fetch the
            # grayscale image from the ALE. This is a little faster.
            _, reward, game_over, truncated, info = self.environment.step(action)
            accumulated_reward += reward

            if self.terminal_on_life_loss:
                new_lives = self.environment.ale.lives()
                is_terminal = game_over or new_lives < self.lives
                self.lives = new_lives
            else:
                is_terminal = game_over

            # We max-pool over the last two frames, in grayscale.
            if time_step >= self.frame_skip - 2:
                t = time_step - (self.frame_skip - 2)
                self._fetch_grayscale_observation(self.screen_buffer[t])

            if is_terminal:
                break

        # Pool the last two observations.
        observation = self._pool_and_resize()

        self.game_over = game_over
        return observation, accumulated_reward, is_terminal, info

    def _fetch_grayscale_observation(self, output):
        """Returns the current observation in grayscale.
        The returned observation is stored in 'output'.
        Args:
        output: numpy array, screen buffer to hold the returned observation.
        Returns:
        observation: numpy array, the current observation in grayscale.
        """
        self.environment.ale.getScreenGrayscale(output)
        return output

    def _pool_and_resize(self):
        """Transforms two frames into a Nature DQN observation.
        For efficiency, the transformation is done in-place in self.screen_buffer.
        Returns:
        transformed_screen: numpy array, pooled, resized screen.
        """
        # Pool if there are enough screens to do so.
        if self.frame_skip > 1:
            np.maximum(self.screen_buffer[0], self.screen_buffer[1],
                        out=self.screen_buffer[0])

        transformed_image = cv2.resize(self.screen_buffer[0],
                                    (self.screen_size, self.screen_size),
                                    interpolation=cv2.INTER_AREA)
        int_image = np.asarray(transformed_image, dtype=np.uint8)
        return np.expand_dims(int_image, axis=2)

def make(name, frame_stack, action_repeat, seed, render_mode="rgb_array"):
    pixels_key = 'pixels'

    assert name is not None
    game_version = 'v0'
    tmp = name.split('-')
    name = ''
    for subname in tmp:
        name += subname.capitalize()
    full_game_name = '{}NoFrameskip-{}'.format(name, game_version)
    env = gym.make(full_game_name, render_mode=render_mode)
    env.seed(seed=seed)
    # Strip out the TimeLimit wrapper from Gym, which caps us at 100k frames. We
    # handle this time limit internally instead, which lets us cap at 108k frames
    # (30 minutes). The TimeLimit wrapper also plays poorly with saving and
    # restoring states.
    env = env.env
    env = AtariPreprocessing(env, frame_skip=action_repeat)   
    env = _AtariDopamineWrapper(env, max_episode_steps=20_000)
    
    # add wrappers
    # env = ActionRepeatWrapper(env, action_repeat)

    # stack several frames
    env = FrameStackWrapper(env, frame_stack, pixels_key)
    env = ExtendedTimeStepWrapper(env)
    return env
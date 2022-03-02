from abc import abstractmethod
from typing import TypeVar, Generic, Tuple
from typing import Optional

import gym
from gym import error, spaces

from gym.utils import closer, seeding
# from gym.logger import deprecation

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

import yaml
import rtamt
import sys

class STLGym(gym.core.Env):
    """The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.
    The main API methods that users of this class need to know are:
        step
        reset
        render
        close
        seed
    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
    The methods are accessed publicly as "step", "reset", etc...
    
    Wraps the environment to allow a modular transformation.
    This class is the base class for all wrappers. The subclass could override
    some methods to change the behavior of the original environment without touching the
    original code.

    """
    def __init__(self, config_path: str, env=None):
        """
        TODO: description
        """

        # Read the config YAML file
        with open(config_path, "r") as stream:
            try:
                config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        # print(config_dict)

        # Make the environment if it is not already provided
        if env is not None:
            self.env = env
        else:
            self.env = gym.make(config_dict['env_name'])

        self._action_space = None
        self._observation_space = None
        self._reward_range = None
        self._metadata = None

        # Initialize variables for analyzing STL specifications
        self.stl_spec = rtamt.STLDiscreteTimeSpecification()
        self.data = dict()
        self.data['time'] = []
        self.step_num = 0

        # Sort through specified constants that will be used in the specifications
        if 'constants' in config_dict.keys():
            constants = config_dict['constants']
            for i in constants:
                self.stl_spec.declare_const(i['name'], i['type'], i['value'])

        # Sort through specified variables that will be tracked
        self.stl_variables = config_dict['variables']
        for i in self.stl_variables:
            self.stl_spec.declare_var(i['name'], i['type'])
            self.data[i['name']] = []
            if 'i/o' in i.keys():
                self.stl_spec.set_var_io_type(i['name'], i['i/o'])

        # Collect specifications
        self.specifications = config_dict['specifications']
        spec_str = "out = "
        for i in self.specifications:
            self.stl_spec.declare_var(i['name'], 'float')
            self.stl_spec.add_sub_spec(i['spec'])
            spec_str += i['name'] + ' and '
            if 'weight' not in i.keys():
                i['weight'] = 1.0
        spec_str = spec_str[:-5]
        self.stl_spec.declare_var('out', 'float')
        self.stl_spec.spec = spec_str

        # Parse the specification
        try:
            self.stl_spec.parse()
        except rtamt.STLParseException as err:
            print('STL Parse Exception: {}'.format(err))
            sys.exit()
        

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(f"attempted to get missing private attribute {name}")
        return getattr(self.env, name)

    @property
    def spec(self):
        return self.env.spec

    @classmethod
    def class_name(cls):
        return cls.__name__

    @property
    def action_space(self):
        if self._action_space is None:
            return self.env.action_space
        return self._action_space

    @action_space.setter
    def action_space(self, space):
        self._action_space = space

    @property
    def observation_space(self):
        if self._observation_space is None:
            return self.env.observation_space
        return self._observation_space

    @observation_space.setter
    def observation_space(self, space):
        self._observation_space = space

    @property
    def reward_range(self):
        if self._reward_range is None:
            return self.env.reward_range
        return self._reward_range

    @reward_range.setter
    def reward_range(self, value):
        self._reward_range = value

    @property
    def metadata(self):
        if self._metadata is None:
            return self.env.metadata
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        # TODO: modify this function
        o, r, done, info = self.env.step(action)

        # Record and increment the time
        self.data['time'].append(self.step_num)
        self.step_num += 1

        # Add variables to their lists
        for i in self.stl_variables:
            if i['location'] == 'obs':
                self.data[i['name']].append(o[i['identifier']])
            elif i['location'] == 'info':
                self.data[i['name']].append(info[i['identifier']])
            elif i['location'] == 'state':
                self.data[i['name']].append(self.__getattr__(i['identifier']))
            else:
                # make an error for this
                print('ERROR ERROR')
        
        # Calculate the reward
        reward = self.compute_reward(done)

        return o, reward, done, info

    def reset(self, **kwargs):
        """
        Resets the environment to an initial state and returns an initial
        observation.
        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.
        Returns:
            observation (object): the initial observation.
        """
        # Reset the STL variable data
        self.step_num = 0
        for key in self.data.keys():
            self.data[key] = []
        return self.env.reset(**kwargs)

    def render(self, mode="human", **kwargs):
        return self.env.render(mode, **kwargs)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def compute_reward(self, done: bool) -> float:
        """TODO: write-up information
        """
        reward = 0
        
        if done:
            # TODO: fix this line so it works
            # foo = [[key, self.data[key]] for key in self.data.keys()]
            # print(f'FFFFF: {foo}')
            rob = self.stl_spec.evaluate(self.data)
            for i in self.specifications:
                # print(self.stl_spec.get_value(i['name']))
                reward += float(i['weight']) * self.stl_spec.get_value(i['name'])[-1]
            print(f'robustness: {str(rob[-1])}, reward: {reward}')
        return reward

    def __str__(self):
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        return str(self)

    @property
    def unwrapped(self):
        return self.env.unwrapped

if __name__ == "__main__":
    import numpy as np
    from environments import *
    
    config_path = './examples/pendulum.yaml'
    env = STLGym(config_path)
    num_evals = 100
    max_ep_len = 200
    render = False

    ep_returns = []
    ep_lengths = []

    for ep in range(num_evals):
        env.reset()
        ep_return = 0
        ep_len = 0
        for i in range(max_ep_len):
            if render:
                env.render()
                # time.sleep(1e-3)
            th, thdot = env.state
            u = np.array([((-32.0 / np.pi) * th)])  # + ((-1.0 / np.pi) * thdot)])
            _, r, done, info = env.step(u)
            ep_return += r
            ep_len += 1
            if done and i < (max_ep_len - 1):
                print(f"Failed: {env.state[0]} > {np.pi / 3.}; step: {i}")
                break
        ep_returns.append(ep_return)
        ep_lengths.append(ep_len)
    
    # Compute the averages and print them
    ep_rets = np.array(ep_returns)
    ep_lens = np.array(ep_lengths)
    print(f'Avg Return: {np.mean(ep_rets)} +- {np.std(ep_rets)}, Avg Length: {np.mean(ep_lens)}')
    
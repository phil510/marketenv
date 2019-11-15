import abc
import numpy as np
import warnings
import os

from .inventory import Inventory
from .spaces import TradeSpace, MarketSpace
from ..common.utils import (get_current_prices,
                            get_positions, 
                            get_cash_balance,
                            get_portfolio_value)
                   
class Environment(abc.ABC):
    def __init__(self):
        self.action_space = None
        self.observation_space = None
        self.reward_range = (-np.inf, np.inf)
        self.env_spec = {}
    
    @abc.abstractmethod
    def step(self, action):
        pass
    
    @abc.abstractmethod
    def reset(self):
        pass
    
    @abc.abstractmethod
    def close(self):
        pass
    
    @abc.abstractmethod
    def seed(self, seed = None):
        pass
        
    def render(self):
        raise NotImplementedError

    def __str__(self):
        return '{} Instance'.format(type(self).__name__)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        
    @property
    def unwrapped(self):
        return self
        
class MarketEnvironment(Environment):
    def __init__(self,
                 simulator,
                 n_tradable,
                 n_non_tradable = 0,
                 commission = 10,
                 beginning_cash = 10000):
        super().__init__()
        
        assert (n_tradable > 0), 'TODO'
        assert (n_non_tradable >= 0), 'TODO'
        self._n = n_tradable
        self._m = n_non_tradable
        
        self.action_space = TradeSpace(self._n)
        self.observation_space = MarketSpace(self._n, self._m)
        
        assert (hasattr(simulator, 'simulate')), 'TODO'
        self._simulator = simulator
        
        self._commission = commission
        self._inventory = Inventory(self._n, beginning_cash)
        
        self._time_step = None
        self._obs = None
        self._terminal = False
        
        self.seed()
        
        self.env_spec['n_tradable'] = self._n
        self.env_spec['n_non_tradable'] = self._m
        self.env_spec['commission'] = self._commission
        self.env_spec['beginning_cash'] = beginning_cash
    
    @property
    def observation(self):
        assert (self._obs is not None), 'TODO'
        return self._obs
    
    @property
    def positions(self):
        assert (self.observation is not None), 'TODO'
        return get_positions(self._obs, self._n)
    
    @property
    def cash_balance(self):
        assert (self.observation is not None), 'TODO'
        return get_cash_balance(self._obs)
    
    @property    
    def current_prices(self):
        assert (self.observation is not None), 'TODO'
        return get_current_prices(self._obs, self._n)
    
    @property
    def portfolio_value(self):
        assert (self.observation is not None), 'TODO'
        return get_portfolio_value(self._obs, self._n)
        
    @property
    def terminal(self):
        return self._terminal
    
    def seed(self, seed = None):
        self._rng = np.random.RandomState(seed)
        
        self._simulator.seed(seed = seed)
        self.action_space.seed(seed = seed)
        self.observation_space.seed(seed = seed)
    
    def reset(self):
        self._time_step = 0
        self._inventory.reset()
        self._terminal = False
        
        self._simulation, self._simulation_info = self._simulator.simulate()
        self._episode_len  = self._simulation.shape[0] - 1
        self._obs = self._create_obs()
        
        return np.array(self._obs)
     
    def close(self):
        self._time_step = None
        self._current_pos = None
    
    def step(self, action):
        assert (self._time_step is not None), 'TODO'
        assert (action in self.action_space), 'TODO'
        
        if self.terminal:
            warnings.warn('You are calling step() after the environment has'
                          + 'reached a terminal state; You should always call'
                          + 'reset() after reaching a terminal state')
            return np.array(self._obs), 0.0, self.terminal, {}
            
        current_value = self.portfolio_value
        
        self._inventory.update(action, self.current_prices, self._commission)
        self._time_step += 1
        self._obs = self._create_obs()
        
        new_value = self.portfolio_value
        reward = new_value - current_value
        
        if self._time_step >= self._episode_len:
            self._terminal = True
        
        return np.array(self._obs), reward, self._terminal, {}
        
    def _create_obs(self):
        obs = np.concatenate([np.array([self._inventory.cash_balance]),
                              self._inventory.positions,
                              self._simulation[self._time_step, :].squeeze()])
    
        return obs
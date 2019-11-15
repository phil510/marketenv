import abc
import numpy as np
import warnings
import os

from .environments import MarketEnvironment
from .simulators import (MultiRegimeGBM, 
                         GBMRegime, 
                         HistoricalMarket,
                         UniformPrices)

PATH = os.path.abspath(os.path.dirname(__file__))

def make_env(env_name, episode_len = 252):
    if env_name == 'MiniMarket-v1':
        init_prices = UniformPrices(0.8, 1.2, (4, ))
        simulator = MultiRegimeGBM(init_prices, episode_len = episode_len)
        
        return_period = 'annual'
        
        bull_regime = GBMRegime()
        mu = np.array([0.15, 0.10, 0.05, -0.05])
        sigma = np.array([0.15, 0.2, 0.04, 0.35])
        rho = np.array([[ 1.00,  0.80,  0.30, -0.35],
                        [ 0.80,  1.00,  0.30, -0.50],
                        [ 0.30,  0.30,  1.00, -0.35],
                        [-0.35, -0.50, -0.35,  1.00]])
        bull_regime.set_params(mu, sigma, rho, return_period)
        
        bear_regime = GBMRegime()
        mu = np.array([-0.125, -0.075, 0.04, 0.20])
        sigma = np.array([0.2, 0.225, 0.08, 0.40])
        rho = np.array([[ 1.00,  0.90,  0.35, -0.50],
                        [ 0.90,  1.00,  0.45, -0.65],
                        [ 0.35,  0.45,  1.00, -0.60],
                        [-0.50, -0.65, -0.60,  1.00]])
        bear_regime.set_params(mu, sigma, rho, return_period)
        
        init_probs = np.array([0.80, 0.20])
        transition_probs = np.array([[0.9995, 0.0005],
                                     [0.0020, 0.9980]])
        
        simulator.set_params([bull_regime, bear_regime], 
                              transition_probs, init_probs)
        env = MarketEnvironment(simulator, 4, beginning_cash = 10000.0, 
                                commission = 0.0)
 
    else:
        raise NotImplementedError('Cannot create {}'.format(env_name))
                  
    return env
    
def describe_env(env_name):
    if env_name == 'MiniMarket-v1':
        print('MiniMarket-v1'
        '\nMiniMarket-v1 is a small, long only stock market environment.' 
        '\nThere are four tradable assets, which are simulated using'
        '\nGeometric Brownian motion (GBM), and zero non-tradable data series.'
        '\nAt the beginning of each episode, an agent has $10,000 with which'
        '\nto trade, and no commissions are charged when trading.'
        '\n'
        '\nObservation Space: '
        '\nVector of floats shape (9, )'
        '\nThe first item is the current cash balance.'
        '\nThe second through fifth items are the current number of shares' 
        '\nowned for each tradable asset.'
        '\nThe last four items are the current prices for each tradable asset.'
        '\n'
        '\nAction Space:'
        '\nVector of integers shape (4, )'
        '\nAn agent cannot sell more shares than it owns and cannot purchase'
        '\nmore shares than it has cash available for.'
        '\n'
        '\nReward Range:'
        '\n(-inf, inf)'
        '\nThe reward at each time step is the change in current portfolio'
        '\nvalue, which is calculated as:'
        '\nportfolio_value = cash_balance + np.dot(positions, current_prices)')
 
    else:
        raise NotImplementedError('Cannot describe {}'.format(env_name))
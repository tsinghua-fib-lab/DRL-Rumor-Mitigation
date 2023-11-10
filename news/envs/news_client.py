import logging
import math
import copy
# import pickle
from pprint import pprint
from typing import Tuple, Dict, List, Text, Callable
from functools import partial

import numpy as np
import geopandas as gpd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# from urban_planning.envs.plan_client import PlanClient
from news.envs.news import News
from news.utils.config import Config

import time, random, pickle
import networkx as nx

class InfeasibleActionError(ValueError):
    """An infeasible action were passed to the env."""

    def __init__(self, action, mask):
        """Initialize an infeasible action error.

        Args:
          action: Infeasible action that was performed.
          mask: The mask associated with the current observation. mask[action] is
            `0` for infeasible actions.
        """
        super().__init__(self, action, mask)
        self.action = action
        self.mask = mask

    def __str__(self):
        return 'Infeasible action ({}) when the mask is ({})'.format(
            self.action, self.mask)


def reward_info_function(news: News, stage) -> Tuple[float, Dict]:
    
    # if stage == 'done':
    final_i_rate = news.get_final_i_rate()
    total_i_rate = news.get_total_i_rate()
    full_final_i_rate = news.get_full_final_i_rate()
    full_total_i_rate = news.get_full_total_i_rate()
    reward = news.get_reward() * 10
    # else:
    #     final_i_rate = 0
    #     total_i_rate = 0
    #     full_final_i_rate = 0
    #     full_total_i_rate = 0
    #     reward = 0

    return reward, {'reward': reward, 'fir': final_i_rate, 'tir': total_i_rate, \
                    'ffir': full_final_i_rate, 'ftir': full_total_i_rate}


class NewsEnv:

    FAILURE_REWARD = -4.0
    INTERMEDIATE_REWARD = -4.0

    def __init__(self,
                 cfg: Config,
                 is_eval: bool = False,
                 reward_info_fn=reward_info_function):

        self.cfg = cfg
        self._is_eval = is_eval
        self._frozen = False
        self._action_history = []
        self._news =  self.load_graph(cfg)
        self._copy_news = copy.deepcopy(self._news)
        self._reward_info_fn = partial(reward_info_fn)

        self._done = False


    def load_graph(self,cfg):
        
        data_source = cfg.data_source
        spread_param = cfg.env_param
        spread_param['seed'] = cfg.seed
        
        n = News(data_source, spread_param)

        return n

    def _set_cached_reward_info(self):
        """
        Set the cached reward.
        """
        if not self._frozen:
            self._cached_life_circle_reward = -1.0
            self._cached_greeness_reward = -1.0
            self._cached_concept_reward = -1.0

            self._cached_life_circle_info = dict()
            self._cached_concept_info = dict()

            self._cached_land_use_reward = -1.0
            self._cached_land_use_gdf = self.snapshot_land_use()

    def get_reward_info(self) -> Tuple[float, Dict]:
        return self._reward_info_fn(self._news, self._stage)


    def eval(self):
        self._is_eval = True

    def train(self):
        self._is_eval = False

    def get_info(self):
        return self._news.get_env_info_dict()

    def get_numerical_feature_size(self):
        return self._news.get_numerical_dim()

    def get_node_dim(self):
        return self._news.get_node_dim()
    
    def get_edge_dim(self):
        return self._news.get_edge_dim()
    
    def get_max_node_num(self):
        return self._news.get_max_node_num()
    
    def get_max_edge_num(self):
        return self._news.get_max_edge_num()
    
    def get_stage(self):
        if self._stage == 'build':
            return [1,0]
        elif self._stage == 'done':
            return [0,1]

    def _get_obs(self) -> List:
        numerical, node_feature, edge_feature, edge_index, node_mask = self._news.get_obs()
        stage = self.get_stage()

        return [numerical, node_feature, edge_feature, edge_index, node_mask, stage]

    def action(self, action):
        self._news.cut_edge_from_action(int(action))

    def get_action_num(self):
        return self._news.get_cut_num()
    
    def get_total_action(self):
        return self._news.get_total_cut_num()

    def snapshot_land_use(self):
        return self._news.snapshot()
       
    def save_step_data(self):
        return
        self._news.save_step_data()

    def failure_step(self, logging_str, logger):
        """
        Logging and reset after a failure step.
        """
        logger.info('{}: {}'.format(logging_str, self._action_history))
        info = {
            'road_network': -1.0,
            'life_circle': -1.0,
            'greeness': -1.0,
        }
        return self._get_obs(), self.FAILURE_REWARD, True, info


    def step(self, action, logger: logging.Logger) -> Tuple[List, float, bool, Dict]:
        if self._done:
            raise RuntimeError('Action taken after episode is done.')

        else:
            if self._stage == 'build':
                if self.get_action_num() >=  self.get_total_action():
                    self.transition_stage()
                else:
                    self.action(action)
                    self._action_history.append(int(action))

            if self._news.get_done():
                self.transition_stage()

            reward, info = self.get_reward_info()
            if self._stage == 'done':
                self.save_step_data()

        return self._get_obs(), reward, self._done, info

    def reset(self,eval=False,agent_dict=None):
        # self._news = copy.deepcopy(self._copy_news)
        self._news.reset(eval, agent_dict=agent_dict)
        self._action_history = []
        self._set_stage()
        self._done = False

        return self._get_obs()
    
    def get_env_info_dict(self):
        return self._news.get_env_info_dict()

    def _set_stage(self):
        self._stage = 'build'

    def transition_stage(self):
        if self._stage == 'build':
            self._stage = 'done'
            self._done = True
        else:
            raise RuntimeError('Error stage!')
        
    def plot_and_save(self,
                          save_fig: bool = False,
                          path: Text = None,
                          show=False) -> None:
        """
        Plot and save the gdf.
        """
        self._news.plot()
        if save_fig:
            assert path is not None
            plt.savefig(path + '.svg', format='svg', transparent=True)
            data = plt.gca().get_lines()
            y_data = []
            for d in data:
                x, y = d.get_data()
                y_data.append(y)

            # save only numbers to txt, with , as delimiter

            with open(path + '.txt', 'w') as f:
                for i in range(len(y_data)):
                    f.write('[')
                    for y_idx in range(len(y_data[i])):
                        if y_idx < len(y_data[i]) - 1:
                            f.write(str(y_data[i][y_idx]) + ',')
                        else:
                            f.write(str(y_data[i][y_idx]))
                    f.write(']\n')

        if show:
            plt.show()

        plt.cla()
        plt.close('all')

    def visualize(self,
                  save_fig: bool = False,
                  path: Text = None,
                  show=False, final=None) -> None:
        """
        Visualize the city plan.
        """
        self.plot_and_save(save_fig, path, show)

    # def load_plan(self, gdf: GeoDataFrame) -> None:
    #     """
    #     Load a city plan.
    #     """
    #     self._plc.load_plan(gdf)

    # def score_plan(self, verbose=True) -> Tuple[float, Dict]:
    #     """
    #     Score the city plan.
    #     """
    #     reward, info = self._get_all_reward_info()
    #     if verbose:
    #         print(f'reward: {reward}')
    #         pprint(info, indent=4, sort_dicts=False)
    #     return reward, info

    # def get_init_plan(self) -> Dict:
    #     """
    #     Get the gdf of the city plan.
    #     """
    #     return self._plc.get_init_plan()

import networkx as nx
import numpy as np
import random, pickle
import copy
from itertools import combinations
# import geopandas as gpd
# import pandas as pd
# from shapely import geometry
# from haversine import haversine, Unit
import matplotlib.pyplot as plt
# import plotly.express as px

import datetime
import time

RESULT = 0
NETWORK_ID = 62985251
SOURCE = None
# COMMUNITY = [90888992, 200559228, 17434613, 115224382, 16503181, 28019653, 17675120, 72720307, 73050189, 546135380, 211277445, 210428550, 57496410, 318505435, 44483734]
COMMUNITY = []
SIZE = 'k'

class News(object):

    def __init__(self, data_source, spread_param) -> None:
    
        self.data_source = data_source
        self.spread_param = spread_param
        self.graph_construct()
        # self.dynamic_init()
    
    def graph_construct(self):
        if self.data_source == 'twitter':
            if self.spread_param['network_size'] == 'h':
                data = pickle.load(open('./data/t1he.pkl', 'rb'))
            elif self.spread_param['network_size'] == 'k':
                data = pickle.load(open('./data/t1ke.pkl', 'rb'))
            elif self.spread_param['network_size'] == 'w':
                data = pickle.load(open('./data/t1we.pkl', 'rb'))
            else:
                raise ValueError('network_size error')
        elif self.data_source == 'facebook':
            if self.spread_param['network_size'] == 'h':
                data = pickle.load(open('./data/f1he.pkl', 'rb'))
            elif self.spread_param['network_size'] == 'k':
                data = pickle.load(open('./data/f1ke.pkl', 'rb'))
            elif self.spread_param['network_size'] == 'w':
                data = pickle.load(open('./data/f1we.pkl', 'rb'))
            else:
                raise ValueError('network_size error')
                
        valid_network = data[0]
        self.max_node_num = data[1]
        self.max_edge_num = data[2]
        self.network_id = random.choice(valid_network)

        if RESULT:
            self.network_id = NETWORK_ID

        edges_file_path = f"./data/{self.data_source}/{self.network_id}.edges"

        self.G = nx.DiGraph()
            # with open(feat_file_path, 'r') as feat_file:
            #     for line in feat_file:
            #         parts = line.strip().split()
            #         if len(parts) > 0:
            #             node = int(parts[0])
            #             self.G.add_node(node)

        with open(edges_file_path, 'r') as edges_file:
            for line in edges_file:
                parts = line.strip().split()
                if len(parts) == 2:
                    follower = int(parts[0])
                    followee = int(parts[1])
                    if self.data_source == 'twitter':
                        self.G.add_edge(followee, follower)
                    elif self.data_source == 'facebook':
                        self.G.add_edge(followee, follower)
                        self.G.add_edge(follower, followee)

        self.G = nx.DiGraph(self.G.subgraph(max(nx.strongly_connected_components(self.G), key=len)))
        
        self.node_list = [n for n in self.G.nodes()]
        self.node_list_idx = dict(zip(self.node_list, [i for i in range(len(self.node_list))]))
        self.init_node_successor_num = dict(zip(self.node_list, [len(list(self.G.successors(n))) for n in self.node_list]))
        self.node_predecessor_num = dict(zip(self.node_list, [len(list(self.G.predecessors(n))) for n in self.node_list]))
        avg_init_node_successor_num = sum(self.init_node_successor_num.values()) / len(self.node_list)
        self.vailde_source = [n for n in self.node_list if self.init_node_successor_num[n] > avg_init_node_successor_num]
        self.total_cut = len(self.G.edges()) * self.spread_param['total_cut_ration']
        if self.total_cut < 5:
            self.graph_construct()

        if self.spread_param['source'] is None:
            avg_init_node_successor_num = sum(self.init_node_successor_num.values()) / len(self.node_list)
            initial_infected_node = random.choice(self.node_list)
            try_count = 0
            while self.init_node_successor_num[initial_infected_node] < max(avg_init_node_successor_num, 5) or initial_infected_node in COMMUNITY:
                initial_infected_node = random.choice(self.node_list)
                try_count += 1
                if try_count > 100:
                    assert('no initial infected node')
            self.source = initial_infected_node
        else:
            self.source = self.spread_param['source']

        if RESULT and SOURCE is not None:
            self.source = SOURCE

        print(self.network_id, self.source)

        dis_list = np.arange(0.5, 15, 0.5)
        self.node_cut_dis = dict(zip(dis_list, [0 for i in range(len(self.node_list))]))
        self.node_dis = dict(zip(dis_list, [0 for i in range(len(self.node_list))]))
        init_shorest_path = dict(nx.shortest_path_length(self.G, source = self.source))
        for e in self.G.edges():
            dis1 = init_shorest_path[e[0]]
            dis2 = init_shorest_path[e[1]]
            self.node_dis[(dis1+dis2)/2] += 1

        # self.G_deepcopy = copy.deepcopy(self.G)

    def dynamic_init(self):
        self.init_result()
        self.node_cut_successor_num = dict(zip(self.node_list, [0 for i in range(len(self.node_list))]))
        self.node_cut_predecessor_num = dict(zip(self.node_list, [0 for i in range(len(self.node_list))]))
        self.edge_list = [e for e in self.G.edges()]
        self.edge_index = self._cal_edge_index()
        self.cut_num = 0

    def init_result(self):
        if hasattr(self, 'result_full'):
            del self.result_full
        if hasattr(self, 'result_cut'):
            del self.result_cut
        if hasattr(self, 'result_cut_pre'):
            del self.result_cut_pre

    def reset(self,eval=False,agent_dict=None):
        self.eval = eval
        if agent_dict is not None:
            self.spread_param['total_cut_ration'] = agent_dict['cut_ration']
        self.graph_construct()
        self.dynamic_init()
        self.propagation_simulation()

    def propagation_eval(self):
        if self.spread_param['model'] == 'SIR':
            sir_gamma = self.spread_param['gamma']
            sir_beta = self.spread_param['beta']

            total_steps = 30
            wfir = [0 for i in range(total_steps)]
            wtir = [0 for i in range(total_steps)]
            
            final_i_rate = 0
            total_i_rate = 0
            count = 0
            simulation_count = self.spread_param['simulation_count'] * 5 
            for _ in range(simulation_count):
                for node in self.node_list:
                    self.G.nodes[node]['status'] = 'S'
                self.G.nodes[self.source]['status'] = 'I'

                for t in range(total_steps):
                    new_status = {}
                    for node in self.node_list:
                        if self.G.nodes[node]['status'] == 'I' and node != self.source:
                            new_status[node] = 'R' if random.random() < sir_gamma else 'I'
                        elif self.G.nodes[node]['status'] == 'S':
                            sources = list(self.G.predecessors(node))
                            infected_successors = sum(1 for s in sources if self.G.nodes[s]['status'] == 'I')
                            new_status[node] = 'I' if random.random() < (1 - (1 - sir_beta) ** infected_successors) else 'S'
                
                    for node, status in new_status.items():
                        self.G.nodes[node]['status'] = status

                    if len(COMMUNITY) > 0:
                        final_statuses = [self.G.nodes[node]['status'] for node in COMMUNITY]
                    else:
                        final_statuses = [self.G.nodes[node]['status'] for node in self.node_list]
                    infected_count = final_statuses.count('I')
                    recovered_count = final_statuses.count('R')
                    wfir[t] += infected_count
                    wtir[t] += infected_count + recovered_count

                    if t == self.spread_param['spread_steps'] - 1:
                        final_i_rate += infected_count
                        total_i_rate += (infected_count + recovered_count)
                        count += 1


            # final_i_rate = final_i_rate / (self.spread_param['total_steps'] * len(self.node_list) + 1e-8)
            # total_i_rate = total_i_rate / (self.spread_param['total_steps'] * len(self.node_list) + 1e-8)
            wfir = np.array(wfir) / (simulation_count * len(self.node_list) + 1e-8)
            wtir = np.array(wtir) / (simulation_count * len(self.node_list) + 1e-8)

            # update result
            # if not hasattr(self, 'result_full'):
            #     self.result_full = [final_i_rate, total_i_rate]

            # if not hasattr(self, 'result_cut_pre'):
            #     self.result_cut_pre = [final_i_rate, total_i_rate]
            # else:
            #     self.result_cut_pre = self.result_cut

            # self.result_cut = [final_i_rate, total_i_rate]

            return wfir, wtir


    def propagation_simulation(self):
        # t0 = time.time()
        if self.spread_param['model'] == 'SIR':
            sir_gamma = self.spread_param['gamma']
            sir_beta = self.spread_param['beta']

            final_i_rate = 0
            total_i_rate = 0
            count = 0
            simulation_count = self.spread_param['simulation_count'] * 5 if self.eval else self.spread_param['simulation_count']
            for _ in range(simulation_count):
                for node in self.node_list:
                    self.G.nodes[node]['status'] = 'S'
                self.G.nodes[self.source]['status'] = 'I'
            
                for t in range(self.spread_param['spread_steps']):
                    new_status = {}
                    for node in self.node_list:
                        if self.G.nodes[node]['status'] == 'I' and node != self.source:
                            new_status[node] = 'R' if random.random() < sir_gamma else 'I'
                        elif self.G.nodes[node]['status'] == 'S':
                            sources = list(self.G.predecessors(node))
                            infected_successors = sum(1 for s in sources if self.G.nodes[s]['status'] == 'I')
                            new_status[node] = 'I' if random.random() < (1 - (1 - sir_beta) ** infected_successors) else 'S'
                
                    for node, status in new_status.items():
                        self.G.nodes[node]['status'] = status

                if len(COMMUNITY) > 0:
                    final_statuses = [self.G.nodes[node]['status'] for node in COMMUNITY]
                else:
                    final_statuses = [self.G.nodes[node]['status'] for node in self.node_list]
                infected_count = final_statuses.count('I')
                recovered_count = final_statuses.count('R')

                if infected_count + recovered_count > max(0.01 * len(self.node_list), 5):
                    final_i_rate += infected_count
                    total_i_rate += (infected_count + recovered_count)
                    count += 1

            final_i_rate = final_i_rate / (count * len(self.node_list) + 1e-8)
            total_i_rate = total_i_rate / (count * len(self.node_list) + 1e-8)

            if not hasattr(self, 'result_full'):
                self.result_full = [final_i_rate, total_i_rate]

            if not hasattr(self, 'result_cut_pre'):
                self.result_cut_pre = [final_i_rate, total_i_rate]
            else:
                self.result_cut_pre = self.result_cut

            self.result_cut = [final_i_rate, total_i_rate]
            # t1 = time.time()
            # print('propagation_simulation: ',t1-t0)
            # print('result_full: ',self.result_full)
            # print('result_cut_pre: ',self.result_cut_pre)
            # print('result_cut: ',self.result_cut)
        
            return final_i_rate, total_i_rate

    


    def _cal_node_static(self):

        def _dict_normalize(d):
            avg = sum(d.values()) / len(d)
            if avg == 0:
                return d
            return {k: v / avg for k, v in d.items()}
        
        def _dict_zero(d):
            return {k: 0 for k, v in d.items()}
        
        # t0 = time.time()
        degree_cen = nx.degree_centrality(self.G)
        degree_cen = _dict_normalize(degree_cen)
        # degree_cen = _dict_zero(degree_cen)
        # t1 = time.time()
        betweenness_cen = nx.betweenness_centrality(self.G , normalized = False)
        betweenness_cen = _dict_normalize(betweenness_cen)
        # betweenness_cen = _dict_zero(betweenness_cen)
        # t2 = time.time()
        betweenness_cen_s = nx.betweenness_centrality_subset(self.G, normalized = False, sources=[self.source], targets=self.node_list)
        betweenness_cen_s = _dict_normalize(betweenness_cen_s)
        # betweenness_cen_s = _dict_zero(betweenness_cen_s)
        # t3 = time.time()
        eigenvector_cen = nx.eigenvector_centrality_numpy(self.G)
        eigenvector_cen = _dict_normalize(eigenvector_cen)
        # eigenvector_cent = _dict_zero(eigenvector_cen)
        # t4 = time.time()
        closeness_cen = nx.closeness_centrality(self.G)
        closeness_cen = _dict_normalize(closeness_cen)
        # closeness_cent = _dict_zero(closeness_cen)
        # t5 = time.time()
        clustering_cen = nx.clustering(self.G)
        clustering_cen = _dict_normalize(clustering_cen)
        # clustering_cen = _dict_zero(clustering_cen)
        # t6 = time.time()
        shortest_path = dict(nx.shortest_path_length(self.G, source = self.source))
        shortest_path = _dict_normalize(shortest_path)
        # shortest_path = _dict_zero(shortest_path)
        # shortest_path2 = dict(nx.shortest_path_length(self.G, target = self.source))
        # t7 = time.time()


        # print(len(self.node_list))
        # print(len(self.edge_list))
        # print(len(shortest_path))

        # print('degree_cen: ',t1-t0)
        # print('betweenness_cen: ',t2-t1)
        # print('betweenness_cen_s: ',t3-t2)
        # print('eigenvector_cen: ',t4-t3)
        # print('closeness_cen: ',t5-t4)
        # print('clustering_cen: ',t6-t5)
        # print('shortest_path: ',t7-t6)

        node_static = {}
        for node in self.node_list:
            if node not in shortest_path:
                shortest_path[node] = self.max_node_num - 1

            successors = len(list(self.G.successors(node)))
            presuccessor = len(list(self.G.predecessors(node)))
                
            node_static[node] = [degree_cen[node], betweenness_cen[node], betweenness_cen_s[node], eigenvector_cen[node], closeness_cen[node], clustering_cen[node], successors, presuccessor, shortest_path[node]]

        return node_static

    def _cal_edge_static(self):
        def _dict_normalize(d):
            avg = sum(d.values()) / len(d)
            return {k: v / avg for k, v in d.items()}

        def _dict_zero(d):
            return {k: 0 for k, v in d.items()}
        
        e_betweenness_cen_s = nx.edge_betweenness_centrality_subset(self.G, normalized = False, sources=[self.source], targets=self.node_list)
        e_betweenness_cen_s = _dict_normalize(e_betweenness_cen_s)
        # e_betweenness_cen_s = _dict_zero(e_betweenness_cen_s)


        edge_static = {}
        for e in self.edge_list:
            successors_followee = list(self.G.successors(e[0]))
            successors_follower = list(self.G.successors(e[1]))
            successors_difference_len = len(set(successors_followee).symmetric_difference(set(successors_follower)))
            # successors_difference_len = 0
            presuccessor_followee = list(self.G.predecessors(e[0]))
            presuccessor_follower = list(self.G.predecessors(e[1]))
            presuccessor_difference_len = len(set(presuccessor_followee).symmetric_difference(set(presuccessor_follower)))
            # presuccessor_difference_len = 0

            edge_static[e] = [e_betweenness_cen_s[e], successors_difference_len, presuccessor_difference_len]

        return edge_static
    
    def _cal_node_all(self):
        node_static = self._cal_node_static()
        node_all = {}
        for n in self.node_list:
            node_all[n] = node_static[n] + [1 if n == self.source else 0]

        return node_all
    
    def _cal_edge_all(self):
        # edge_static = self._cal_edge_static()
        # edge_all = {}
        # for e in self.edge_list:
        #     edge_all[e] = edge_static[e]

        return self._cal_edge_static()
            
    def get_numerical_dim(self):
        return 4
    
    def get_node_dim(self):
        return 10
    
    def get_edge_dim(self):
        return 3

    def _cal_edge_index(self):
        edge_index = []

        for e in self.edge_list:
            idx1 = self.node_list_idx[e[0]]
            idx2 = self.node_list_idx[e[1]]
            edge_index.append([idx1, idx2])

        edge_index = edge_index + [[self.max_node_num-1, self.max_node_num-1] for i in range(self.max_edge_num - len(self.edge_list))]
        
        return edge_index
    
    
    def _get_numerical(self):
        numerical = [len(self.node_list) / 1000, len(self.edge_list) / 1000, \
                     self.total_cut / len(self.edge_list), self.cut_num / self.total_cut]

        return numerical

            
    def get_obs(self):
        numerical = self._get_numerical()
        
        node_all = self._cal_node_all()
        node_feature = np.concatenate([[node_all[n] for n in self.node_list], np.zeros(((self.max_node_num - len(self.node_list)), self.get_node_dim()))], axis=0)
        edghe_all = self._cal_edge_all()
        edge_feature = np.concatenate([[edghe_all[e] for e in self.edge_list], np.zeros(((self.max_edge_num - len(self.edge_list)), self.get_edge_dim()))], axis=0)
        mask = self.get_mask()

        return numerical, node_feature, edge_feature, self.edge_index, mask


    def cut_edge_from_action(self,action):
        try:
            # print(action, self.total_cut,self.cut_num)
            cut_edge = self.edge_list[action]
        except:
            print((self._cal_mask())[action])
            raise ValueError('action error')
        
        self.G.remove_edge(cut_edge[0],cut_edge[1])
        self.node_cut_successor_num[cut_edge[0]] += 1
        self.node_cut_predecessor_num[cut_edge[1]] += 1
        self.cut_num += 1
        # dis1 = nx.shortest_path_length(self.G, source = self.source, target = cut_edge[0])
        # dis2 = nx.shortest_path_length(self.G, source = self.source, target = cut_edge[1])
        # self.node_cut_dis[(dis1+dis2)/2] += 1

        self.edge_list.remove(cut_edge)
        self.edge_index.remove([self.node_list_idx[cut_edge[0]],self.node_list_idx[cut_edge[1]]])
        self.edge_index.append([self.max_node_num-1, self.max_node_num-1])

        self.propagation_simulation()


    def _cal_mask(self):

        mask = [False for _ in range(self.max_edge_num)]
        for idx in range(len(self.edge_list)):
            edge = self.edge_list[idx]
            # (self.node_cut_successor_num[edge[1]] + 1) <= self.init_node_successor_num[edge[1]] * self.spread_param['max_node_cut_ration'] and \
            # (self.node_cut_predecessor_num[edge[0]] + 1) <= self.node_predecessor_num[edge[0]] * self.spread_param['max_node_cut_ration'] and \
            if (self.node_cut_successor_num[edge[0]] + 1) <= self.init_node_successor_num[edge[0]] * self.spread_param['max_node_cut_ration'] and \
            (self.node_cut_predecessor_num[edge[1]] + 1) <= self.node_predecessor_num[edge[1]] * self.spread_param['max_node_cut_ration']:
                mask[idx] = True
            if len(COMMUNITY) > 0:
                if edge[0] in COMMUNITY or edge[1] in COMMUNITY:
                    mask[idx] = False
        
        if np.array(mask).sum() == 0:
            self.done = 1
        else:
            self.done = 0

        return mask


    def get_cut_num(self):
        return self.cut_num
    
    def get_total_cut_num(self):
        return self.total_cut

    def get_mask(self):
        return self._cal_mask()
    
    def get_edge_index(self):
        return self._cal_edge_index()

    def get_reward(self):
        if self.spread_param['model'] == 'SIR':
            if self.result_full[0] < 1e-1 or self.result_full[1] < 1e-1:
                return 0
            else:
                r1 = (self.result_cut_pre[0] - self.result_cut[0]) / (self.result_full[0] + 1e-8)
                r2 = (self.result_cut_pre[1] - self.result_cut[1]) / (self.result_full[1] + 1e-8)
                # print(self.result_cut_pre[0],self.result_cut[0],self.result_full[0])
                return self.spread_param['reward']['a1'] * r1 + self.spread_param['reward']['a2'] * r2
        
    def get_final_i_rate(self):
        return self.result_cut[0]
    
    def get_total_i_rate(self):
        return self.result_cut[1]
    
    def get_full_final_i_rate(self):
        return self.result_full[0]
    
    def get_full_total_i_rate(self):
        return self.result_full[1]
    
    def get_done(self):
        self._cal_mask()
        return self.done
            
    def get_env_info_dict(self):
        return {'network_id': self.network_id, 'source': self.source, 'node_num': len(self.node_list), 'edge_num': len(self.edge_list)}
    
    def get_max_edge_num(self):
        return self.max_edge_num
    
    def get_max_node_num(self):
        return self.max_node_num
    
    def plot(self):
        self.init_result()
        fir, tir = self.propagation_eval()
        steps = range(len(fir))
        plt.plot(steps, tir, label='TIR', color='red')
        plt.plot(steps, fir, label='CIR', color='blue')
        tr = {k: v / (self.total_cut + 1e-6) for k, v in self.node_cut_dis.items()}
        lr = {k: v / (self.node_dis[k] + 1e-6) for k, v in self.node_cut_dis.items()}
        print(tr)
        print(lr)

        # plt.legend()

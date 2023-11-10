import torch
import torch.nn as nn
import math

class RLStateEncoder(nn.Module):

    EPSILON = 1e-6

    def __init__(self, cfg, agent):
        super().__init__()
        self.cfg = cfg
        self.agent = agent
        self.numerical_feature_encoder = self.create_numerical_feature_encoder(cfg)
        self.node_encoder = self.create_node_encoder(cfg)
        self.num_gcn_layers = cfg['num_gcn_layers']

        self.max_num_nodes = cfg['max_num_nodes']
        self.max_num_edges = cfg['max_num_edges']

        self.attention_layer1 = nn.MultiheadAttention(cfg['gcn_node_dim'], cfg['num_attention_heads'])
        self.attention_query_layer1 = nn.Linear(cfg['gcn_node_dim'],cfg['gcn_node_dim'])
        self.attention_key_layer1 = nn.Linear(cfg['gcn_node_dim'],cfg['gcn_node_dim'])
        self.attention_value_layer1 = nn.Linear(cfg['gcn_node_dim'],cfg['gcn_node_dim'])

        self.output_policy_road_size = cfg['gcn_node_dim']
        self.output_value_size = cfg['gcn_node_dim'] + cfg['state_encoder_hidden_size'][-1] + 2

    def create_numerical_feature_encoder(self, cfg):
        """Create the numerical feature encoder."""
        feature_encoder = nn.Sequential()
        for i in range(len(cfg['state_encoder_hidden_size'])):
            if i == 0:
                feature_encoder.add_module('flatten_{}'.format(i),
                                           nn.Flatten())
                feature_encoder.add_module(
                    'linear_{}'.format(i),
                    nn.Linear(self.agent.numerical_feature_size,
                              cfg['state_encoder_hidden_size'][i]))
            else:
                feature_encoder.add_module(
                    'linear_{}'.format(i),
                    nn.Linear(cfg['state_encoder_hidden_size'][i - 1],
                              cfg['state_encoder_hidden_size'][i]))
            feature_encoder.add_module('tanh_{}'.format(i), nn.Tanh())
        return feature_encoder

    def create_node_encoder(self, cfg):
        feature_encoder = nn.Sequential()
        feature_encoder.add_module(
            'linear_{}'.format(1),
            nn.Linear(7, cfg['gcn_node_dim']))
        feature_encoder.add_module('tanh_{}'.format(2), nn.Tanh())

        return feature_encoder

    def self_attention1(self, h_nodes):
        """Self attention."""
        query = self.attention_query_layer1(h_nodes).transpose(0, 1)
        keys = self.attention_key_layer1(h_nodes).transpose(0, 1)
        values = self.attention_value_layer1(h_nodes).transpose(0, 1)
        h_current_node_attended, _ = self.attention_layer1(query, keys, values)
        h_current_node_attended = h_current_node_attended.transpose(0, 1).squeeze(1)
        return h_current_node_attended
    
    def attention1(self,h_nodes_q,h_nodes_kv):
        """Self attention."""
        query = self.attention_query_layer1(h_nodes_q).transpose(0, 1)
        keys = self.attention_key_layer1(h_nodes_kv).transpose(0, 1)
        values = self.attention_value_layer1(h_nodes_kv).transpose(0, 1)
        h_current_node_attended, _ = self.attention_layer1(query, keys, values)
        h_current_node_attended = h_current_node_attended.transpose(0, 1).squeeze(1)
        return h_current_node_attended    
    

    @staticmethod
    def batch_data(x):
        numerical, node_feature, edge_index_dis, edge_index_od , mask, stage = zip(*x)
        numerical = torch.stack(numerical)
        node_feature = torch.stack(node_feature)
        edge_index_dis = torch.stack(edge_index_dis)
        edge_index_od = torch.stack(edge_index_od)
        mask = torch.stack(mask)
        stage = torch.stack(stage)

        return numerical, node_feature, edge_index_dis, edge_index_od , mask, stage

    @staticmethod
    def mean_features(h, mask=None):
        if mask is not None:
            mean_h = (h * mask.unsqueeze(-1).float()).sum(dim=1) / mask.float().sum(dim=1, keepdim=True)
        else:
            mean_h = (h).mean(dim=1)
        return mean_h

    def forward(self, x):
        numerical, node_feature, _, _, mask, stage = self.batch_data(x)
        node_feature = node_feature[:,:,[0,1,-5,-4,-3,-2,-1]]

        h_numerical_features = self.numerical_feature_encoder(numerical.to(torch.float32))
        h_nodes = self.node_encoder(node_feature.to(torch.float32))

        build_idx = torch.where(node_feature[:,:,-5][0] > 0)[0].tolist()
        h_node_attended = self.attention1(h_nodes, h_nodes[:,build_idx,:])

        h_nodes_mean = self.mean_features(h_node_attended)

        state_value = torch.cat([h_numerical_features, h_nodes_mean, stage],dim=-1)
        state_policy = torch.cat([h_node_attended.to(torch.float32)],dim=-1)

        return state_policy, state_value, mask, stage

class GNN1StateEncoder(nn.Module):
    """
    New GNN state encoder network.
    """
    EPSILON = 1e-6

    def __init__(self, cfg, agent):
        super().__init__()
        self.cfg = cfg
        self.agent = agent
        self.numerical_feature_encoder = self.create_numerical_feature_encoder(
            cfg)
        self.node_encoder = nn.Linear(agent.node_dim, cfg['gcn_node_dim'])
        self.num_gcn_layers = cfg['num_gcn_layers']
        self.num_edge_fc_layers = cfg['num_edge_fc_layers']

        self.edge_fc_layers1 = self.create_edge_fc_layers1(cfg)
        self.edge_fc_layers2 = self.create_edge_fc_layers2(cfg)

        self.max_num_nodes = cfg['max_num_nodes']
        self.max_num_edges = cfg['max_num_edges']

        self.attention_layer1 = nn.MultiheadAttention(cfg['gcn_node_dim'], cfg['num_attention_heads'])
        self.attention_query_layer1 = nn.Linear(cfg['gcn_node_dim'],cfg['gcn_node_dim'])
        self.attention_key_layer1 = nn.Linear(cfg['gcn_node_dim'],cfg['gcn_node_dim'])
        self.attention_value_layer1 = nn.Linear(cfg['gcn_node_dim'],cfg['gcn_node_dim'])

        self.attention_layer2 = nn.MultiheadAttention(cfg['gcn_node_dim'], cfg['num_attention_heads'])
        self.attention_query_layer2 = nn.Linear(cfg['gcn_node_dim'],cfg['gcn_node_dim'])
        self.attention_key_layer2 = nn.Linear(cfg['gcn_node_dim'],cfg['gcn_node_dim'])
        self.attention_value_layer2 = nn.Linear(cfg['gcn_node_dim'],cfg['gcn_node_dim'])

        self.output_policy_road_size = cfg['gcn_node_dim'] * 2
        self.output_value_size = cfg['gcn_node_dim'] * 2 + cfg['state_encoder_hidden_size'][-1] + 2

    def create_numerical_feature_encoder(self, cfg):
        """Create the numerical feature encoder."""
        feature_encoder = nn.Sequential()
        for i in range(len(cfg['state_encoder_hidden_size'])):
            if i == 0:
                feature_encoder.add_module('flatten_{}'.format(i),
                                           nn.Flatten())
                feature_encoder.add_module(
                    'linear_{}'.format(i),
                    nn.Linear(self.agent.numerical_feature_size,
                              cfg['state_encoder_hidden_size'][i]))
            else:
                feature_encoder.add_module(
                    'linear_{}'.format(i),
                    nn.Linear(cfg['state_encoder_hidden_size'][i - 1],
                              cfg['state_encoder_hidden_size'][i]))
            feature_encoder.add_module('tanh_{}'.format(i), nn.Tanh())
        return feature_encoder

    def create_edge_fc_layers1(self, cfg):
        """Create the edge fc layers."""

        def create_edge_fc():
            seq = nn.Sequential()
            for i in range(self.num_edge_fc_layers):
                seq.add_module(
                    'linear_{}'.format(i),
                    nn.Linear(cfg['gcn_node_dim'], cfg['gcn_node_dim']))
                seq.add_module('tanh_{}'.format(i), nn.Tanh())
            return seq

        edge_fc_layers = nn.ModuleList()
        for _ in range(self.num_gcn_layers):
            edge_fc_layers.append(create_edge_fc())
        return edge_fc_layers

    def create_edge_fc_layers2(self, cfg):
        """Create the edge fc layers."""

        def create_edge_fc():
            seq = nn.Sequential()
            for i in range(self.num_edge_fc_layers):
                if i == 0:
                    seq.add_module(
                        'linear_{}'.format(i),
                        nn.Linear(cfg['gcn_node_dim'],cfg['gcn_node_dim']))
                    seq.add_module('tanh_{}'.format(i), nn.Tanh())
                else:
                    seq.add_module(
                        'linear_{}'.format(i),
                        nn.Linear(cfg['gcn_node_dim'], cfg['gcn_node_dim']))
                    seq.add_module('tanh_{}'.format(i), nn.Tanh())
            return seq

        edge_fc_layers = nn.ModuleList()
        for _ in range(self.num_gcn_layers):
            edge_fc_layers.append(create_edge_fc())
        return edge_fc_layers

    def scatter_count(self, h_edges, indices, max_num_nodes):
        """
        Aggregate edge embeddings to nodes.

        Args:
            h_edges (torch.Tensor): Edge embeddings. Shape: (batch, max_num_edges, node_dim).
            indices (torch.Tensor): Node indices. Shape: (batch, max_num_edges).
            mask (torch.Tensor): Edge mask. Shape: (batch, max_num_edges).
            max_num_nodes (int): Maximum number of nodes.

        Returns:
            h_nodes (torch.Tensor): Node embeddings. Shape: (batch, max_num_nodes, node_dim).
            count_edge (torch.Tensor): Edge counts per node. Shape: (batch, max_num_nodes, node_dim).
        """
        batch_size = h_edges.shape[0]
        num_latents = h_edges.shape[2]

        h_nodes = torch.zeros(batch_size, max_num_nodes,num_latents).to(h_edges.device)
        count_edge = torch.zeros_like(h_nodes)
        count = torch.ones_like(h_edges).float()

        idx = indices.unsqueeze(-1).expand(-1, -1, num_latents)
        h_nodes = h_nodes.scatter_add_(1, idx, h_edges)
        count_edge = count_edge.scatter_add_(1, idx, count)
        return h_nodes, count_edge

    def gather_to_edges(self, h_nodes, edge_index,
                        edge_fc_layer):
        """
        Gather node embeddings to edges.

        Args:
            h_nodes (torch.Tensor): Node embeddings. Shape: (batch, max_num_nodes, node_dim).
            edge_index (torch.Tensor): Edge indices. Shape: (batch, max_num_edges, 2).
            mask (torch.Tensor): Edge mask. Shape: (batch, max_num_edges).
            edge_fc_layer (torch.nn.Module): Edge fc layer.

        Returns:
            h_edges (torch.Tensor): edge embeddings. Shape: (batch, max_num_edges, node_dim).
        """
        h_edges_12 = torch.gather(h_nodes, 1, edge_index[:, :, 0].unsqueeze(-1).expand(-1, -1, h_nodes.size(-1)))
        h_edges_12 = edge_fc_layer(h_edges_12.to(torch.float32))

        h_edges_21 = torch.gather(h_nodes, 1, edge_index[:, :, 1].unsqueeze(-1).expand(-1, -1, h_nodes.size(-1)))
        h_edges_21 = edge_fc_layer(h_edges_21.to(torch.float32))

        h_edges = (h_edges_12 + h_edges_21) / 2

        return h_edges, h_edges_12, h_edges_21

    def scatter_to_nodes(self, h_edges, edge_index, max_num_nodes):
        """
        Scatter edge embeddings to nodes.

        Args:
            h_edges (tuple of torch.Tensor): edge embedding, m12, m21. Shape: (batch, max_num_edges, node_dim).
            edge_index (torch.Tensor): Edge indices. Shape: (batch, max_num_edges, 2).
            mask (torch.Tensor): Edge mask. Shape: (batch, max_num_edges).
            max_num_nodes (int): Maximum number of nodes.

        Returns:
            h_nodes (torch.Tensor): Node embeddings. Shape: (batch, max_num_nodes, node_dim).
        """
        _, h_edges_12, h_edges_21 = h_edges
        h_nodes_1, count_1 = self.scatter_count(h_edges_21, edge_index[:, :,0], max_num_nodes)
        h_nodes_2, count_2 = self.scatter_count(h_edges_12, edge_index[:, :,1], max_num_nodes)
        h_nodes = (h_nodes_1 + h_nodes_2) / (count_1 + count_2 + self.EPSILON)
        return h_nodes

    def self_attention1(self, h_nodes):
        """Self attention."""
        query = self.attention_query_layer1(h_nodes).transpose(0, 1)
        keys = self.attention_key_layer1(h_nodes).transpose(0, 1)
        values = self.attention_value_layer1(h_nodes).transpose(0, 1)
        h_current_node_attended, _ = self.attention_layer1(query, keys, values)
        h_current_node_attended = h_current_node_attended.transpose(0, 1).squeeze(1)
        return h_current_node_attended
    
    def self_attention2(self, h_nodes):
        """Self attention."""
        query = self.attention_query_layer2(h_nodes).transpose(0, 1)
        keys = self.attention_key_layer2(h_nodes).transpose(0, 1)
        values = self.attention_value_layer2(h_nodes).transpose(0, 1)
        h_current_node_attended, _ = self.attention_layer2(query, keys, values)
        h_current_node_attended = h_current_node_attended.transpose(0, 1).squeeze(1)
        return h_current_node_attended

    @staticmethod
    def batch_data(x):
        numerical, node_feature, edge_index_dis, edge_index_od , mask, stage = zip(*x)
        numerical = torch.stack(numerical)
        node_feature = torch.stack(node_feature)
        edge_index_dis = torch.stack(edge_index_dis)
        edge_index_od = torch.stack(edge_index_od)
        mask = torch.stack(mask)
        stage = torch.stack(stage)

        return numerical, node_feature, edge_index_dis, edge_index_od , mask, stage

    @staticmethod
    def mean_features(h, mask=None):
        if mask is not None:
            mean_h = (h * mask.unsqueeze(-1).float()).sum(dim=1) / mask.float().sum(dim=1, keepdim=True)
        else:
            mean_h = (h).mean(dim=1)
        return mean_h

    def forward(self, x):
        numerical, node_feature, edge_index_dis, edge_index_od, mask, stage = self.batch_data(x)

        h_numerical_features = self.numerical_feature_encoder(numerical.to(torch.float32))
        h_nodes = self.node_encoder(node_feature.to(torch.float32))
        h_nodes_tmp = self.node_encoder(node_feature.to(torch.float32))

        for edge_fc_layer1 in self.edge_fc_layers1:
            h_edges = self.gather_to_edges(h_nodes_tmp,edge_index_dis,edge_fc_layer1)
            h_nodes_new = self.scatter_to_nodes(h_edges, edge_index_dis,node_feature.shape[1])
            h_nodes = h_nodes + h_nodes_new

        for edge_fc_layer2 in self.edge_fc_layers2:
            h_edges = self.gather_to_edges(h_nodes_tmp,edge_index_od,edge_fc_layer2)
            h_nodes_new = self.scatter_to_nodes(h_edges, edge_index_od,node_feature.shape[1])
            h_nodes = h_nodes + h_nodes_new

        h_node_atten_all = self.self_attention1(h_nodes)

        h_nodes_final = torch.cat([h_nodes,h_node_atten_all], dim = -1)
        h_nodes_mean = self.mean_features(h_nodes_final)

        state_value = torch.cat([h_numerical_features, h_nodes_mean, stage],dim=-1)
        state_policy = torch.cat([h_nodes_final.to(torch.float32)],dim=-1)
        
        return state_policy, state_value, mask, stage


class GNN2StateEncoder(nn.Module):
    """
    New GNN state encoder network.
    """
    EPSILON = 1e-6

    def __init__(self, cfg, agent):
        super().__init__()
        self.cfg = cfg
        self.agent = agent
        self.numerical_feature_encoder = self.create_numerical_feature_encoder(cfg)
        self.node_encoder = self.create_node_encoder(cfg,agent)
        self.num_gcn_layers = cfg['num_gcn_layers']
        self.num_edge_fc_layers = cfg['num_edge_fc_layers']

        self.edge_fc = self.create_edge_fc_layers(cfg)
        self.edge_fc_layers1 = self.create_edge_fc_layers1(cfg)

        self.max_num_nodes = agent.max_num_nodes
        self.max_num_edges = agent.max_num_edges

        self.output_policy_road_size = cfg['gcn_node_dim']
        self.output_value_size = cfg['gcn_node_dim'] + cfg['state_encoder_hidden_size'][-1] + 2

    def create_numerical_feature_encoder(self, cfg):
        """Create the numerical feature encoder."""
        feature_encoder = nn.Sequential()
        for i in range(len(cfg['state_encoder_hidden_size'])):
            if i == 0:
                feature_encoder.add_module('flatten_{}'.format(i),
                                           nn.Flatten())
                feature_encoder.add_module(
                    'linear_{}'.format(i),
                    nn.Linear(self.agent.numerical_feature_size,
                              cfg['state_encoder_hidden_size'][i]))
            else:
                feature_encoder.add_module(
                    'linear_{}'.format(i),
                    nn.Linear(cfg['state_encoder_hidden_size'][i - 1],
                              cfg['state_encoder_hidden_size'][i]))
            feature_encoder.add_module('tanh_{}'.format(i), nn.Tanh())
        return feature_encoder
    
    def create_node_encoder(self, cfg, agent):
        feature_encoder = nn.Sequential()
        feature_encoder.add_module(
            'linear_{}'.format(1),
            nn.Linear(agent.node_dim, cfg['gcn_node_dim']))
        feature_encoder.add_module('tanh_{}'.format(2), nn.Tanh())

        return feature_encoder

    def create_edge_fc_layers(self, cfg):
        seq = nn.Sequential()
        for i in range(1):
            seq.add_module(
                'linear_{}'.format(i),
                nn.Linear(cfg['gcn_node_dim'] * 2, cfg['gcn_node_dim']))
            seq.add_module('tanh_{}'.format(i), nn.Tanh())
        return seq

    def create_edge_fc_layers1(self, cfg):
        """Create the edge fc layers."""
        def create_edge_fc():
            seq = nn.Sequential()
            for i in range(self.num_edge_fc_layers):
                if i == 0:
                    seq.add_module(
                        'linear_{}'.format(i),
                        nn.Linear(cfg['gcn_node_dim'] * 2, cfg['gcn_node_dim'])
                    )
                else:
                    seq.add_module(
                        'linear_{}'.format(i),
                        nn.Linear(cfg['gcn_node_dim'], cfg['gcn_node_dim'])
                    )
                seq.add_module(
                    'tanh_{}'.format(i),
                    nn.Tanh()
                )
            return seq
        edge_fc_layers = nn.ModuleList()
        for _ in range(self.num_gcn_layers):
            edge_fc_layers.append(create_edge_fc())
        return edge_fc_layers

    def scatter_count(self, h_edges, indices, edge_mask, max_num_nodes):
        """
        Aggregate edge embeddings to nodes.

        Args:
            h_edges (torch.Tensor): Edge embeddings. Shape: (batch, max_num_edges, node_dim).
            indices (torch.Tensor): Node indices. Shape: (batch, max_num_edges).
            mask (torch.Tensor): Edge mask. Shape: (batch, max_num_edges).
            max_num_nodes (int): Maximum number of nodes.

        Returns:
            h_nodes (torch.Tensor): Node embeddings. Shape: (batch, max_num_nodes, node_dim).
            count_edge (torch.Tensor): Edge counts per node. Shape: (batch, max_num_nodes, node_dim).
        """
        batch_size = h_edges.shape[0]
        num_latents = h_edges.shape[2]

        h_nodes = torch.zeros(batch_size, max_num_nodes,num_latents).to(h_edges.device)
        count_edge = torch.zeros_like(h_nodes)
        count = torch.broadcast_to(edge_mask.unsqueeze(-1), h_edges.shape).float()

        idx = indices.unsqueeze(-1).expand(-1, -1, num_latents)
        h_nodes = h_nodes.scatter_add_(1, idx, h_edges)
        count_edge = count_edge.scatter_add_(1, idx, count)
        return h_nodes, count_edge

    def gather_to_edges(self, h_nodes, edge_index, edge_mask, edge_fc_layer):
        """
        Gather node embeddings to edges.

        Args:
            h_nodes (torch.Tensor): Node embeddings. Shape: (batch, max_num_nodes, node_dim).
            edge_index (torch.Tensor): Edge indices. Shape: (batch, max_num_edges, 2).
            mask (torch.Tensor): Edge mask. Shape: (batch, max_num_edges).
            edge_fc_layer (torch.nn.Module): Edge fc layer.

        Returns:
            h_edges (torch.Tensor): edge embeddings. Shape: (batch, max_num_edges, node_dim).
        """
        h_edges_1 = torch.gather(h_nodes, 1, edge_index[:, :, 0].unsqueeze(-1).expand(-1, -1, h_nodes.size(-1)))
        h_edges_2 = torch.gather(h_nodes, 1, edge_index[:, :, 1].unsqueeze(-1).expand(-1, -1, h_nodes.size(-1)))
        h_edges_12 = torch.cat([h_edges_1,h_edges_2],dim=-1)
        h_edges_21 = torch.cat([h_edges_2,h_edges_1],dim=-1)
        h_edges = (edge_fc_layer(h_edges_12) + edge_fc_layer(h_edges_21)) / 2

        mask = torch.broadcast_to(edge_mask.unsqueeze(-1), h_edges.shape)
        h_edges = torch.where(mask, h_edges, torch.zeros_like(h_edges))
        return h_edges

    def scatter_to_nodes(self, h_edges, edge_index, edge_mask, max_num_nodes):
        """
        Scatter edge embeddings to nodes.

        Args:
            h_edges (tuple of torch.Tensor): edge embedding, m12, m21. Shape: (batch, max_num_edges, node_dim).
            edge_index (torch.Tensor): Edge indices. Shape: (batch, max_num_edges, 2).
            mask (torch.Tensor): Edge mask. Shape: (batch, max_num_edges).
            max_num_nodes (int): Maximum number of nodes.

        Returns:
            h_nodes (torch.Tensor): Node embeddings. Shape: (batch, max_num_nodes, node_dim).
        """
        h_nodes_1, count_1 = self.scatter_count(h_edges, edge_index[:, :,0], edge_mask, max_num_nodes)
        h_nodes_2, count_2 = self.scatter_count(h_edges, edge_index[:, :,1], edge_mask, max_num_nodes)
        h_nodes = (h_nodes_1 + h_nodes_2) / (count_1 + count_2 + self.EPSILON)
        return h_nodes

    @staticmethod
    def batch_data(x):
        numerical, node_feature, edge_feature, edge_index, mask, stage = zip(*x)
        numerical = torch.stack(numerical)
        node_feature = torch.stack(node_feature)
        edge_index = torch.stack(edge_index)
        edge_feature = torch.stack(edge_feature)
        mask = torch.stack(mask)
        stage = torch.stack(stage)

        return numerical, node_feature, edge_feature, edge_index, mask, stage

    @staticmethod
    def mean_features(h, mask=None):
        if mask is not None:
            mean_h = (h * mask.unsqueeze(-1).float()).sum(dim=1) / mask.float().sum(dim=1, keepdim=True)
        else:
            mean_h = (h).mean(dim=1)
        return mean_h

    def forward(self, x):
        numerical, node_feature, edge_feature, edge_index, edge_mask, stage = self.batch_data(x)

        h_numerical_features = self.numerical_feature_encoder(numerical.to(torch.float32))
        h_nodes = self.node_encoder(node_feature.to(torch.float32))

        h_edge_final = self.gather_to_edges(h_nodes,edge_index,edge_mask,self.edge_fc)

        h_final = h_edge_final
        h_mean = self.mean_features(h_final, edge_mask)

        state_value = torch.cat([h_numerical_features, h_mean, stage],dim=-1)
        state_policy = torch.cat([h_final.to(torch.float32)],dim=-1)

        return state_policy, state_value, edge_mask, stage


class GNN3StateEncoder(nn.Module):
    """
    New GNN state encoder network.
    """
    EPSILON = 1e-6

    def __init__(self, cfg, agent):
        super().__init__()
        self.cfg = cfg
        self.agent = agent
        self.numerical_feature_encoder = self.create_numerical_feature_encoder(cfg)
        self.node_encoder = self.create_node_encoder(cfg,agent)
        self.edge_encoder = self.create_edge_encoder(cfg,agent)
        self.num_gcn_layers = cfg['num_gcn_layers']
        self.num_edge_fc_layers = cfg['num_edge_fc_layers']

        self.edge_fc = self.create_edge_fc_layer(cfg)
        self.edge_fc_layers_node = self.create_edge_fc_layers_node(cfg)
        self.edge_fc_layers_edge = self.create_edge_fc_layers_edge(cfg)

        self.max_num_nodes = agent.max_num_nodes
        self.max_num_edges = agent.max_num_edges

        self.output_policy_road_size = cfg['gcn_node_dim'] * 2 + cfg['gcn_edge_dim']
        self.output_value_size = cfg['gcn_node_dim'] + cfg['gcn_edge_dim'] + cfg['state_encoder_hidden_size'][-1]

    def create_numerical_feature_encoder(self, cfg):
        """Create the numerical feature encoder."""
        feature_encoder = nn.Sequential()
        for i in range(len(cfg['state_encoder_hidden_size'])):
            if i == 0:
                feature_encoder.add_module('flatten_{}'.format(i),
                                           nn.Flatten())
                feature_encoder.add_module(
                    'linear_{}'.format(i),
                    nn.Linear(self.agent.numerical_feature_size,
                              cfg['state_encoder_hidden_size'][i]))
            else:
                feature_encoder.add_module(
                    'linear_{}'.format(i),
                    nn.Linear(cfg['state_encoder_hidden_size'][i - 1],
                              cfg['state_encoder_hidden_size'][i]))
            feature_encoder.add_module('tanh_{}'.format(i), nn.Tanh())
        return feature_encoder
    
    def create_node_encoder(self, cfg, agent):
        feature_encoder = nn.Sequential()
        feature_encoder.add_module(
            'linear_{}'.format(1),
            nn.Linear(agent.node_dim, cfg['gcn_node_dim']))
        feature_encoder.add_module('tanh_{}'.format(2), nn.Tanh())

        return feature_encoder
    
    def create_edge_encoder(self, cfg, agent):
        feature_encoder = nn.Sequential()
        feature_encoder.add_module(
            'linear_{}'.format(1),
            nn.Linear(agent.edge_dim, cfg['gcn_edge_dim']))
        feature_encoder.add_module('tanh_{}'.format(2), nn.Tanh())

        return feature_encoder

    def create_edge_fc_layer(self, cfg):
        seq = nn.Sequential()
        for i in range(1):
            seq.add_module(
                'linear_{}'.format(i),
                nn.Linear(cfg['gcn_node_dim'] * 2 + self.agent.node_dim * 2, cfg['gcn_node_dim']))
            seq.add_module('tanh_{}'.format(i), nn.Tanh())
        return seq

    def create_edge_fc_layers_node(self, cfg):
        """Create the edge fc layers."""
        def create_edge_fc():
            seq = nn.Sequential()
            for i in range(self.num_edge_fc_layers):
                if i == 0:
                    seq.add_module(
                        'linear_{}'.format(i),
                        nn.Linear(cfg['gcn_node_dim'] * 2, cfg['gcn_node_dim'])
                    )
                else:
                    seq.add_module(
                        'linear_{}'.format(i),
                        nn.Linear(cfg['gcn_node_dim'], cfg['gcn_node_dim'])
                    )
                seq.add_module(
                    'tanh_{}'.format(i),
                    nn.Tanh()
                )
            return seq
        edge_fc_layers = nn.ModuleList()
        for _ in range(self.num_gcn_layers):
            edge_fc_layers.append(create_edge_fc())
        return edge_fc_layers

    def create_edge_fc_layers_edge(self, cfg):
        """Create the edge fc layers."""
        def create_edge_fc():
            seq = nn.Sequential()
            for i in range(self.num_edge_fc_layers):
                if i == 0:
                    seq.add_module(
                        'linear_{}'.format(i),
                        nn.Linear(cfg['gcn_edge_dim'] * 2, cfg['gcn_edge_dim'])
                    )
                else:
                    seq.add_module(
                        'linear_{}'.format(i),
                        nn.Linear(cfg['gcn_edge_dim'], cfg['gcn_edge_dim'])
                    )
                seq.add_module(
                    'tanh_{}'.format(i),
                    nn.Tanh()
                )
            return seq
        edge_fc_layers = nn.ModuleList()
        for _ in range(self.num_gcn_layers):
            edge_fc_layers.append(create_edge_fc())
        return edge_fc_layers

    def scatter_count(self, h_edges, indices, edge_mask, max_num_nodes):
        """
        Aggregate edge embeddings to nodes.

        Args:
            h_edges (torch.Tensor): Edge embeddings. Shape: (batch, max_num_edges, node_dim).
            indices (torch.Tensor): Node indices. Shape: (batch, max_num_edges).
            mask (torch.Tensor): Edge mask. Shape: (batch, max_num_edges).
            max_num_nodes (int): Maximum number of nodes.

        Returns:
            h_nodes (torch.Tensor): Node embeddings. Shape: (batch, max_num_nodes, node_dim).
            count_edge (torch.Tensor): Edge counts per node. Shape: (batch, max_num_nodes, node_dim).
        """
        batch_size = h_edges.shape[0]
        num_latents = h_edges.shape[2]

        h_nodes = torch.zeros(batch_size, max_num_nodes,num_latents).to(h_edges.device)
        count_edge = torch.zeros_like(h_nodes)
        count = torch.broadcast_to(edge_mask.unsqueeze(-1), h_edges.shape).float()

        idx = indices.unsqueeze(-1).expand(-1, -1, num_latents)
        h_nodes = h_nodes.scatter_add_(1, idx, h_edges)
        count_edge = count_edge.scatter_add_(1, idx, count)
        return h_nodes, count_edge

    def gather_to_edges(self, h_nodes, edge_index, edge_mask, edge_fc_layer):
        """
        Gather node embeddings to edges.

        Args:
            h_nodes (torch.Tensor): Node embeddings. Shape: (batch, max_num_nodes, node_dim).
            edge_index (torch.Tensor): Edge indices. Shape: (batch, max_num_edges, 2).
            mask (torch.Tensor): Edge mask. Shape: (batch, max_num_edges).
            edge_fc_layer (torch.nn.Module): Edge fc layer.

        Returns:
            h_edges (torch.Tensor): edge embeddings. Shape: (batch, max_num_edges, node_dim).
        """
        h_edges_1 = torch.gather(h_nodes, 1, edge_index[:, :, 0].unsqueeze(-1).expand(-1, -1, h_nodes.size(-1)))
        h_edges_2 = torch.gather(h_nodes, 1, edge_index[:, :, 1].unsqueeze(-1).expand(-1, -1, h_nodes.size(-1)))
        h_edges_12 = torch.cat([h_edges_1,h_edges_2],dim=-1)
        # h_edges_21 = torch.cat([h_edges_2,h_edges_1],dim=-1)
        # h_edges = (edge_fc_layer(h_edges_12) + edge_fc_layer(h_edges_21)) / 2
        h_edges = edge_fc_layer(h_edges_12.to(torch.float32))

        mask = torch.broadcast_to(edge_mask.unsqueeze(-1), h_edges.shape)
        h_edges = torch.where(mask, h_edges, torch.zeros_like(h_edges))
        return h_edges

    def scatter_to_nodes(self, h_edges, edge_index, edge_mask, max_num_nodes):
        """
        Scatter edge embeddings to nodes.

        Args:
            h_edges (tuple of torch.Tensor): edge embedding, m12, m21. Shape: (batch, max_num_edges, node_dim).
            edge_index (torch.Tensor): Edge indices. Shape: (batch, max_num_edges, 2).
            mask (torch.Tensor): Edge mask. Shape: (batch, max_num_edges).
            max_num_nodes (int): Maximum number of nodes.

        Returns:
            h_nodes (torch.Tensor): Node embeddings. Shape: (batch, max_num_nodes, node_dim).
        """
        h_nodes_1, count_1 = self.scatter_count(h_edges, edge_index[:, :,0], edge_mask, max_num_nodes)
        h_nodes_2, count_2 = self.scatter_count(h_edges, edge_index[:, :,1], edge_mask, max_num_nodes)
        h_nodes = (h_nodes_1 + h_nodes_2) / (count_1 + count_2 + self.EPSILON)
        return h_nodes

    def cluster_avg(self, index, node_feature):

        node_layer_index = node_feature[:,:,index].to(torch.int64)
        _, inverse_indices = torch.unique(node_layer_index, return_inverse=True, sorted=True)

        cluster_avg = torch.zeros_like(node_feature)
        expanded_indices = inverse_indices.unsqueeze(-1).expand(-1, -1, node_feature.shape[-1])
        cluster_avg.scatter_add_(1, expanded_indices, node_feature)

        layer_count = torch.zeros_like(cluster_avg)
        layer_count.scatter_add_(1, expanded_indices, torch.ones_like(node_feature))
        cluster_avg.div_(layer_count + self.EPSILON)
        # print(node_layer_index)
        h_expand = torch.gather(cluster_avg, 1, node_layer_index.unsqueeze(-1).expand(-1, -1, node_feature.shape[-1]))

        return h_expand

    @staticmethod
    def batch_data(x):
        numerical, node_feature, edge_feature, edge_index, mask, stage = zip(*x)
        numerical = torch.stack(numerical)
        node_feature = torch.stack(node_feature)
        edge_feature = torch.stack(edge_feature)
        edge_index = torch.stack(edge_index)
        mask = torch.stack(mask)
        stage = torch.stack(stage)

        return numerical, node_feature, edge_feature, edge_index, mask, stage

    @staticmethod
    def mean_features(h, mask=None):
        if mask is not None:
            mean_h = (h * mask.unsqueeze(-1).float()).sum(dim=1) / mask.float().sum(dim=1, keepdim=True)
        else:
            mean_h = (h).mean(dim=1)
        return mean_h

    def forward(self, x):
        numerical, node_feature, edge_feature, edge_index, edge_mask, stage = self.batch_data(x)

        h_numerical_features = self.numerical_feature_encoder(numerical.to(torch.float32))
        h_nodes = self.node_encoder(node_feature.to(torch.float32))
        h_edges = self.edge_encoder(edge_feature.to(torch.float32))

        for edge_fc_layer_node in self.edge_fc_layers_node:
            h_edges_gnn = self.gather_to_edges(h_nodes,edge_index,edge_mask,edge_fc_layer_node)
            h_nodes_new = self.scatter_to_nodes(h_edges_gnn, edge_index,edge_mask,node_feature.shape[1])
            h_nodes = h_nodes + h_nodes_new

        for edge_fc_layer_edge in self.edge_fc_layers_edge:
            h_nodes_gnn = self.scatter_to_nodes(h_edges, edge_index,edge_mask,node_feature.shape[1])
            h_edges_new = self.gather_to_edges(h_nodes_gnn, edge_index,edge_mask, edge_fc_layer_edge)
            h_edges = h_edges + h_edges_new

        h_nodes_cluster_avg = self.cluster_avg(index=-2, node_feature=node_feature)
        h_nodes_cluster_avg = torch.zeros_like(h_nodes_cluster_avg)
        h_nodes_all = torch.cat([h_nodes,h_nodes_cluster_avg], dim = -1)
        h_nodes2edges = self.gather_to_edges(h_nodes_all, edge_index, edge_mask, self.edge_fc)
        h_mean = self.mean_features(torch.cat([h_edges, h_nodes2edges], dim=-1), edge_mask)

        h_source = h_nodes[node_feature[:,:,-1] == 1].unsqueeze(1)
        h_source = h_source.expand(-1, h_edges.shape[1], -1)
        h_source = torch.zeros_like(h_source)

        state_value = torch.cat([h_numerical_features, h_mean],dim=-1)
        state_policy = torch.cat([h_edges, h_nodes2edges, h_source],dim=-1)

        return state_policy, state_value, edge_mask, stage

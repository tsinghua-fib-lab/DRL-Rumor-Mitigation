import torch
import torch.nn as nn
from news.models.state_encoder import RLStateEncoder, GNN1StateEncoder, GNN2StateEncoder, GNN3StateEncoder
from news.models.policy import Policy
from news.models.value import Value


def create_RL_model(cfg, agent):
    """Create policy and value network from a config file.
    Args:
        cfg: A config object.
        agent: An agent.
    Returns:
        A tuple containing the policy network and the value network.
    """
    shared_net = RLStateEncoder(cfg.state_encoder_specs, agent)
    policy_net = Policy(cfg.policy_specs, agent, shared_net)
    value_net = Value(cfg.value_specs, agent, shared_net)
    return policy_net, value_net

def create_GNN1_model(cfg, agent):
    """Create policy and value network from a config file.
    Args:
        cfg: A config object.
        agent: An agent.
    Returns:
        A tuple containing the policy network and the value network.
    """
    shared_net = GNN1StateEncoder(cfg.state_encoder_specs, agent)
    policy_net = Policy(cfg.policy_specs, agent, shared_net)
    value_net = Value(cfg.value_specs, agent, shared_net)
    return policy_net, value_net


def create_GNN2_model(cfg, agent):
    """Create policy and value network from a config file.
    Args:
        cfg: A config object.
        agent: An agent.
    Returns:
        A tuple containing the policy network and the value network.
    """
    shared_net = GNN2StateEncoder(cfg.state_encoder_specs, agent)
    policy_net = Policy(cfg.policy_specs, agent, shared_net)
    value_net = Value(cfg.value_specs, agent, shared_net)
    return policy_net, value_net


def create_GNN3_model(cfg, agent):
    """Create policy and value network from a config file.
    Args:
        cfg: A config object.
        agent: An agent.
    Returns:
        A tuple containing the policy network and the value network.
    """
    shared_net = GNN3StateEncoder(cfg.state_encoder_specs, agent)
    policy_net = Policy(cfg.policy_specs, agent, shared_net)
    value_net = Value(cfg.value_specs, agent, shared_net)
    return policy_net, value_net


def create_mlp_model(cfg, agent):
    """Create a multi-layer perceptron model.
    Args:
        cfg: A config object.
        agent: An agent.
    Returns:
        A tuple containing the policy network and the value network.
    """
    shared_net = MLPStateEncoder(cfg.state_encoder_specs, agent)
    policy_net = Policy(cfg.policy_specs, agent, shared_net)
    value_net = Value(cfg.value_specs, agent, shared_net)
    return policy_net, value_net


def create_rmlp_model(cfg, agent):
    """Create a multi-layer perceptron model.
    Args:
        cfg: A config object.
        agent: An agent.
    Returns:
        A tuple containing the policy network and the value network.
    """
    shared_net = RMLPStateEncoder(cfg.state_encoder_specs, agent)
    policy_net = Policy(cfg.policy_specs, agent, shared_net)
    value_net = Value(cfg.value_specs, agent, shared_net)
    return policy_net, value_net


class ActorCritic(nn.Module):
    """
    An Actor-Critic network for parsing parameters.

    Args:
        actor_net (nn.Module): actor network.
        value_net (nn.Module): value network.
    """
    def __init__(self, actor_net, value_net):
        super().__init__()
        self.actor_net = actor_net
        self.value_net = value_net

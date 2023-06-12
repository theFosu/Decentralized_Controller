from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBase(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(MLPBase, self).__init__()
        self.l1 = nn.Linear(num_inputs, 120)
        self.l2 = nn.Linear(120, 60)
        self.l3 = nn.Linear(60, num_outputs)

    def forward(self, inputs):
        x = F.relu(self.l1(inputs))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class SINBase(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(SINBase, self).__init__()
        self.r1 = nn.Linear(num_inputs, 120)
        self.l1 = nn.Linear(120, 60)
        self.l2 = nn.Linear(60, 30)
        self.l3 = nn.Linear(30, num_outputs)

    def forward(self, inputs):
        x = F.tanh(self.r1(inputs))
        x = torch.sin(self.l1(x))
        x = torch.sin(self.l2(x))
        x = self.l3(x)
        return x


class ActorUp(nn.Module):
    """a bottom-up module used in bothway message passing that only passes message to its parent"""
    def __init__(self, state_dim, msg_dim, max_children):
        super(ActorUp, self).__init__()
        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, msg_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.normalize(x, dim=-1)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        x = F.normalize(x, dim=-1)
        msg_up = x

        return 0, msg_up


class ActorDownAction(nn.Module):
    """a top-down module used in bothway message passing that passes messages to children and outputs action"""
    # input dim is state dim if only using top down message passing
    # if using bottom up and then top down, it is the node's outgoing message dim
    def __init__(self, self_input_dim, action_dim, msg_dim, max_action, max_children):
        super(ActorDownAction, self).__init__()
        self.max_action = max_action
        self.action_base = SINBase(msg_dim * max_children, action_dim)
        self.msg_base = MLPBase(msg_dim * max_children, msg_dim)

    def forward(self, m):
        action = self.max_action * torch.tanh(self.action_base(m))
        msg_down = self.msg_base(m)
        msg_down = F.normalize(msg_down, dim=-1)
        return action, msg_down


class JointPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, msg_dim, batch_size, max_action, max_children):
        super(JointPolicy, self).__init__()
        self.batch_size = batch_size
        self.full_message = msg_dim * max_children

        self.up_module = ActorUp(state_dim, msg_dim, max_children)
        self.down_module = ActorDownAction(self.full_message, action_dim, msg_dim, max_action, max_children)

    def forward(self, input_data, bu):
        if bu:
            action, message = self.up_module(input_data)
        else:
            action, message = self.down_module(input_data)
        return action, message

import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from fc import FCNet


class NewAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2, activation=nn.ReLU):
        super(NewAttention, self).__init__()
        self.v_proj = FCNet([v_dim, num_hid], activation)
        self.q_proj = FCNet([q_dim, num_hid], activation)
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim], values are img features
        q: [batch, qdim], queries are question embeddings
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, k ,_ = v.size()
        v_proj = self.v_proj(v)  # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1) 
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits

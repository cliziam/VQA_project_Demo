import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import weight_norm
import numpy as np
from transformers import DistilBertModel


class WordEmbedding(nn.Module):
    """Word Embedding

    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """

    def __init__(self, ntoken, emb_dim, dropout):
        super(WordEmbedding, self).__init__()
        self.emb = nn.Embedding(ntoken + 1, emb_dim, padding_idx=ntoken)
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = emb_dim

    def init_embedding(self, np_file):
        weight_init = torch.from_numpy(np.load(np_file))
        assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data[: self.ntoken] = weight_init

    def forward(self, x):
        emb = self.emb(x)
        emb = self.dropout(emb)
        return emb

class BertEmbedding(nn.Module):
    def __init__(self, dropout):
        super(BertEmbedding, self).__init__()
        self.bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        #just in case of freezing the model de-comment this
        
        for param in self.bert_model.parameters(): 
            param.requires_grad = False
        
        self.bert_model.eval()
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
       #just in case of freezing the model de-comment this
      with torch.no_grad():
        emb = self.bert_model(x)["last_hidden_state"]
        emb = self.dropout(emb) 
      return emb


class QuestionEmbedding(nn.Module):
    def __init__(
        self,
        in_dim,
        num_hid,
        nlayers,
        bidirect,
        dropout,
        rnn_type="GRU",
        personalized_init=False,
    ):
        """Module for question embedding"""
        super(QuestionEmbedding, self).__init__()
        assert rnn_type == "LSTM" or rnn_type == "GRU"
        rnn_cls = nn.LSTM if rnn_type == "LSTM" else nn.GRU

        self.rnn = rnn_cls(
            in_dim,
            num_hid,
            nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True,
        )

        if personalized_init:  # personalized initialization of weights
            if rnn_type == "GRU":
                self.init_gru()
            elif rnn_type == "LSTM":
                self.init_lstm()

        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)
        self.out_size = num_hid * 2 if bidirect else num_hid

    def init_gru(self, gain=1):
        self.rnn.reset_parameters()

        # orthogonal initialization of recurrent weights
        for name, p in self.rnn.named_parameters():
            if "weight_hh" in name:
                for i in range(0, p.size(0), self.rnn.hidden_size):
                    nn.init.orthogonal_(p[i : i + self.rnn.hidden_size], gain=gain)

    def init_lstm(self, gain=1):
        self.init_gru()

        for name, p in self.rnn.named_parameters():
            if "weight_ih" in name:
                for i in range(0, p.size(0), self.rnn.hidden_size):
                    nn.init.xavier_uniform_(p[i : i + self.rnn.hidden_size], gain=gain)
            elif "bias_ih" in name:
                # positive forget gate bias (Jozefowicz et al., 2015)
                p.data.fill_(0)
                n = p.size(0)
                p.data[(n // 4) : (n // 2)].fill_(1.0)
            elif "bias_hh" in name:
                p.data.fill_(0)
                n = p.size(0)
                p.data[(n // 4) : (n // 2)].fill_(1.0)

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        if self.rnn_type == "LSTM":
            return (
                Variable(weight.new(*hid_shape).zero_()),
                Variable(weight.new(*hid_shape).zero_()),
            )
        else:
            return Variable(weight.new(*hid_shape).zero_())

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, _ = self.rnn(x, hidden)

        if self.ndirections == 1:
            return output[:, -1]

        forward_ = output[:, -1, : self.num_hid]
        backward = output[:, 0, self.num_hid :]
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, _ = self.rnn(x, hidden)
        return output


class TemporalBlock(nn.Module):
    """code adapted from https://www.kaggle.com/code/ceshine/pytorch-temporal-convolutional-networks/script"""

    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv2d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding="same",
                dilation=dilation,
            )
        )
        self.pad = torch.nn.ZeroPad2d((padding, 0, 0, 0))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = weight_norm(
            nn.Conv2d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding="same",
                dilation=dilation,
            )
        )
        self.net = nn.Sequential(
            self.pad,
            self.conv1,
            self.relu,
            self.dropout,
            self.pad,
            self.conv2,
            self.relu,
            self.dropout,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x.unsqueeze(2)).squeeze(2)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(
        self, num_inputs, num_channels, out_size, kernel_size=(1, 3), dropout=0.2
    ):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=0,  # (kernel_size[0] - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.temporal_net = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(
            300 * num_channels[-1], out_size
        )  # 300 is word embedding size
        self.out_size = out_size

    def forward(self, x):
        x = self.temporal_net(x)
        x = x.reshape(x.size(0), -1)
        return self.output(self.dropout(x))

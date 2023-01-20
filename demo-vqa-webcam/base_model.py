import torch
import torch.nn as nn
from attention import NewAttention
from language_model import (
    WordEmbedding,
    QuestionEmbedding,
    TemporalConvNet,
    BertEmbedding,
)
from classifier import SimpleClassifier
from fc import FCNet


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, q,  attention_output=False):
        """Forward

        v: [batch, num_objs, obj_dim], visual features
        b: [batch, num_objs, b_dim], spatial features
        q: [batch_size, seq_length], tokenized question

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim]
        att = self.v_att(v, q_emb)
        # use att weights to compute attention output
        v_emb = (att * v).sum(1)  # [batch, v_dim], values are img features

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)

        if attention_output:
            return logits, att
        return logits


def build_baseline0_newatt(
    dataset,
    num_hid,
    bidirectional=False,
    emb_dim=300,
    w_emb_type="baseline",
    rnn_type="GRU",
    activation=nn.ReLU,
    rnn_init=False,
    relu_init=False,
    var_analysis=False,
):

    if w_emb_type == "BERT":
        w_emb = BertEmbedding(0.0)
    else:
        w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim, 0.0)

    if rnn_type == "TCN":
        q_emb = TemporalConvNet(14, [14] * 2, num_hid, kernel_size=(3, 300))
    else:
        q_emb = QuestionEmbedding(
            emb_dim,
            num_hid,
            1,
            bidirectional,
            0.0,
            rnn_type=rnn_type,
            personalized_init=rnn_init,
        )

    num_hid = num_hid * 2 if bidirectional else num_hid  # to double number of params
    v_att = NewAttention(dataset.v_dim, q_emb.out_size, num_hid, activation=activation)
    q_net = FCNet(
        [q_emb.out_size, num_hid],
        activation,
        relu_init=relu_init,
        var_analysis=var_analysis,
        name="q_net",
    )
    v_net = FCNet(
        [dataset.v_dim, num_hid],
        activation,
        relu_init=relu_init,
        var_analysis=var_analysis,
        name="v_net",
    )
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5, activation
    )
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)

import math
import IPython
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import pickle
import gzip
import numpy as np
import time

class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, residual=False):
        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(adj, support)
        if residual:
            output += support
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GraphEncoder(Module):
    def __init__(self, device, entity, emb_size, kg, embeddings=None, fix_emb=True, seq='rnn', gcn=True, hidden_size=100, layers=2, rnn_layer=1, gat=False):
        super(GraphEncoder, self).__init__()
        self.embedding = nn.Embedding(entity, emb_size, padding_idx=entity-1)
        if embeddings is not None:
            print("pre-trained embeddings")
            self.embedding = self.embedding.from_pretrained(embeddings, freeze=fix_emb)
        self.layers = layers
        self.user_num = len(kg.G['user'])
        self.item_num = len(kg.G['item'])
        self.PADDING_ID = entity-1
        self.device = device
        self.seq = seq
        self.gcn = gcn
        self.graph_emb = None
        n_heads = 4

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        if self.seq == 'rnn':
            self.rnn = nn.GRU(hidden_size, hidden_size, rnn_layer, batch_first=True)
        elif self.seq == 'transformer':
            self.transformer = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=400), num_layers=rnn_layer)

        self.pos_gate = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Sigmoid())
        self.neg_gate = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Sigmoid())

        self.sign_embedding = nn.Embedding(3, hidden_size)
         # self.transformer = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=400), num_layers=rnn_layer)
        if self.gcn:
            indim, outdim = emb_size, hidden_size
            self.gnns = nn.ModuleList()
            for l in range(layers):
                if not gat:
                    self.gnns.append(GCNConv(indim, outdim))
                else:
                    self.gnns.append(GATConv(indim, outdim // n_heads, heads=n_heads))
                indim = outdim
        else:
            self.fc2 = nn.Linear(emb_size, hidden_size)

    def graph_prop(self, adj):
        input_state = self.embedding.weight
        adj = adj.to(self.device)
        for gnn in self.gnns:
            output_state = F.relu(gnn(input_state, adj))
            input_state = output_state
        return output_state

    def forward(self, b_state):
        """
        :param b_state [N]
        :return: [N x L x d]
        """
        batch_output = []
        for s in b_state:
            neighbors = s['neighbors'].to(self.device)
            input_state = self.embedding.weight
            if self.gcn:
                graph_emb = self.graph_prop(adj)
                batch_output.append(graph_emb[neighbors])
            else:
                output_state = self.fc2(input_state[neighbors])
                batch_output.append(output_state)
        
        seq_embeddings = []
        rej_item_embeds = []
        rej_embeddings = []

        # cand_embeddings = []
        for s, o in zip(b_state, batch_output):
            seq_embeddings.append(o[:len(s['cur_node']),:][None,:]) 
            seq_embeddings.append(o[len(s['cur_node']) : len(s['cur_node']) + s['cand_num'],:][None,:].mean(1, keepdim=True))
            
            if len(s['rej_attrs']):
                rej_embeddings.append(o[len(s['cur_node']) + s['cand_num']:len(s['cur_node']) + s['cand_num'] +  len(s['rej_attrs']), :][None, :])
            
            if len(s['rej_items']):
                rej_embeddings.append(o[-len(s['rej_items']):, :][None, :].mean(1, keepdim=True))

        if len(batch_output) > 1:
            seq_embeddings = self.padding_seq(seq_embeddings)
        
        seq_embeddings = torch.cat(seq_embeddings, dim=1)  # [N x L x d]
        if s['rej_num']:
            rej_embeddings = torch.cat(rej_embeddings, dim=1)
        else:
            rej_embeddings = torch.zeros_like(seq_embeddings)

        if self.seq == 'rnn':
            _, h = self.rnn(seq_embeddings)
            seq_embeddings = h.permute(1,0,2) #[N*1*D]

        elif self.seq == 'transformer':
            # TransGate
            pos_sign = self.sign_embedding.weight[0].repeat(1, seq_embeddings.shape[1], 1)
            neg_sign = self.sign_embedding.weight[1].repeat(1, rej_embeddings.shape[1], 1)
            seq_embeddings *= math.sqrt(seq_embeddings.shape[2])
            rej_embeddings *= math.sqrt(rej_embeddings.shape[2])
            all_embeddings = torch.cat((seq_embeddings + pos_sign, rej_embeddings + neg_sign), 1)
            all_embeddings = self.transformer(all_embeddings)

            seq_embeddings = torch.mean(all_embeddings[:, :seq_embeddings.shape[1]], dim=1, keepdim=True)
            rej_embeddings = torch.mean(all_embeddings[:, seq_embeddings.shape[1]:], dim=1, keepdim=True)
            
            seq_embeddings = self.pos_gate(rej_embeddings) * seq_embeddings
            rej_embeddings = self.neg_gate(seq_embeddings) * rej_embeddings

            out_embeddings = seq_embeddings - rej_embeddings
            out_embeddings = F.relu(self.fc1(out_embeddings))

        elif self.seq == 'mean':
            seq_embeddings = torch.mean(seq_embeddings, dim=1, keepdim=True)

        elif self.seq == 'neg_gates':
            seq_embeddings = torch.mean(seq_embeddings, dim=1, keepdim=True)
            if rej_item_embeds.nelement() != 0:
                acc_attr_rep = seq_embeddings.squeeze().repeat(1, rej_item_embeds.size(1), 1)
                gate_input = torch.cat((acc_attr_rep, rej_item_embeds, rej_item_embeds * acc_attr_rep), 2)
                gates = F.sigmoid(self.item_gate(gate_input))
                rej_item_embeds = torch.mean(gates * rej_item_embeds, dim=1, keepdim=True)
                seq_embeddings = seq_embeddings - rej_item_embeds

        elif self.seq == 'linear':
            seq_embeddings = torch.mean(seq_embeddings, dim=1, keepdim=True)
            rej_embeddings = torch.mean(rej_embeddings, dim=1, keepdim=True)
            out_embeddings = seq_embeddings - rej_embeddings
            out_embeddings = F.relu(self.fc1(out_embeddings))
            
        return out_embeddings, seq_embeddings, rej_embeddings
    
    def padding_seq(self, seq):
        padding_size = max([len(x[0]) for x in seq])
        padded_seq = []
        for s in seq:
            cur_size = len(s[0])
            emb_size = len(s[0][0])
            new_s = torch.zeros((padding_size, emb_size)).to(self.device)
            new_s[:cur_size,:] = s[0]
            padded_seq.append(new_s[None,:])
        return padded_seq

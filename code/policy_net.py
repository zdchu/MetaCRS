import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import IPython



class NegRanker(nn.Module):
    def __init__(self, hidden_size, state_size, action_size, act='tanh'):
        super(NegRanker, self).__init__()
        self.act = 'Sigmoid'
        self.acc_linear = nn.Linear(hidden_size, action_size)
        self.rej_linear = nn.Linear(hidden_size, action_size)
    
    def forward(self, x, acc_embed, rej_embed):
        '''
        state: state embedding
        x: attribute/item embedding
        '''

        acc_embed = self.acc_linear(F.relu(acc_embed))
        rej_embed = self.rej_linear(F.relu(rej_embed))

        pos_score = torch.mm(acc_embed.squeeze(0), x.squeeze(0).T)
        neg_score = torch.mm(rej_embed.squeeze(0), x.squeeze(0).T)

        return pos_score - neg_score


class DQN_rnn(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=100, conditional=False, conditional_type='sigmoid', reward_size=10, dual_rnn=True):
        super(DQN_rnn, self).__init__()
        self.linear1 = nn.Linear(state_size + action_size + reward_size, hidden_size)
        self.linear2 = nn.Linear(action_size * 2, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.user_rnn = nn.GRU(hidden_size, action_size, num_layers=1, batch_first=True)
        self.dual_rnn = dual_rnn

        self.reward_layer = nn.Linear(1, reward_size)
        self.item_ranker = NegRanker(hidden_size, state_size, action_size, act='sigmoid')
        
    def forward(self, x, y, choose_action=True, user_emb=None, vars=None, prev_reward=None, prev_action=None, 
                            hidden=None, local_hidden=None, conditional_emb=None):
        """
        :param x: encode history [N*L*D]; y: action embedding [N*K*D]
        :return: v: action score [N*K]
        """
        state = torch.cat((x.squeeze(), prev_reward, prev_action.squeeze()))
        state = F.relu(self.linear1(state))

        state_emb, hidden = self.user_rnn(state.view(1, 1, -1), hidden)
        if self.dual_rnn:
            local_state_emb, local_hidden = self.local_rnn(state.view(1, 1, -1), local_hidden)
            z = F.sigmoid(self.comp_gate(local_hidden))
            state_emb = z * local_state_emb + (1-z) * state_emb

        state_emb = state_emb.repeat(1, y.size(1), 1)
        state_cat_action = torch.cat((state_emb, y), dim=2)
        advantage = self.linear3(F.relu(self.linear2(state_cat_action))).squeeze(dim=2) #[N*K]
        return advantage, hidden, local_hidden

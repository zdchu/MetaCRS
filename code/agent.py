from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from torch import optim
import IPython
from torch.autograd import Variable
from torch.distributions import Categorical
from policy_net import DQN_rnn

class Decoupled_Agent(nn.Module):
    def __init__(self, device, state_size, action_size, hidden_size, encoder, conditional=False, conditional_type='sigmoid', init_lr=1e-3, meta_method='raw', reward_size=10,  num_users=100, dual_rnn=False):
        super(Decoupled_Agent, self).__init__()
        self.device = device
        self.encoder = encoder
        self.policy_net = DQN_rnn(state_size, action_size, hidden_size, conditional, conditional_type, dual_rnn=dual_rnn).to(device)
        self.exp_policy = DQN_rnn(state_size, action_size, hidden_size, conditional, conditional_type, dual_rnn=False).to(device)


    def explore_action(self, state, cand_feature, cand_item, action_space, is_test=False, is_last_turn=False, user_emb=None, vars=None, user_attr=None, 
                                prev_info=None, hidden=None, local_hidden=None, idx=1):
        state_emb, acc_embed, rej_embed = self.encoder([state])

        cand_feature = torch.LongTensor([cand_feature]).to(self.device)
        cand_item = torch.LongTensor([cand_item]).to(self.device)
        
        cand_feat_emb = self.encoder.embedding(cand_feature)
        cand_item_emb = self.encoder.embedding(cand_item)
        cand_item_emb = cand_item_emb[:, :10]
        cand = torch.cat((cand_feature, cand_item[:, :10]), 1)
        cand_emb = torch.cat((cand_feat_emb, cand_item_emb), 1)

        prev_reward = torch.FloatTensor([prev_info['prev_reward']]).to(self.device)
        reward_emb = self.exp_policy.reward_layer(prev_reward)

        prev_action = torch.LongTensor([prev_info['prev_action']]).to(self.device)
        prev_action = self.encoder.embedding(prev_action)
        
        action_value, hidden, _ = self.exp_policy(state_emb, cand_emb, user_emb=user_emb, vars=vars, 
                       prev_reward=reward_emb, prev_action=prev_action, hidden=hidden, local_hidden=None)

        prob = Categorical(action_value.softmax(1))
        if is_test:
            if (len(action_space[1]) <= 10 or is_last_turn):
                return torch.tensor(action_space[1][0], device=self.device, dtype=torch.long), action_space[1], None, None, hidden, local_hidden
            action = cand[0][action_value.argmax().item()]
            sorted_actions = cand[0][action_value.sort(1, True)[1].tolist()]
            return action, sorted_actions.tolist(), prob.log_prob(action_value.argmax()), prob.probs.detach(), hidden, local_hidden
        else:
            action = prob.sample()
            log_prob = prob.log_prob(action)
            action = cand[0][action]

            sorted_actions = cand[0][action_value.sort(1, True)[1].tolist()]
            return action, sorted_actions.tolist(), log_prob, prob.probs.detach(), hidden, local_hidden
        

    def select_action(self, state, cand_feature, cand_item, action_space, is_test=False, is_last_turn=False, user_emb=None, vars=None, user_attr=None, prev_info=None, hidden=None, local_hidden=None):
        state_emb, acc_embed, rej_embed = self.encoder([state])
        cand_feature = torch.LongTensor([cand_feature]).to(self.device)
        cand_item = torch.LongTensor([cand_item]).to(self.device)
        
        cand_feat_emb = self.encoder.embedding(cand_feature)
        cand_item_emb = self.encoder.embedding(cand_item)

        raw_item_score = self.policy_net.item_ranker(cand_item_emb, acc_embed, rej_embed)
        item_score = F.log_softmax(raw_item_score).squeeze(0)
        cand_item_emb = cand_item_emb[:, item_score.argsort()[-10:]]
        cand = torch.cat((cand_feature, cand_item[:, item_score.argsort()[-10:]]), 1)
        cand_emb = torch.cat((cand_feat_emb, cand_item_emb), 1)

        prev_reward = torch.FloatTensor([prev_info['prev_reward']]).to(self.device)
        reward_emb = self.policy_net.reward_layer(prev_reward)

        prev_action = torch.LongTensor([prev_info['prev_action']]).to(self.device)
        prev_action = self.encoder.embedding(prev_action)

        action_value, hidden, local_hidden = self.policy_net(state_emb, cand_emb, user_emb=user_emb, vars=vars, 
                                    prev_reward=reward_emb, prev_action=prev_action, hidden=hidden, local_hidden=local_hidden)

        prob = Categorical(action_value.softmax(1))
        if is_test:
            if (len(action_space[1]) <= 10 or is_last_turn):
                return torch.tensor(action_space[1][0], device=self.device, dtype=torch.long), action_space[1], None, None, hidden, local_hidden, None, None
            action = cand[0][action_value.argmax().item()]
            sorted_actions = cand[0][action_value.sort(1, True)[1].tolist()]
            return action, sorted_actions.tolist(), prob.log_prob(action_value.argmax()), prob.probs.detach(), hidden, local_hidden, None, None
        else:
            action = prob.sample()
            log_prob = prob.log_prob(action)
            action = cand[0][action]

            sorted_actions = cand[0][action_value.sort(1, True)[1].tolist()]
            return action, sorted_actions.tolist(), log_prob, prob.probs.detach(), hidden, local_hidden, item_score, cand_item
import imp
from re import S
from telnetlib import IP
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from torch import optim
from torch.autograd import Variable
from episode import run_episode_rnn, run_episode_explore
from collections import OrderedDict, Counter
import IPython


class MetaRL_rnn(nn.Module):
    def __init__(self, args, env, agent):
        super(MetaRL_rnn, self).__init__()
        self.lr_meta = args.lr_meta
        self.lr_phi = args.lr_phi
        self.inner_steps = args.inner_steps
        self.env = env
        self.args = args
        self.agent = agent
        self.meta_opt = optim.Adam(self.agent.parameters(), lr=args.lr_meta, weight_decay = args.l2_norm, betas=(0, 0.999))
        self.explore_opt = optim.Adam(list(self.agent.encoder.parameters()) + 
                                list(self.agent.exp_policy.parameters()), lr=1e-3, weight_decay = args.l2_norm)
        self.visited_user = []

        self.jac_good = []
        self.jac_bad = []
        self.jac = []
        self.suc = []

        self.jac_raw = []
        self.max_jac = []
        self.suc_raw = []
        self.suc_ada = []

    def forward(self, spt, qry):
        env = self.env
        args = self.args
        task_num, _, _ = spt.shape
        losses_q = 0
        correct = 0
        qry_size = 0

        avg_turn = 0
        avg_turn_raw = 0

        self.agent.train()
        self.kept_agent = deepcopy(self.agent.state_dict())

        inner_loss = 0
        inner_turn = 0
        inner_reward = 0

        inner_rewards = []
        meta_gradient = OrderedDict()
        explore_loss = 0
        for task_id in range(task_num):
            inner_turn = 0.1
            inner_reward = 0
            reward_list = []

            hidden = None
            fast_agent = deepcopy(self.agent)
            for idx, episode in enumerate(spt[task_id][:args.num_explore]):
                user_id, item_id = episode[0], episode[1]
                rewards, log_prob, _, hidden, _, _, _, _, raw_rewards, _, attr_list, aux_loss = run_episode_explore(args, env, 
                                                                    self.agent, user_id, item_id, is_test=False, hidden=hidden, idx=idx)
                hidden = hidden.detach()
                loss = torch.sum(torch.mul(log_prob, Variable(rewards)).mul(-1)) + aux_loss / len(rewards)
                explore_loss += loss
                
            cur_epi = 0
            for k in range(self.inner_steps):
                grad = OrderedDict()
                loss = 0
                for idx, episode in enumerate(spt[task_id][args.num_explore:]):
                    lr_phi = args.lr_phi
                    cur_epi += 1
                    user_id, item_id = episode[0], episode[1]
                    print('lr phi: ', lr_phi)

                    rewards, log_prob, success, hidden, _, _, _, _, raw_rewards, _, attr_list, item_loss = run_episode_rnn(self.args, env, fast_agent, user_id, item_id, is_test=False, hidden=hidden)
                    hidden = hidden.detach()

                    loss = torch.sum(torch.mul(log_prob, Variable(rewards)).mul(-1))

                    if self.args.item_rank:
                        loss += item_loss
                    
                    inner_loss += loss.item()
                    inner_turn += len(log_prob)
                    inner_reward += np.sum(raw_rewards)
                    
                    fast_agent.zero_grad()
                    params = fast_agent.parameters() if self.args.full_update else fast_agent.policy_net.parameters() 
                    
                    tmp_grad = torch.autograd.grad(loss, params, allow_unused=True)
                    named_params = fast_agent.named_parameters() if self.args.full_update else fast_agent.policy_net.named_parameters() 
                    
                    for i, (n, p) in enumerate(named_params):
                        if n in grad:
                            grad[n] += tmp_grad[i] if tmp_grad[i] is not None else 0
                        else:
                            grad[n] = tmp_grad[i] if tmp_grad[i] is not None else 0

                    if (idx + 1) % args.inner_batch == 0:
                        fast_weight = OrderedDict()
                        params = fast_agent.named_parameters() if self.args.full_update else fast_agent.policy_net.named_parameters() 
                        for n, p in params:
                            # fast_weight[n] = p - self.lr_phi * grad[n] / args.inner_batch
                            fast_weight[n] = p - lr_phi * grad[n] / args.inner_batch

                        if self.args.full_update:
                            fast_agent.load_state_dict(fast_weight)
                        else:
                            fast_agent.policy_net.load_state_dict(fast_weight)
                        grad = OrderedDict()
                    
                    reward_list.append(inner_reward / inner_turn)
                inner_rewards.append(reward_list)

            for episode in qry[task_id]:
                user_id, item_id = episode[0], episode[1]
                rewards, log_prob_q, success, _, _, _, _, _, _, _, attr_list, item_loss = run_episode_rnn(self.args, env, fast_agent, user_id, item_id, is_test=False, hidden=hidden)
                loss_q = torch.sum(torch.mul(log_prob_q, Variable(rewards)).mul(-1))

                if self.args.item_rank:
                    loss_q += item_loss
                
                losses_q += loss_q
                params =  list(fast_agent.policy_net.parameters())
                params += [p for p in list(fast_agent.encoder.parameters()) if p.requires_grad]
                grad = torch.autograd.grad(loss_q, params, allow_unused=True)
                
                for i, g in enumerate(grad):
                    if i not in meta_gradient:
                        meta_gradient[i] = g if g is not None else 0
                    else:
                        meta_gradient[i] += g if g is not None else 0
                correct += success
                avg_turn += len(rewards)
                qry_size += len(qry[task_id])
        
        if args.num_explore > 0:
            explore_loss /= task_num * args.num_explore
            self.explore_opt.zero_grad()
            explore_loss.backward(retain_graph=True)
            self.explore_opt.step()
        
        self.meta_opt.zero_grad()
        params =  list(self.agent.policy_net.parameters())
        params += [p for p in list(self.agent.encoder.parameters()) if p.requires_grad]
        for i, p in enumerate(params):
            if type(meta_gradient[i]) != int:
                p.grad = meta_gradient[i].detach() / qry_size
        self.meta_opt.step()
        return correct / float(qry_size), avg_turn / float(qry_size)

    def fintunning(self, spt, qry):
        env = self.env
        args = self.args
        correct = 0
        
        fast_agent = deepcopy(self.agent)
        raw_agent = deepcopy(self.agent)

        avg_turn = 0
        avg_turn_raw = 0
        
        fast_agent.train()
        inner_rewards = []
        train_correct = 0

        hidden = None    
        inner_turn = 0
        inner_reward = 0
        reward_list = []
        for k in range(self.inner_steps):
            for idx, episode in enumerate(spt[:args.num_explore]):
                user_id, item_id = episode[0], episode[1]
                rewards, log_prob, _, hidden, _, _, _, _, raw_rewards, _, attr_list, _ = run_episode_explore(args, env, self.agent, user_id, item_id, is_test=False, hidden=hidden, idx=idx)
                hidden = hidden.detach()

        total_epi = self.inner_steps * len(spt[args.num_explore:])
        cur_epi = 0
        for k in range(self.inner_steps):
            grad = OrderedDict()
            for idx, episode in enumerate(spt[args.num_explore:]):
                user_id, item_id = episode[0], episode[1]
                lr_phi = args.lr_phi
                cur_epi += 1

                rewards, log_prob, success, hidden, _, _, _, _, raw_rewards, _, attr_list, item_loss = run_episode_rnn(args, env, fast_agent, user_id, item_id, is_test=False, hidden=hidden)
                hidden = hidden.detach()

                loss = torch.sum(torch.mul(log_prob, Variable(rewards)).mul(-1))

                if self.args.item_rank:
                    loss += item_loss

                inner_turn += len(log_prob)
                inner_reward += np.sum(raw_rewards)
                reward_list.append(inner_reward / inner_turn)

                params = fast_agent.parameters() if self.args.full_update else fast_agent.policy_net.parameters() 
                tmp_grad = torch.autograd.grad(loss, params, allow_unused=True)
                named_params = self.agent.named_parameters() if self.args.full_update else self.agent.policy_net.named_parameters() 
                for i, (n, p) in enumerate(named_params):
                    if n in grad:
                        grad[n] += tmp_grad[i] if tmp_grad[i] is not None else 0
                    else:
                        grad[n] = tmp_grad[i] if tmp_grad[i] is not None else 0
                
                if (idx + 1) % args.inner_batch == 0:
                    fast_weight = OrderedDict()
                    params = fast_agent.named_parameters() if self.args.full_update else fast_agent.policy_net.named_parameters() 
                    for n, p in params:
                        fast_weight[n] = p - lr_phi * grad[n] / args.inner_batch

                    if self.args.full_update:
                        fast_agent.load_state_dict(fast_weight)
                    else:
                        fast_agent.policy_net.load_state_dict(fast_weight)
                    grad = OrderedDict()
                train_correct += success
            inner_rewards.append(reward_list)
        
        mrr = []
        fast_agent.eval()
        for episode in qry:
            user_id, item_id = episode[0], episode[1]
            rewards, log_prob_q, success, _, _, _, _, rank_idx, _, _, attr_list, _ = run_episode_rnn(args, env, fast_agent, user_id, item_id, is_test=True, hidden=hidden)
            
            correct += success
            mrr.append(rank_idx)
            avg_turn += len(rewards)
        return correct, avg_turn
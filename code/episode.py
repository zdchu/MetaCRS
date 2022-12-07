from itertools import count
import IPython
from torch.autograd import Variable
import torch
import torch.nn.functional as F



def run_episode_rnn(args, env, agent, user_id, item_id, is_test=True, vars=None, init_attr=None, hidden=None):
    state, cand_feature, cand_item, action_space = env.reset(agent.encoder.embedding.weight.data.cpu().detach().numpy(),
                                    user_id=user_id, item_id=item_id, init_attr=init_attr) 
                                    
    reward_list = []
    log_prob_list = Variable(torch.Tensor()).to(args.device)
    is_last_turn = False
    
    action_list = []
    state_list = []
    cand_list = []
    prob_list = []
    
    attr_list = []

    prev_info = {'prev_reward': 0.1,
                'prev_action': env.init_attr + env.user_length + env.item_length}

    local_hidden = None
    item_loss = 0
    for t in count():
        if t ==  (args.max_turn - 1):
            is_last_turn = True
        
        action, sorted_actions, log_prob, prob, next_hidden, next_local_hidden, item_score, item_idx = agent.select_action(state, cand_feature, cand_item, action_space, is_last_turn=is_last_turn,
                                         is_test=is_test, vars=vars, prev_info=prev_info, hidden=hidden, local_hidden=local_hidden)
        next_state, next_cand_feature, next_cand_item, action_space, reward, done, success = env.step(action.item(), sorted_actions, agent.encoder.embedding.weight.data.cpu().detach().numpy())

        reward_list.append(reward)
        action_list.append(action)
        state_list.append(state)
        prob_list.append(prob)

        if item_idx is not None:
            item_idx -= env.user_length
            if env.target_item in item_idx:
                item_loss += -item_score[torch.where(item_idx == env.target_item)[1]][0]

        if action - env.user_length - env.item_length >= 0:
            attr_list.append((action - env.user_length - env.item_length).item())
        else:
            attr_list.append(env.feature_length)

        if not is_test:
            log_prob_list = torch.cat([log_prob_list, log_prob.reshape(1)])

        prev_info = {'prev_reward': reward,
            'prev_action': action}

        state = next_state
        cand_feature = next_cand_feature
        cand_item = next_cand_item
        hidden = next_hidden
        local_hidden = next_local_hidden
        if done:
            if not success:
                item_loss = torch.tensor(0.).to(args.device)
            else:
                item_loss /= len(reward_list)
            break

    rewards = []
    R = 0
    for r in reward_list[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
        
    return torch.Tensor(rewards).to(args.device), log_prob_list,\
                 success, hidden, action_list, state_list, cand_list, env.rank_idx, reward_list, prob_list, attr_list, item_loss

def run_episode_explore(args, env, agent, user_id, item_id, is_test=True, vars=None, init_attr=None, hidden=None, idx=0):
    user_emb = torch.Tensor(env.ui_embeds[:env.user_length]).to(args.device)
    state, cand_feature, cand_item, action_space = env.reset(agent.encoder.embedding.weight.data.cpu().detach().numpy(),
                                    user_id=user_id, item_id=item_id, init_attr=init_attr) 

    reward_list = []
    log_prob_list = Variable(torch.Tensor()).to(args.device)
    is_last_turn = False
    
    action_list = []
    state_list = []
    cand_list = []
    prob_list = []
    
    attr_list = []

    prev_info = {'prev_reward': 0.1,
                'prev_action': env.init_attr + env.user_length + env.item_length}
    init_emb = agent.encoder.embedding(torch.LongTensor([prev_info['prev_action']]).to(args.device))

    local_hidden = None
    aux_loss = 0
    
    T = 0.1
    for t in count():
        if t ==  (args.max_turn - 1):
            is_last_turn = True
        
        action, sorted_actions, log_prob, prob, next_hidden, _ = agent.explore_action(state, cand_feature, cand_item, action_space, is_last_turn=is_last_turn,
                                          is_test=is_test, vars=vars, prev_info=prev_info, hidden=hidden, local_hidden=local_hidden, idx=idx)

        next_state, next_cand_feature, next_cand_item, action_space, reward, done, success = env.step(action.item(), sorted_actions, agent.encoder.embedding.weight.data.cpu().detach().numpy())

        
        if reward < 0:
            reward = -0.05
        if reward > 0:
            reward = 0.2
        
        reward += -torch.log_softmax(torch.mm(user_emb, init_emb.T) / T, 0)[user_id] if hidden is None else -torch.log_softmax(torch.mm(user_emb, hidden.squeeze(1).T) / T, 0)[user_id]
        reward += torch.log_softmax(torch.mm(user_emb, next_hidden.squeeze(1).T) / T, 0)[user_id]
        reward = torch.clamp(reward, min=-2, max=3)
        reward = reward.item()
        aux_loss += -torch.log_softmax(torch.mm(user_emb, next_hidden.squeeze(1).T), 0)[user_id].squeeze()

        reward_list.append(reward)
        action_list.append(action)
        state_list.append(state)
        prob_list.append(prob)

        if action - env.user_length - env.item_length >= 0:
            attr_list.append((action - env.user_length - env.item_length).item())
        else:
            attr_list.append(env.feature_length)

        if not is_test:
            log_prob_list = torch.cat([log_prob_list, log_prob.reshape(1)])

        prev_info = {'prev_reward': reward,
            'prev_action': action}

        state = next_state
        cand_feature = next_cand_feature
        cand_item = next_cand_item
        hidden = next_hidden
        if done:
            break

    rewards = []
    R = 0
    for r in reward_list[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    return torch.Tensor(rewards).to(args.device), log_prob_list,\
                 success, hidden, action_list, state_list, cand_list, env.rank_idx, reward_list, prob_list, attr_list, aux_loss
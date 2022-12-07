
import json
import numpy as np
import os
import random
import scipy.sparse as sp

from utils import *
from torch import nn
import IPython
from itertools import chain

from collections import Counter
class BinaryRecommendEnv(object):
    def __init__(self, args, kg, dataset, data_name, embed, seed=1, max_turn=15, cand_num=10, cand_item_num=10, attr_num=20, mode='train', ask_num=1, entropy_way='weight entropy', fm_epoch=0, with_user=False):
        self.data_name = data_name
        self.mode = mode
        self.with_user = with_user
        self.seed = seed
        self.max_turn = max_turn    #MAX_TURN
        self.attr_state_num = attr_num
        self.kg = kg
        self.dataset = dataset
        self.feature_length = getattr(self.dataset, 'feature').value_len
        self.user_length = getattr(self.dataset, 'user').value_len
        self.item_length = getattr(self.dataset, 'item').value_len

        self.args = args

        # action parameters
        self.ask_num = ask_num
        self.rec_num = 10
        self.random_sample_feature = False
        self.random_sample_item = False
        if cand_num == 0:
            self.cand_num = 10
            self.random_sample_feature = True
        else:
            self.cand_num = cand_num
        if cand_item_num == 0:
            self.cand_item_num = 10
            self.random_sample_item = True
        else:
            self.cand_item_num = cand_item_num
        #  entropy  or weight entropy
        self.ent_way = entropy_way

        # user's profile
        self.reachable_feature = []   # user reachable feature
        self.user_acc_feature = []  # user accepted feature which asked by agent
        self.user_rej_feature = []  # user rejected feature which asked by agent
        self.cand_items = []   # candidate items
        self.item_feature_pair = {}
        self.cand_item_score = []

        #user_id  item_id   cur_step   cur_node_set
        self.user_id = None
        self.target_item = None
        self.cur_conver_step = 0        #  the number of conversation in current step
        self.cur_node_set = []     # maybe a node or a node set  /   normally save feature node
        # state veactor
        self.user_list = None
        self.conver_his = []    #conversation_history
        self.attr_ent = []  # attribute entropy

        self.user_weight_dict = dict()
        self.user_items_dict = dict()

        embeds = load_embed(data_name, embed, epoch=fm_epoch)
        if embeds:
            self.ui_embeds =embeds['ui_emb']
            self.feature_emb = embeds['feature_emb']
        else:
            self.ui_embeds = nn.Embedding(self.user_length+self.item_length, 64).weight.data.numpy()
            self.feature_emb = nn.Embedding(self.feature_length, 64).weight.data.numpy()
        # self.feature_length = self.feature_emb.shape[0]-1

        self.action_space = 2
        self.reward_dict = {
            'ask_suc': 0.1,
            'ask_fail': -0.1,
            'rec_suc': 1,
            'rec_fail': -0.1,
            'until_T': -0.3,      # MAX_Turn
            'cand_none': -0.1
        }
        self.history_dict = {
            'ask_suc': 1,
            'ask_fail': -1,
            'rec_scu': 2,
            'rec_fail': -2,
            'until_T': 0
        }
        self.attr_count_dict = dict() 
        with open(DATA_DIR[data_name] + '/user_attr.json', 'r') as f:
            self.user_attr = json.load(f)
        attr_dist = np.bincount(list(chain(*self.user_attr.values())))
        self.attr_dist = attr_dist / np.sum(attr_dist)


    def __load_rl_data__(self, data_name, mode):
        if mode == 'train':
            with open(os.path.join(DATA_DIR[data_name], 'UI_Interaction_data/review_dict_valid.json'), encoding='utf-8') as f:
                print('train_data: load RL valid data')
                mydict = json.load(f)
        elif mode == 'test':
            with open(os.path.join(DATA_DIR[data_name], 'UI_Interaction_data/review_dict_test.json'), encoding='utf-8') as f:
                print('test_data: load RL test data')
                mydict = json.load(f)
        return mydict


    def reset(self, embed=None, user_id=None, item_id=None, interaction=None, users=None, init_attr=None):
        if embed is not None:
            self.ui_embeds = embed[:self.user_length+self.item_length]
            self.feature_emb = embed[self.user_length+self.item_length:]
        self.cur_conver_step = 0   #reset cur_conversation step
        self.cur_node_set = []

        self.suc_items = [] # reset successful items
        self.rej_items = [] # reset rejected items

        if self.mode == 'train' and users is not None:
            self.user_list = set(interaction.user.drop_duplicates().tolist()) if self.user_list is None else self.user_list

            users = set(users) & self.user_list
            self.user_id = np.random.choice(list(users))
            self.target_item = np.random.choice(interaction[interaction.user==self.user_id].item)

        if user_id is not None and item_id is not None:
            self.user_id = user_id
            self.target_item = item_id
        
        print('-----------reset state vector------------')
        print('user_id:{}, target_item:{}'.format(self.user_id, self.target_item))
        self.reachable_feature = []  # user reachable feature in cur_step
        self.user_acc_feature = []  # user accepted feature which asked by agent
        self.user_rej_feature = []  # user rejected feature which asked by agent
        self.cand_items = list(range(self.item_length))

        # init state vector
        self.conver_his = [0] * self.max_turn  # conversation_history
        self.attr_ent = [0] * self.attr_state_num  # attribute entropy

        # initialize dialog by randomly asked a question from ui interaction
        if init_attr is None:
            # user_like_random_fea = random.choice(self.kg.G['item'][self.target_item]['belong_to'])
            feat = list(set(self.kg.G['item'][self.target_item]['belong_to']) & set(self.user_attr[str(self.user_id)]))
            if len(feat) == 0:
                feat = list(self.kg.G['item'][self.target_item]['belong_to'])

            user_like_random_fea = np.random.choice(feat)
            self.init_attr = user_like_random_fea
        else:
            user_like_random_fea = init_attr

        if user_like_random_fea is not None:
            self.user_acc_feature.append(user_like_random_fea) #update user acc_fea
            self.cur_node_set.append(user_like_random_fea)
            
            self._update_cand_items(user_like_random_fea, acc_rej=True)
            self._updata_reachable_feature()  # self.reachable_feature = []
            self.conver_his[self.cur_conver_step] = self.history_dict['ask_suc']
            self.cur_conver_step += 1
        else:
            self._update_cand_items(user_like_random_fea, acc_rej=False)
            self._updata_reachable_feature()
            self.conver_his[self.cur_conver_step] = self.history_dict['ask_fail']
            self.cur_conver_step += 1

        print('=== init user prefer feature: {}'.format(self.cur_node_set))
        self._update_feature_entropy()  #update entropy
        print('reset_reachable_feature num: {}'.format(len(self.reachable_feature)))

        # Sort reachable features according to the entropy of features
        reach_fea_score = self._feature_score()
        max_ind_list = []
        for k in range(self.cand_num):
            max_score = max(reach_fea_score)
            max_ind = reach_fea_score.index(max_score)
            reach_fea_score[max_ind] = 0
            if max_ind in max_ind_list:
                break
            max_ind_list.append(max_ind)

        max_fea_id = [self.reachable_feature[i] for i in max_ind_list]
        [self.reachable_feature.remove(v) for v in max_fea_id]
        [self.reachable_feature.insert(0, v) for v in max_fea_id[::-1]]
        self.rank_idx = 100
        
        cand_feature, cand_item = self._get_cand()
        return self._get_state(), cand_feature, cand_item, self._get_action_space()

    def _get_cand(self):
        if self.random_sample_feature:
            cand_feature = self._map_to_all_id(random.sample(self.reachable_feature, min(len(self.reachable_feature),self.cand_num)),'feature')
        else:
            cand_feature = self._map_to_all_id(self.reachable_feature[:self.cand_num],'feature')
        if self.random_sample_item:
            cand_item =  self._map_to_all_id(random.sample(self.cand_items, min(len(self.cand_items),self.cand_item_num)),'item')
        else:
            cand_item = self._map_to_all_id(self.cand_items[:self.cand_item_num],'item')
        return cand_feature, cand_item
    
    def _get_action_space(self):
        action_space = [self._map_to_all_id(self.reachable_feature,'feature'), self._map_to_all_id(self.cand_items,'item')]
        return action_space

    def _get_state(self):
        if len(self.cand_items) <= 500:
            self_cand_items = self.cand_items
        else:
            self_cand_items = self.cand_items[:40]
        
        cur_node = [x + self.user_length + self.item_length for x in self.cur_node_set] 
        cand_items = [x + self.user_length for x in self_cand_items[:40]]
  
        rej_items = [x + self.user_length for x in self.rej_items]
        rej_attrs = [x + self.user_length + self.item_length for x in self.user_rej_feature]
        neighbors = cur_node + cand_items + rej_attrs + rej_items 
        neighbors = torch.LongTensor(neighbors)
        adj = None
        state = {'cur_node': cur_node,
                 'neighbors': neighbors,
                 'cand_num': len(cand_items),
                 'adj': adj,
                'rej_items': rej_items,
                'rej_attrs': rej_attrs,
                'rej_num': len(rej_attrs) + len(rej_items)}
        return state
    
    def _preprocess_adj(self, i, v, shape):
        adj = sp.coo_matrix((v, i.T), shape=(shape, shape))
        norm_adj = self._normalize_adj(adj + sp.eye(adj.shape[0]))
        
        values = norm_adj.data
        indices = np.vstack((norm_adj.row, norm_adj.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = norm_adj.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))
    
    def _normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        rowsum = np.array(adj.sum(1)) # D
        d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    
    def step(self, action, sorted_actions, embed=None):  
        if embed is not None:
            self.ui_embeds = embed[:self.user_length+self.item_length]
            self.feature_emb = embed[self.user_length+self.item_length:]

        done = 0
        print('---------------step:{}-------------'.format(self.cur_conver_step))
        success = 0
        if self.cur_conver_step == self.max_turn:
            reward = self.reward_dict['until_T']
            self.conver_his[self.cur_conver_step-1] = self.history_dict['until_T']
            print('--> Maximum number of turns reached !')
            done = 1
        elif action >= self.user_length + self.item_length:   #ask feature
            asked_feature = self._map_to_old_id(action)
            print('-->action: ask features {}, max entropy feature {}'.format(asked_feature, self.reachable_feature[:self.cand_num]))
            reward, done, acc_rej = self.get_user_response(asked_feature)
            self._update_cand_items(asked_feature, acc_rej)   #update cand_items
        else:  #recommend items
            
            #===================== rec update=========
            recom_items = []
            for act in sorted_actions:
                if act < self.user_length + self.item_length:
                    recom_items.append(self._map_to_old_id(act))
                    if len(recom_items) == self.rec_num:
                        break
            reward, done, self.rank_idx = self._recommend_update(recom_items)
            #========================================
            if reward > 0:
                success = 1
                print('-->Recommend successfully!')
            else:
                print('-->Recommend fail !')

        self._updata_reachable_feature()  # update user's profile: reachable_feature
        print('reachable_feature num: {}'.format(len(self.reachable_feature)))
        print('cand_item num: {}'.format(len(self.cand_items)))

        self._update_feature_entropy()
        if len(self.reachable_feature) != 0:  # if reachable_feature == 0 :cand_item= 1
            reach_fea_score = self._feature_score()  # compute feature score
            max_ind_list = []
            for k in range(self.cand_num):
                max_score = max(reach_fea_score)
                max_ind = reach_fea_score.index(max_score)

                reach_fea_score[max_ind] = 0
                if max_ind in max_ind_list:
                    break
                max_ind_list.append(max_ind)

            max_fea_id = [self.reachable_feature[i] for i in max_ind_list]
            [self.reachable_feature.remove(v) for v in max_fea_id]
            [self.reachable_feature.insert(0, v) for v in max_fea_id[::-1]]

        self.cur_conver_step += 1
        
        # self.update_cand_item_scores()
        cand_feature, cand_item = self._get_cand()
        return self._get_state(), cand_feature, cand_item, self._get_action_space(), reward, done, success

    def _updata_reachable_feature(self):
        next_reachable_feature = []
        
        for cand in self.cand_items[:500]:
            fea_belong_items = list(self.kg.G['item'][cand]['belong_to']) # A-I
            next_reachable_feature.extend(fea_belong_items)
        
        self.reachable_feature = list(set(next_reachable_feature) - set(self.user_acc_feature) - set(self.user_rej_feature))

    def _feature_score(self):
        reach_fea_score = []
        for feature_id in self.reachable_feature:
            if self.ent_way == 'entropy' or self.ent_way == 'weight_entropy':
                score = self.attr_ent[feature_id]
                reach_fea_score.append(score)
                continue
            feature_embed = self.feature_emb[feature_id]
            score = 0
            prefer_embed = self.feature_emb[self.user_acc_feature, :]  #np.array (x*64)
            unprefer_embed = self.feature_emb[self.user_rej_feature, :] 
            if len(self.user_acc_feature) > 0:
                prefer_embed = np.mean(prefer_embed, 0)
                score += np.inner(prefer_embed, feature_embed)
            if len(self.user_rej_feature) > 0:
                unprefer_embed = np.mean(unprefer_embed, 0)
                score -= np.inner(unprefer_embed, feature_embed)
            reach_fea_score.append(score)
        return reach_fea_score

    def _item_score(self):
        cand_item_score = []
        cand_items = np.array(self.cand_items)
        item_embed = self.ui_embeds[cand_items + self.user_length]

        prefer_embed = self.feature_emb[self.user_acc_feature, :]
        unprefer_feature = list(set(self.user_rej_feature))
        unprefer_embed = self.feature_emb[unprefer_feature, :]
        
        score = 0
        
        if len(self.user_acc_feature) > 0:
            prefer_embed = np.mean(prefer_embed, 0)
            score += np.inner(prefer_embed, item_embed)
        cand_item_score = score
        return cand_item_score

    def _update_cand_items(self, asked_feature, acc_rej):
        if acc_rej:    # accept feature
            print('=== ask acc: update cand_items')
            feature_items = self.kg.G['feature'][asked_feature]['belong_to']
            self.cand_items = set(self.cand_items) & set(feature_items)   #  itersection
            self.cand_items = list(self.cand_items)
        else:    # reject feature
            print('=== ask rej: update cand_items')

        self.update_cand_item_scores()

    def update_cand_item_scores(self):
        cand_item_score = self._item_score()
        item_score_tuple = list(zip(self.cand_items, cand_item_score))
        sort_tuple = sorted(item_score_tuple, key=lambda x: x[1], reverse=True)
        self.cand_items, self.cand_item_score = zip(*sort_tuple)
    
    def _recommend_update(self, recom_items):
        print('-->action: recommend items')
        self.cand_items = list(self.cand_items)
        self.cand_item_score = list(self.cand_item_score)

        rank_idx = 100
        if self.target_item in recom_items:
            reward = self.reward_dict['rec_suc'] 
            self.conver_his[self.cur_conver_step] = self.history_dict['rec_scu'] #update state vector: conver_his
            tmp_score = []
            for item in recom_items:
                idx = self.cand_items.index(item)
                tmp_score.append(self.cand_item_score[idx])
            self.cand_items = recom_items
            self.cand_item_score = tmp_score
            done = recom_items.index(self.target_item) + 1
            rank_idx = np.where(recom_items == self.target_item)[0]
            self.suc_items.append(self.target_item)
        else:
            reward = self.reward_dict['rec_fail']
            self.conver_his[self.cur_conver_step] = self.history_dict['rec_fail']  #update state vector: conver_his
            if len(self.cand_items) > self.rec_num:
                for item in recom_items:
                    idx = self.cand_items.index(item)
                    self.cand_items.pop(idx)
                    self.cand_item_score.pop(idx)
            done = 0
            self.rej_items.extend(recom_items)
        return reward, done, rank_idx

    def _update_feature_entropy(self):
        if self.ent_way == 'entropy':
            cand_items_fea_list = []
            for item_id in self.cand_items:
                cand_items_fea_list.append(list(self.kg.G['item'][item_id]['belong_to']))
            cand_items_fea_list = list(chain(*cand_items_fea_list))

            self.attr_count_dict = dict(Counter(cand_items_fea_list))
            self.attr_ent = [0] * self.attr_state_num  # reset attr_ent
            real_ask_able = list(set(self.reachable_feature) & set(self.attr_count_dict.keys()))
            for fea_id in real_ask_able:
                p1 = float(self.attr_count_dict[fea_id]) / len(self.cand_items)
                p2 = 1.0 - p1
                if p1 == 1:
                    self.attr_ent[fea_id] = 0
                else:
                    ent = (- p1 * np.log2(p1) - p2 * np.log2(p2))
                    self.attr_ent[fea_id] = ent
        elif self.ent_way == 'weight_entropy':
            cand_items_fea_list = []
            self.attr_count_dict = {}
            
            cand_item_score_sig = self.sigmoid(self.cand_item_score)  # sigmoid(score)
            for score_ind, item_id in enumerate(self.cand_items):
                cand_items_fea_list = list(self.kg.G['item'][item_id]['belong_to'])
                for fea_id in cand_items_fea_list:
                    if self.attr_count_dict.get(fea_id) == None:
                        self.attr_count_dict[fea_id] = 0
                    self.attr_count_dict[fea_id] += cand_item_score_sig[score_ind]

            self.attr_ent = [0] * self.attr_state_num  # reset attr_ent
            real_ask_able = list(set(self.reachable_feature) & set(self.attr_count_dict.keys()))
            sum_score_sig = sum(cand_item_score_sig)

            for fea_id in real_ask_able:
                p1 = float(self.attr_count_dict[fea_id]) / sum_score_sig
                p2 = 1.0 - p1
                if p1 == 1 or p1 <= 0:
                    self.attr_ent[fea_id] = 0
                else:
                    ent = (- p1 * np.log2(p1) - p2 * np.log2(p2))
                    self.attr_ent[fea_id] = ent

    def sigmoid(self, x_list):
        x_np = np.array(x_list)
        s = 1 / (1 + np.exp(-x_np))
        return s.tolist()

    def _map_to_all_id(self, x_list, old_type):
        if old_type == 'item':
            return [x + self.user_length for x in x_list]
        elif old_type == 'feature':
            return [x + self.user_length + self.item_length for x in x_list]
        else:
            return x_list

    def _map_to_old_id(self, x):
        if x >= self.user_length + self.item_length:
            x -= (self.user_length + self.item_length)
        elif x >= self.user_length:
            x -= self.user_length
        return x

    def get_user_response(self, asked_feature):
        '''
        :return: reward, acc_feature, rej_feature
        '''
        done = 0

        item_attrs = self.kg.G['item'][self.target_item]['belong_to']
        user_attrs = self.user_attr[str(self.user_id)]
        
        if asked_feature in item_attrs and asked_feature in user_attrs:
            acc_rej = True
            self.user_acc_feature.append(asked_feature)
            self.cur_node_set.append(asked_feature)

            reward = self.reward_dict['ask_suc']
            self.conver_his[self.cur_conver_step] = self.history_dict['ask_suc']   #update conver_his
        else:
            acc_rej = False
            self.user_rej_feature.append(asked_feature)
            reward = self.reward_dict['ask_fail']
            self.conver_his[self.cur_conver_step] = self.history_dict['ask_fail']  #update conver_his

        if self.cand_items == []: 
            done = 1
            reward = self.reward_dict['cand_none']

        return reward, done, acc_rej

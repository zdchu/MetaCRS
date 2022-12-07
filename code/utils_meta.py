import numpy as np
import random
import pandas as pd
import IPython

class FewShotCRS:
    def __init__(self, path, batchsz, k_shot, k_query, seed=1234, test_rate=0.2):
        self.path = path
        self.indexes = {'train': 0, 'test': 0}
        self.interaction = pd.read_csv(path + '/interaction.csv')
        self.test_rate = test_rate
        self.load_user()

        self.k_shot = k_shot
        self.k_query = k_query
        self.batchsz = batchsz
        self.datasets_cache = {'train': self.load_data_cache(self.user_split['train']),
                                'test': self.load_data_cache(self.user_split['test'])}
        self.valid_index = 0
    
    def reset(self):
        self.valid_index = 0
    
    def random_split(self):
        user = self.interaction['user'].unique()
        train_user = np.random.choice(user, int(len(user) * (1 - self.test_rate)))
        test_user = np.setdiff1d(user, train_user)
        self.user_split = {'train': train_user, 'test': test_user}

    def load_user(self):
        train_user = np.load(self.path + '/user_train.npy')
        valid_user = np.load(self.path + '/user_valid.npy')
        
        users = set(self.interaction.user.drop_duplicates())
        train_user = np.array(list(users & set(train_user)))
        valid_user = np.array(list(users & set(valid_user)))
        random.shuffle(train_user)
        random.shuffle(valid_user)
        self.user_split = {'train': train_user, 'test': valid_user}
        
    def load_data_cache(self, user_pack):
        data_cache = []
        user_pack = list(set(self.interaction.user.drop_duplicates()) & set(user_pack))
        for _ in range(10):
            spt, qry = [], []
            selected_users = np.random.choice(user_pack, self.batchsz, False)
            
            for user in selected_users:
                items = self.interaction[self.interaction['user'] == user]['item']
                selected_items = np.random.choice(items, self.k_query + self.k_shot, False)
                spt.append([(user, i) for i in selected_items[:self.k_shot]])
                qry.append([(user, i) for i in selected_items[-self.k_query:]])
            data_cache.append([np.array(spt), np.array(qry)])
        return data_cache

    def next(self, mode='train'):
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.user_split[mode])
        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1
        return next_batch
    
    def next_batch(self, mode='train'):
        spt, qry = [], []
        while len(spt) < self.batchsz:
            user = self.user_split[mode][self.indexes[mode]]
            items = self.interaction[self.interaction['user'] == user]['item']
            selected_items = np.random.choice(items, self.k_query + self.k_shot, False)
            spt.append([(user, i) for i in selected_items[:self.k_shot]])
            qry.append([(user, i) for i in selected_items[-self.k_query:]])

            self.indexes[mode] += 1
            if self.indexes[mode] >= len(self.user_split[mode]):
                random.shuffle(self.user_split[mode])
                self.indexes[mode] = 0
        return np.array(spt), np.array(qry)
    

class FewShotCRS_test:
    def __init__(self, path, batchsz, k_shot, k_query, seed=1234, test_rate=0.2, mode='test'):
        self.path = path
        self.interaction = pd.read_csv(path + '/interaction.csv')

        self.k_shot = k_shot
        self.k_query = k_query
        self.batchsz = batchsz

        user = list(set(np.load(self.path + '/user_{}.npy'.format(mode))) & set(self.interaction.user.drop_duplicates()))
        self.user  = user
        
        self.num_user = len(self.user)
        self.index = 0

    def reset(self):
        self.index = 0
    
    def next(self):
        user = self.user[self.index]
        items = self.interaction[self.interaction['user'] == user]['item']
        selected_items = items
        spt = [(user, i) for i in selected_items[:self.k_shot]]
        qry = [(user, i) for i in selected_items[-self.k_query:]]
        self.index += 1
        return np.array(spt), np.array(qry)

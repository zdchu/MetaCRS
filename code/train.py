import numpy as np
import os
import sys
sys.path.append('../')

from collections import namedtuple
import argparse
import torch
from tqdm import tqdm
from copy import deepcopy
from utils import * 

#TODO select env
from RL.env_binary_question import BinaryRecommendEnv
from encoder import TransGate
import pandas as pd
import time
import IPython
import warnings
from utils_meta import FewShotCRS, FewShotCRS_test
from meta_rl import MetaRL_rnn
from agent import Decoupled_Agent

warnings.filterwarnings("ignore")
EnvDict = {
    LAST_FM: BinaryRecommendEnv,
    BOOK: BinaryRecommendEnv, 
    MOVIE: BinaryRecommendEnv, 
    }
FeatureDict = {
    LAST_FM: 'feature',
    BOOK: 'feature',
    MOVIE: 'feature'
}

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'next_cand'))

def train(args, kg, dataset):
    options = vars(args)
    print(options)
    env = EnvDict[args.data_name](args, kg, dataset, args.data_name, args.embed, seed=args.seed, max_turn=args.max_turn, cand_num=args.cand_num, cand_item_num=args.cand_item_num,
                       attr_num=args.attr_num, mode='train', ask_num=args.ask_num, entropy_way=args.entropy_method, fm_epoch=args.fm_epoch)
    set_random_seed(args.seed)
    embed = torch.FloatTensor(np.concatenate((env.ui_embeds, env.feature_emb, np.zeros((1,env.ui_embeds.shape[1]))), axis=0))
    
    encoder = TransGate(device=args.device, entity=embed.size(0), emb_size=embed.size(1), kg=kg, embeddings=embed, \
        fix_emb=args.fix_emb, seq=args.seq, gcn=args.gcn, hidden_size=args.state_size).to(args.device)

    agent = Decoupled_Agent(device=args.device, state_size=args.state_size, action_size=embed.size(1), \
            hidden_size=args.hidden, encoder=encoder, init_lr=args.lr_phi, meta_method=args.meta_method, num_users=env.user_length)
    
    fewshot_loader = FewShotCRS(path=DATA_DIR[args.data_name], batchsz=args.num_tasks, 
                        k_shot=args.k_shot, k_query=args.k_query)
    
    metaRL = MetaRL_rnn(args=args, env=env, agent=agent)    
    all_test_accs = []
    best_test = 0

    best_agent = None

    for train_step in tqdm(range(1, args.max_steps+1)):
        spt, qry = fewshot_loader.next_batch()
        acc, avg_turn = metaRL(spt, qry)
        
        if train_step % args.eval_every == 0:
            print('-----------Valid at step: {}------------'.format(train_step))
            test_acc, avg_turn  = test(args, kg, dataset, mode='valid', agent=metaRL.agent, eval_num=50)
            print('Valid acc: {}'.format(test_acc))

            all_test_accs.append(test_acc)
            if test_acc > best_test:
                best_test = test_acc
                best_agent = deepcopy(metaRL.agent)
                torch.save(metaRL.agent, DATA_DIR[args.data_name] + '/saved_model/saved_p_net_inner_{}_lrm_{}_lrp_{}_tasks_{}_steps_{}_{}_full_{}_hidden_{}_shot_{}_rank_{}.txt'.format(args.inner_steps, args.lr_meta, args.lr_phi,
                                                                                                                            args.num_tasks, args.max_steps, args.meta_method, args.full_update, 
                                                                                                                                 args.hidden, args.k_shot, args.item_rank))
    print('All test acc: {}, '.format(str(all_test_accs)))
    return best_agent


def test(args, kg, dataset, mode='test', agent=None, eval_num=None):
    test_env = EnvDict[args.data_name](args, kg, dataset, args.data_name, args.embed, seed=args.seed, max_turn=args.max_turn,
                                       cand_num=args.cand_num, cand_item_num=args.cand_item_num, attr_num=args.attr_num, mode='test', ask_num=args.ask_num, entropy_way=args.entropy_method,
                                       fm_epoch=args.fm_epoch)

    fewshot_loader = FewShotCRS_test(path=DATA_DIR[args.data_name], batchsz=args.num_tasks, 
                        k_shot=args.k_shot, k_query=args.k_query, mode=mode)
    
    if agent is None:
        agent = torch.load(DATA_DIR[args.data_name] + '/saved_model/saved_p_net_inner_{}_lrm_{}_lrp_{}_tasks_{}_steps_{}_{}_full_{}_hidden_{}_shot_{}_rank_{}.txt'.format(args.inner_steps, args.lr_meta, args.lr_phi,
                                                                                                                            args.num_tasks, args.max_steps, args.meta_method, args.full_update, 
                                                                                                                                  args.hidden, args.k_shot, args.item_rank))

    metaRL = MetaRL_rnn(args=args, env=test_env, agent=agent)
    count = 0
    accs = 0
    max_num = fewshot_loader.num_user if eval_num is None else eval_num
    avgTs = 0

    while fewshot_loader.index < max_num and fewshot_loader.index < fewshot_loader.num_user:
        print('-----------Test turn: {}------------'.format(count))
        spt, qry = fewshot_loader.next()
        acc, avgT = metaRL.fintunning(spt, qry)
        
        accs += acc
        count += qry.shape[0]
        avgTs += avgT
    print('All test acc: {}, Avg Turn: {}'.format(accs / float(count), avgTs / float(count)))
    return accs / float(count), avgTs / float(count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-seed', type=int, default=1, help='random seed.')
    parser.add_argument('-gpu', type=str, default='2', help='gpu device.')
    parser.add_argument('--fm_epoch', type=int, default=0, help='the epoch of FM embedding')
    parser.add_argument('--gamma', type=float, default=0.999, help='reward discount factor.')
    parser.add_argument('--l2_norm', type=float, default=1e-6, help='l2 regularization.')
    parser.add_argument('--state_size', type=int, default=100, help='size of state repr.')
    parser.add_argument('--hidden', type=int, default=100, help='number of samples')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate.')
    
    parser.add_argument('-lr_meta', type=float, default=5e-3)
    parser.add_argument('-lr_phi', type=float, default=1e-2)
    parser.add_argument('-inner_batch', type=int, default=1)

    parser.add_argument('--data_name', type=str, default=LAST_FM, choices=[LAST_FM, BOOK, MOVIE])
    parser.add_argument('--entropy_method', type=str, default='weight_entropy', help='entropy_method is one of {entropy, weight entropy, similarity}')
    
    parser.add_argument('--max_turn', type=int, default=10, help='max conversation turn')
    parser.add_argument('--attr_num', type=int, help='the number of attributes')
    parser.add_argument('--mode', type=str, default='train', help='the mode in [train, test]')
    parser.add_argument('--ask_num', type=int, default=1, help='the number of features asked in a turn')

    parser.add_argument('--max_steps', type=int, default=600, help='max training steps')
    parser.add_argument('--eval_every', type=int, default=50, help='the number of steps to evaluate RL model and metric')
    parser.add_argument('--save_num', type=int, default=10, help='the number of steps to save RL model and metric')
    parser.add_argument('--cand_num', type=int, default=10, help='candidate sampling number')
    parser.add_argument('--cand_item_num', type=int, default=500, help='candidate item sampling number')
    parser.add_argument('--fix_emb', action='store_false', help='fix embedding or not')
    parser.add_argument('--embed', type=str, default='graph', help='pretrained embeddings')
    parser.add_argument('--seq', type=str, default='transformer', choices=['rnn', 'transformer', 'mean', 'linear'], help='sequential learning method')
    parser.add_argument('--gcn', action='store_true', help='use GCN or not')

    parser.add_argument('-num_tasks', type=int, default=5)
    parser.add_argument('-k_shot', type=int, default=15)
    parser.add_argument('-k_query', type=int, default=10)
    parser.add_argument('-inner_steps', type=int, default=1)
    parser.add_argument('-test', action='store_true', default=False, help='train or test')
    parser.add_argument('-full_update', action='store_true')
    parser.add_argument('-meta_method', type=str, default='raw')
    
    parser.add_argument('-item_rank', action='store_true')
    parser.add_argument('-num_explore', type=int, default=5)
    

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    
    print(args.device)
    print('data_set:{}'.format(args.data_name))
    kg = load_kg(args.data_name)
    #reset attr_num
    
    feature_name = FeatureDict[args.data_name]
    feature_length = len(kg.G[feature_name].keys())
    print('dataset:{}, feature_length:{}'.format(args.data_name, feature_length))
    args.attr_num = feature_length  # set attr_num  = feature_length
    print('args.attr_num:', args.attr_num)
    print('args.entropy_method:', args.entropy_method)

    dataset = load_dataset(args.data_name)

    if not args.test:
        agent = train(args, kg, dataset)
        test(args, kg, dataset, agent=agent)
    else:
        test(args, kg, dataset)

if __name__ == '__main__':
    main()
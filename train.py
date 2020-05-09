# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 00:21:34 2019

@author: Xulin
"""

import os
import argparse
import numpy as np
import torch
from res_net_gru import *
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from dataset import *
from unzip_pdbe_maps import *

def total_param_num(net):
    num = 0
    for param in net.parameters():
        num += param.numel()
    return num

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_train', type = bool, default = True)
    parser.add_argument('--train_root_dir', type = str, default = '../data_synthesis/fake/*/')
    parser.add_argument('--test_root_dir', type = str, default = '../../dataset/testSet/EMD-????/')
    parser.add_argument('--train_proj_re', type = str, default = '*_projs.mrcs')
    parser.add_argument('--train_structure_re', type = str, default = '*.mrc')
    parser.add_argument('--test_proj_re', type = str, default = '*_64_norm_24projs.mrcs')
    parser.add_argument('--test_structure_re', type = str, default = '*_64_norm.mrc')
    parser.add_argument('--bs', type = int, help = 'Batch size.', default = 4)
    parser.add_argument('--seq_len', type = int, help = 'Sequence length.', default = 5)
    parser.add_argument('--epoch', type = int, help = 'Epoch.', default = 20)
    parser.add_argument('--lr', type = float, help = 'Initial learning rate.', default = 1e-3)
    # parser.add_argument('--dropout_prob', type = float, default = 0)
    # parser.add_argument('--mc_dropout', type = bool, default = False)
    parser.add_argument('--model_path', type = str, help = 'Model path.', default = './model/')
    parser.add_argument('--model_name', type = str, default = 'model.pkl')
    parser.add_argument('--summary_path', type = str, help = 'Summary path.', default = './summary/')
    parser.add_argument('--test_path', type = str, default = './test/')
    
    config = parser.parse_args()
    if config.is_train:
        train(config)

def train(config):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Create the network
    net = Res_Gru_Net(config.seq_len).to(device)
    
    full_model_path = config.model_path + config.model_name
    if os.path.exists(full_model_path):
        net.load_state_dict(torch.load(full_model_path))
        print('Load the pre-trained model successfully!')
    else:
        weight_init(net)
        if not os.path.exists(config.model_path):
            os.makedirs(config.model_path)
        print('First time training!')
    net.train()

    print(total_param_num(net))

    # Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr = config.lr, betas = [0.5, 0.999])

    # Loss function
    loss_func = torch.nn.MSELoss()

    # Save the results
    writer = SummaryWriter(config.summary_path)
    
    if not os.path.exists(config.test_path):
        os.makedirs(config.test_path)

    # Create the dataset
    train_mrc_root_paths = sorted(glob.glob(config.train_root_dir))
    trainset = ReadMRCs(config.seq_len, train_mrc_root_paths, config.train_proj_re, config.train_structure_re)
    train_loader = DataLoader(trainset, config.bs, shuffle = True)

    test_mrc_root_paths = sorted(glob.glob(config.test_root_dir))
    testset = ReadMRCs(config.seq_len, test_mrc_root_paths, config.test_proj_re, config.test_structure_re)
    test_loader = DataLoader(testset, 16, shuffle = False)

    print(len(trainset), len(testset))

    # Start training
    total_iter = 1
    for e in range(config.epoch):
        for idx, (proj, structure, _) in enumerate(train_loader):
            proj, structure = proj.to(device), structure.to(device)
            proj.requires_grad_(True)
            structure.requires_grad_(True)

            optimizer.zero_grad()
            pred = net(proj)
            loss = loss_func(pred, structure)
            loss.backward()
            optimizer.step()

            print('[Epoch %d|Batch %d] Loss = %.6f' % (e, idx, loss.item()))

            writer.add_scalar('Train/Loss', loss.item(), total_iter)
            total_iter += 1

        if e % 1 == 0:
            torch.save(net.state_dict(), full_model_path)

        if e % 1 == 0:
            net.eval()

            test_epoch_root_dir = config.test_path + 'epoch_%d/' % e
            if not os.path.exists(test_epoch_root_dir):
                os.makedirs(test_epoch_root_dir)

            with torch.no_grad():
                for proj, _, mrc_paths in test_loader:
                    proj = proj.to(device)
                    pred = net(proj).squeeze().detach().cpu().numpy()

                    bs = proj.size(0)
                    for b in range(bs):
                        mrc_id = mrc_paths[b].split('/')[-2]
                        save_3d_structure(pred[b], test_epoch_root_dir + mrc_id + '.mrc')

            net.train()

    writer.close()


if __name__ == '__main__':
    main()

    # mrc_proj_pair_list = find_data_under_dir('/home/xulin/Documents/Dataset/PDB/testSet', '/*/*_128_norm.mrc', '/*/*_128_projs_norm.mrcs')

    # bl = []
    # for mrc, proj in mrc_proj_pair_list:
    #     _, mrc_prefix, mrc_postfix = split_file_path(mrc)
    #     _, proj_prefix, proj_postfix = split_file_path(proj)

    #     mrc_prefix = mrc_prefix[:-9]
    #     proj_prefix = proj_prefix[:-15]

    #     bl.append(mrc_prefix == proj_prefix)

    # print(np.sum(bl))

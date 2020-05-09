# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 23:37:23 2019

@author: Xulin
"""

from torch.utils.data import Dataset
import torch
import random
import numpy as np
import mrcfile as mf
import os
import glob

# class Read2dProj(Dataset):
#     def __init__(self, projs_stack_path, seq_len, dataset_size):
#         super(Read2dProj, self).__init__()

#         # Read the projection file.
#         self.projs = self.read_single_stack(projs_stack_path)
#         self.projs = np.expand_dims(self.projs, axis = 1)
#         indices = list(range(self.projs.shape[0]))

#         self.indices_seq = [random.sample(indices, seq_len) for _ in range(dataset_size)]

#     def read_single_stack(self, path):
#         with mf.open(path) as f:
#             return f.data

#     def __getitem__(self, idx):
#         indices_seq = self.indices_seq[idx]
#         data = torch.tensor([self.projs[idx] for idx in indices_seq])
#         return data

#     def __len__(self):
#         return len(self.indices_seq)

def read_3d_structure(path):
    with mf.open(path) as f:
        return f.data

def save_3d_structure(data, path):
    with mf.new(path, overwrite = True) as f:
        f.set_data(data)

def create_gt_data(bs, gt):
    if bs > 1:
        return torch.tensor([gt for _ in range(bs)], requires_grad = True)
    elif bs == 1:
        return torch.tensor(gt, requires_grad = True)
    else:
        raise ValueError('Invalid batch size!')


# 2020.2.18
# class Read2dProjs(Dataset):
#     def __init__(self, proj_stack_root_path, structure_root_path, seq_len, dataset_size, is_train = True):
#         super(Read2dProjs, self).__init__()

#         self.proj_stack_paths = sorted(glob.glob(proj_stack_root_path + '*.mrcs'))
#         self.structure_paths = sorted(glob.glob(structure_root_path + '*.mrc'))

#         # print(self.proj_stack_paths)
#         # print(self.structure_paths)

#         self.proj_data = [self.read_single_stack(p) for p in self.proj_stack_paths]
#         self.stru_data = [self.read_single_stack(p) for p in self.structure_paths]
#         self.dataset_size = dataset_size if is_train else len(self.structure_paths)
#         self.proj_num_in_each_stack = [pd.shape[0] for pd in self.proj_data]
        
#         if is_train:
#             self.stru_indices_seq = [random.randint(0, len(self.proj_stack_paths) - 1) for _ in range(dataset_size)]
#         else:
#             self.stru_indices_seq = list(range(self.dataset_size))

#         self.proj_indices_seq = [random.sample(list(range(self.proj_num_in_each_stack[self.stru_indices_seq[s]])), seq_len) for s in range(self.dataset_size)]
        
#     def read_single_stack(self, path):
#         with mf.open(path) as f:
#             return f.data

#     def __len__(self):
#         return self.dataset_size

#     def __getitem__(self, idx):
#         stru_idx = self.stru_indices_seq[idx]
#         proj_idx_seq = self.proj_indices_seq[idx]

#         proj_data = torch.tensor([self.proj_data[stru_idx][idx] for idx in proj_idx_seq]).unsqueeze(1)
#         stru_data = torch.tensor(self.stru_data[stru_idx])

#         return proj_data, stru_data, self.structure_paths[stru_idx]

# 3.3 for PDBe dataset
def find_data_under_dir(root_dir, mrc_re, proj_re):
    return list(zip(sorted(glob.glob(root_dir + mrc_re)), sorted(glob.glob(root_dir + proj_re))))

# class Read2dProjs(Dataset):
#     def __init__(self, mrc_proj_path_pairs, seq_len, data_size_per_mrc):
#         '''
#             mrc_proj_path_pairs: a list of (structure_path, projection_path) (the output of find_data_under_dir())
#             seq_len: the number of views fed into the network once
#             dataset: 
#         '''
#         super(Read2dProjs, self).__init__()

#         # Read data
#         self.mrc_proj_path_pairs = mrc_proj_path_pairs
#         self.dataset_size = data_size_per_mrc * len(self.mrc_proj_path_pairs)

#         self.mrc_data, self.proj_data = [], []
#         for mp, pp in mrc_proj_path_pairs:
#             self.mrc_data.append(self.read_single_mrc(mp))
#             self.proj_data.append(self.read_single_mrc(pp))

#         # Generate indices
#         self.mrc_indices = np.concatenate([np.arange(0, len(self.mrc_proj_path_pairs)) for _ in range(data_size_per_mrc)])
#         np.random.shuffle(self.mrc_indices)
#         self.proj_nums = [pd.shape[0] for pd in self.proj_data]
#         self.proj_indices = [random.sample(list(range(self.proj_nums[idx])), seq_len) for idx in self.mrc_indices]

#     def read_single_mrc(self, path):
#         with mf.open(path) as f:
#             return f.data

#     def __len__(self):
#         return self.dataset_size

#     def __getitem__(self, idx):
#         mrc_idx, proj_idx_seq = self.mrc_indices[idx], self.proj_indices[idx]

#         mrc_path, proj_path = self.mrc_proj_path_pairs[mrc_idx]

#         mrc_t = torch.tensor(self.mrc_data[mrc_idx])
#         proj_t = torch.tensor([self.proj_data[mrc_idx][i] for i in proj_idx_seq]).unsqueeze(1)

#         return mrc_path, proj_path, mrc_idx, proj_idx_seq, mrc_t, proj_t


class ReadMRCs(Dataset):
    def __init__(self, seq_len, root_dirs, proj_re, structure_re):
        super(ReadMRCs, self).__init__()

        self.projs, self.structures, self.proj_nums = [], [], []
        self.seq_len = seq_len
        self.root_dirs = []

        for rd in root_dirs:
            proj_path = sorted(glob.glob(rd + proj_re))
            structure_path = sorted(glob.glob(rd + structure_re))
            if len(proj_path) != 1 or len(structure_path) != 1:
                continue

            proj = self.read_single_mrc(proj_path[0])
            structure = self.read_single_mrc(structure_path[0])

            self.projs.append(proj)
            self.structures.append(structure)
            self.proj_nums.append(proj.shape[0])
            self.root_dirs.append(rd)

    def read_single_mrc(self, path):
        with mf.open(path) as f:
            return f.data

    def __getitem__(self, idx):
        proj, structure, proj_num = self.projs[idx], self.structures[idx], self.proj_nums[idx]

        rdm_projs = random.sample(list(range(proj_num)), self.seq_len)
        proj = proj[rdm_projs, ...]

        mrc_id = self.root_dirs[idx]

        return torch.from_numpy(proj).unsqueeze(1), torch.from_numpy(structure).unsqueeze(0), mrc_id

    def __len__(self):
        return len(self.root_dirs)

if __name__ == '__main__':
    dataset_paths = sorted(glob.glob('../data_synthesis/fake/*/'))[:5]
    dataset = ReadMRCs(5, dataset_paths, '*_projs.mrcs', '*.mrc')
    loader = torch.utils.data.DataLoader(dataset, 4, shuffle = True)

    for proj, structure, mid in loader:
        print(proj.size(), structure.size(), mid)
        break

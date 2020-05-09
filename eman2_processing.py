import argparse
import os

def normalize2d(orig_projs_path, dest_projs_path):
    if not os.path.exists(orig_projs_path):
        raise ValueError('Cannot open the input file! --resize()')

    cmd = 'e2proc2d.py %s %s --process normalize.edgemean' % (orig_projs_path, dest_projs_path)
    os.system(cmd)

    print('OK! --normalize2d()')

def normalize3d(orig_stru_path, dest_stru_path):
    if not os.path.exists(orig_stru_path):
        raise ValueError('Cannot open the input file! --resize()')

    cmd = 'e2proc3d.py %s %s --process normalize.edgemean' % (orig_stru_path, dest_stru_path)
    os.system(cmd)

    print('OK! --normalize3d()')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--projs_path', type = str, default = './data/projs/1081_160_mask_projs.mrcs')
    parser.add_argument('--normalized_projs_path', type = str, default = './data/norm_projs/1081_160_mask_projs.mrcs')
    parser.add_argument('--stru_path', type = str, default = './data/resize/1081_160.mrc')
    parser.add_argument('--normalized_stru_path', type = str, default = './data/norm_resize/1081_160.mrc')

    config = parser.parse_args()
    normalize2d(config.projs_path, config.normalized_projs_path)
    normalize3d(config.stru_path, config.normalized_stru_path)

def multi_process(root_path, proj_dir, stru_dir, norm_proj_dir = 'norm_projs/', norm_stru_dir = 'norm_resize/'):
    proj_names = []
    stru_names = sorted(os.listdir(root_path + stru_dir))

    for pn in os.listdir(root_path + proj_dir):
        if os.path.splitext(pn)[1] == '.mrcs':
            proj_names.append(pn)
    proj_names = sorted(proj_names)

    norm_proj_root_path = root_path + norm_proj_dir
    proj_root_path = root_path + proj_dir
    if not os.path.exists(norm_proj_root_path):
        os.makedirs(norm_proj_root_path)

    norm_stru_root_path = root_path + norm_stru_dir
    stru_root_path = root_path + stru_dir
    if not os.path.exists(norm_stru_root_path):
        os.makedirs(norm_stru_root_path)

    # print(proj_names)
    # print(stru_names)

    for proj in proj_names:
        normalize2d(proj_root_path + proj, norm_proj_root_path + proj)

    for stru in stru_names:
        normalize3d(stru_root_path + stru, norm_stru_root_path + stru)

def single_proj_process(mrc_path):
    mrc_dir, mrc_name = os.path.split(mrc_path)
    mrc_prefix, mrc_postfix = os.path.splitext(mrc_name)

    normalize2d(mrc_path, mrc_dir + '/' + mrc_prefix + '_norm' + mrc_postfix)

def single_mrc_process(proj_path):
    proj_dir, proj_name = os.path.split(proj_path)
    proj_prefix, proj_postfix = os.path.splitext(proj_name)

    normalize3d(proj_path, proj_dir + '/' + proj_prefix + '_norm' + proj_postfix)

if __name__ == '__main__':
    # main()
    # multi_process('./data/test/', 'projs/', 'resize/')

    import glob

    dataset_root_dir = '/home/xulin/Documents/Dataset/PDB/testSet'
    resized_mrcs = glob.glob(dataset_root_dir + '/*/*_64.mrc')
    # projs = glob.glob(dataset_root_dir + '/*/*64_projs.mrcs')

    for m in resized_mrcs:
        single_mrc_process(m)
        print('Finish processing %s!' % m)

    # for p in projs:
    #     single_proj_process(p)
    #     print('Finish processing %s!' % p)
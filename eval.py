import argparse
import torch
import os
from dataset import save_3d_structure
from res_net_gru import Res_Gru_Net
from dataset import Read2dProjs
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--projs_root_path', type = str, help = '2D projection image stack path.', default = './data/test/norm_projs/')
    parser.add_argument('--strus_root_path', type = str, default = './data/test/norm_resize/')
    parser.add_argument('--seq_len', type = int, help = 'Sequence length.', default = 5)
    parser.add_argument('--in_img_size', type = list, default = [160, 160])
    parser.add_argument('--model_path', type = str, help = 'Model path.', default = './model/')
    parser.add_argument('--model_name', type = str, default = 'model.pkl')
    parser.add_argument('--gen_structure_path', type = str, default = './fake/test/')
    
    config = parser.parse_args()
    eval_model(config)

def eval_model(config):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if not os.path.exists(config.gen_structure_path):
        os.makedirs(config.gen_structure_path)

    test_dataset = Read2dProjs(config.projs_root_path, config.strus_root_path, config.seq_len, 4, is_train = False)
    test_loader = DataLoader(test_dataset, 1, shuffle = False)

    net = Res_Gru_Net(config.in_img_size, config.seq_len).to(device)
    net.load_state_dict(torch.load(config.model_path + config.model_name))
    net.eval()

    for data, _, stru_name in test_loader:
        data = data.to(device)
        pred = net(data, device).squeeze().detach().cpu().numpy()
        save_3d_structure(pred, config.gen_structure_path + stru_name[0].split('/')[-1])

    print('Done!')

if __name__ == '__main__':
    main()
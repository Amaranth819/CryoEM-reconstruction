import os
import argparse
import torch
from res_net_gru import Res_Gru_Net
from dataset import Read2dProj, save_3d_structure

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--projs_path', type = str, default = './data/projs_norm.mrcs')
    parser.add_argument('--num', type = int, default = 4)
    parser.add_argument('--in_img_size', type = list, default = [160, 160])
    parser.add_argument('--seq_len', type = int, default = 5)
    parser.add_argument('--model_path', type = str, default = './model/')
    parser.add_argument('--model_name', type = str, default = 'model.pkl')
    parser.add_argument('--pred_path', type = str, default = './pred/')
    parser.add_argument('--pred_name', type = str, default = '10288')

    config = parser.parse_args()
    pred(config)

def pred(config):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Data
    if not os.path.exists(config.projs_path):
        raise ValueError('Unknown image stack path!')

    pred_data = Read2dProj(config.projs_path, config.seq_len, config.num)
    pred_loader = torch.utils.data.DataLoader(pred_data, 1, shuffle = False)

    # Network
    net = Res_Gru_Net(config.in_img_size, config.seq_len).to(device)
    full_model_path = config.model_path + config.model_name
    if not os.path.exists(full_model_path):
        raise ValueError('Unknown pretrained model path!')
    net.load_state_dict(torch.load(full_model_path))
    net.eval()

    # Save the predicted structures
    if not os.path.exists(config.pred_path):
        os.makedirs(config.pred_path)

    # Start predicting
    with torch.no_grad():
        for idx, data in enumerate(pred_loader):
            data = data.to(device)
            pred_structure = net(data, device).squeeze().detach().cpu().numpy()
            save_3d_structure(pred_structure, config.pred_path + config.pred_name + '_%d.mrc' % idx)

    print('Finish predicting!')


if __name__ == '__main__':
    main()
import torch
import torch.nn as nn
import math
from layer import Residual_Block_2D, Residual_Block_3D, BN_FC_Conv3D

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()

		self.main = nn.Sequential(
			Residual_Block_2D(1, 32, shortcut = False),
			nn.MaxPool2d(2, ceil_mode = True),
			Residual_Block_2D(32, 64),
			nn.MaxPool2d(2, ceil_mode = True),
			Residual_Block_2D(64, 128),
			nn.MaxPool2d(2, ceil_mode = True),
			Residual_Block_2D(128, 256, shortcut = False),
			nn.MaxPool2d(2, ceil_mode = True),
			Residual_Block_2D(256, 256),
			nn.MaxPool2d(2, ceil_mode = True),
			nn.Flatten(),
			nn.Linear(256 * 2 * 2, 1024)
		)

	def forward(self, x):
		return self.main(x)


class GRU3d(nn.Module):
	def __init__(self, seq_len, n_gru_vox):
		super(GRU3d, self).__init__()

		self.n_gru_vox = n_gru_vox
		self.gru3d_u = BN_FC_Conv3D(1024, 128, seq_len, self.n_gru_vox)
		self.gru3d_r = BN_FC_Conv3D(1024, 128, seq_len, self.n_gru_vox)
		self.gru3d_rs = BN_FC_Conv3D(1024, 128, seq_len, self.n_gru_vox)
		self.sigmoid = nn.Sigmoid()
		self.tanh = nn.Tanh()
		self.hidden_size = [128, self.n_gru_vox, self.n_gru_vox, self.n_gru_vox]

	def forward(self, x, h, u, idx):
		update = self.sigmoid(self.gru3d_u(x, h, idx))
		reset = self.sigmoid(self.gru3d_r(x, h, idx))
		rs = self.tanh(self.gru3d_rs(x, reset * h, idx))
		x = update * rs + (1.0 - update) * h
		return x, update


class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()

		self.filters = [128, 128, 128, 64, 32, 1]

		# Build the network
		self.main = nn.Sequential(
			nn.ConvTranspose3d(128, 128, 2, 2),
			Residual_Block_3D(128, 128),
			nn.ConvTranspose3d(128, 128, 2, 2),
			Residual_Block_3D(128, 128),
			nn.ConvTranspose3d(128, 128, 2, 2),
			Residual_Block_3D(128, 64),
			nn.ConvTranspose3d(64, 64, 2, 2),
			Residual_Block_3D(64, 32),
			nn.ConvTranspose3d(32, 32, 2, 2),
			nn.Conv3d(32, 1, 3, 1, 1)
		)

	def forward(self, x):
		return self.main(x)


class Res_Gru_Net(nn.Module):
	def __init__(self, seq_len):
		super(Res_Gru_Net, self).__init__()

		n_gru_vox = int(64 / 2**5)
		self.encoder = Encoder()
		self.gru3d = GRU3d(seq_len, n_gru_vox)
		self.decoder = Decoder()
		self.seq_len = seq_len

		# self.h = nn.Parameter(torch.zeros([1] + self.gru3d.hidden_size))
		# self.u = nn.Parameter(torch.zeros([1] + self.gru3d.hidden_size))

	def init_hidden(self, shape, device):
		return torch.zeros(shape, requires_grad = True).to(device)

	def forward(self, x):
		bs = x.size()[0]
		hidden_shape = [bs] + self.gru3d.hidden_size
		device = x.get_device()
		h = self.init_hidden(hidden_shape, device)
		u = self.init_hidden(hidden_shape, device)

		for idx in range(self.seq_len):
			xb = x[:, idx, ...]
			xb = self.encoder(xb)
			h, u = self.gru3d(xb, h, u, idx)

		h = self.decoder(h)

		return h



def weight_init(network):
	for each_module in network.modules():
		if isinstance(each_module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d)):
			torch.nn.init.xavier_uniform_(each_module.weight)
			if each_module.bias is not None:
				each_module.bias.data.zero_()
		elif isinstance(each_module, (nn.BatchNorm2d, nn.BatchNorm3d)):
			each_module.weight.data.fill_(1.)
			if each_module.bias is not None:
				each_module.bias.data.zero_()
		elif isinstance(each_module, nn.Linear):
			each_module.weight.data.normal_(0, 0.01)
			if each_module.bias is not None:
				each_module.bias.data.zero_()

if __name__ == '__main__':
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	# x = torch.zeros((2, 5, 1, 160, 160)).to(device)
	# net = Res_Gru_Net([160, 160], 5).to(device)
	# weight_init(net)
	# y = net(x, device)
	# print(y.size())
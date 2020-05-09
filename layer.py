import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Residual_Block_2D(nn.Module):
	def __init__(self, in_c, out_c, kernel_size = 3, stride = 1, padding = 1, neg_slope = 0.1, shortcut = True):
		super(Residual_Block_2D, self).__init__()

		self.main = nn.Sequential(
			nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias = False),
			nn.BatchNorm2d(out_c),
			nn.LeakyReLU(neg_slope, True),
			nn.Conv2d(out_c, out_c, 3, 1, 1, bias = False),
			nn.BatchNorm2d(out_c)
		)

		if shortcut:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_c, out_c, 1, bias = False),
				nn.BatchNorm2d(out_c)
			)
		else:
			self.shortcut = None

		self.relu = nn.LeakyReLU(neg_slope, True)

	def forward(self, x):
		out = self.main(x)
		if self.shortcut is not None:
			out += self.shortcut(x)
		out = self.relu(out)
		return out

class Residual_Block_3D(nn.Module):
	def __init__(self, in_c, out_c, kernel_size = 3, stride = 1, padding = 1, neg_slope = 0.1, shortcut = True):
		super(Residual_Block_3D, self).__init__()

		self.main = nn.Sequential(
			nn.Conv3d(in_c, out_c, kernel_size, stride, padding, bias = False),
			nn.BatchNorm3d(out_c),
			nn.LeakyReLU(neg_slope, True),
			nn.Conv3d(out_c, out_c, 3, 1, 1, bias = False),
			nn.BatchNorm3d(out_c)
		)

		if shortcut:
			self.shortcut = nn.Sequential(
				nn.Conv3d(in_c, out_c, 1, bias = False),
				nn.BatchNorm3d(out_c)
			)
		else:
			self.shortcut = None

		self.relu = nn.LeakyReLU(neg_slope, True)

	def forward(self, x):
		out = self.main(x)
		if self.shortcut is not None:
			out += self.shortcut(x)
		out = self.relu(out)
		return out

class Recurrent_BatchNorm3d(nn.Module):
	def __init__(self, num_features, T_max, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True):
		super(Recurrent_BatchNorm3d, self).__init__()

		self.T_max = T_max
		self.weight = nn.Parameter(torch.zeros(num_features).fill_(0.1))
		self.bias = nn.Parameter(torch.zeros(num_features))

		for i in range(T_max):
			self.register_buffer('mean_{}'.format(i), torch.zeros(num_features))
			self.register_buffer('var_{}'.format(i), torch.ones(num_features))

	def forward(self, x, t):
		if t >= self.T_max:
			t = self.T_max - 1

		mean = getattr(self, 'mean_{}'.format(t))
		var = getattr(self, 'var_{}'.format(t))

		return nn.functional.batch_norm(
			x,
			running_mean = mean,
			running_var = var,
			weight = self.weight,
			bias = self.bias
		)

class BN_FC_Conv3D(nn.Module):
	def __init__(self, encoder_out_c, decoder_in_c, seq_len, n_gru_vox):
		super(BN_FC_Conv3D, self).__init__()
		
		self.fc = nn.Linear(encoder_out_c, decoder_in_c * n_gru_vox * n_gru_vox * n_gru_vox, bias = False)
		self.fc_out_shape = (-1, decoder_in_c, n_gru_vox, n_gru_vox, n_gru_vox)
		self.conv = nn.Conv3d(decoder_in_c, decoder_in_c, 3, 1, 1, bias = False)
		self.bn1 = Recurrent_BatchNorm3d(decoder_in_c, seq_len)
		self.bn2 = Recurrent_BatchNorm3d(decoder_in_c, seq_len)
		self.bias = nn.Parameter(torch.zeros(1, decoder_in_c, 1, 1, 1).fill_(0.1))

	def forward(self, x, h, idx):
		x = self.fc(x).view(*self.fc_out_shape)
		bn_x = self.bn1(x, idx)
		bn_conv = self.bn2(self.conv(h), idx)
		return bn_x + bn_conv + self.bias
		
if __name__ == '__main__':
	bn = Recurrent_BatchNorm3d(8, 4)
	x = torch.ones((1, 8, 2, 2, 2))
	print(bn(x, 1).size())
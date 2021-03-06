import torch
import torch.nn as nn


class autoencoderV4(nn.Module):
	"""
	docstring for autoencoderV4
	"""
	def __init__(self):
		super(autoencoderV4, self).__init__()

		self.encoderL1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0)
		self.encoderL2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0)

		self.decoderL1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2, padding=0)
		self.decoderL2 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=1, stride=1, padding=0)

		self.relu = nn.ReLU(inplace=True)
		self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)


	def forward(self, x):

		x = self.encoderL1(x)
		x = self.relu(x)
		x = self.encoderL2(x)
		x = self.relu(x)
		x = self.max_pool(x)

		x_encoded = x

		x = self.decoderL1(x)
		x = self.relu(x)
		x = self.decoderL2(x)

		x_decoded = x

		return x_encoded, x_decoded





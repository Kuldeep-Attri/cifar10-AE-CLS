import torch
import torch.nn as nn



class autoencoderV2(nn.Module):
	"""
	docstring for autoencoderV2
	"""
	def __init__(self): # Input shape is 32 for the CIFAR10 (32 = height and width)
		super(autoencoderV2, self).__init__()
		
		self.encoderL1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.encoderL2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
		self.encoderL3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
		self.enocderL4 = nn.Linear(in_features=1024, out_features=256)

		self.decoderL1 = nn.Linear(in_features=256, out_features=1024)
		self.decoderL2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)
		self.decoderL3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2, padding=0)
		self.decoderL4 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=2, stride=2, padding=0)	

		self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
		self.relu = nn.ReLU(inplace=True)
		self.sigmoid = nn.Sigmoid()


	def forward(self, x):

		x = self.encoderL1(x)
		x = self.relu(x)
		x = self.max_pool(x)
		x = self.encoderL2(x)
		x = self.relu(x)
		x = self.max_pool(x)
		x = self.encoderL3(x)
		x = self.relu(x)
		x = self.max_pool(x)
		x = x.view(x.size(0), -1) # For Linear Layer
		x = self.enocderL4(x)
		x = self.relu(x)

		x_encoded = x

		x = self.decoderL1(x)
		x = self.relu(x)
		x = x.view(x.size(0), 64, 4, 4)
		x = self.decoderL2(x)
		x = self.relu(x)
		x = self.decoderL3(x)
		x = self.relu(x)
		x = self.decoderL4(x)
		# x = self.relu(x) # self.sigmoid(x)


		x_decoded = x


		return x_encoded, x_decoded






import torch
import torch.nn as nn



class autoencoderV1(nn.Module):
	"""
	docstring for autoencoderV1
	A simple 3 Linear Layer Autoencoder with ReLU activation function.
	Number of neurons are hard coded, we can play with them for optimal performamce.
	
	Input: Image
		Number of Neurons: [512, 256, 128] for 3 layers
		Default Input Shape:  3072 = 3*32*32 (Cifar 10 data size: RGB*height*width)

	Returns: Encoded features and the Decoded features
	"""

	def __init__(self, input_shape=3072):
		super(autoencoderV1, self).__init__()

		self.input_shape = input_shape

		self.encoderL1 = nn.Linear(in_features=self.input_shape, out_features=512)
		self.encoderL2 = nn.Linear(in_features=512, out_features=256)
		self.encoderL3 = nn.Linear(in_features=256, out_features=128)

		self.decoderL1 = nn.Linear(in_features=128, out_features=256)
		self.decoderL2 = nn.Linear(in_features=256, out_features=512)
		self.decoderL3 = nn.Linear(in_features=512, out_features=self.input_shape)

		self.relu = nn.ReLU(inplace=True)


	def forward(self, x):

		x = self.encoderL1(x)
		x = self.relu(x)
		x = self.encoderL2(x)
		x = self.relu(x)
		x = self.encoderL3(x)
		x = self.relu(x)

		x_encoded = x

		x = self.decoderL1(x)
		x = self.relu(x)
		x = self.decoderL2(x)
		x = self.relu(x)
		x = self.decoderL3(x)

		x_decoded = x

		return x_encoded, x_decoded








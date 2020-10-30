import torch
import torch.nn as nn

from net import autoencoderV1, classifierV1


class modelV1(nn.Module):
	"""
	docstring for modelV1

	Input: Image

	Returns: The Classes values from classifier and the Decoded Image
	Both will be used in the Loss function
	"""
	def __init__(self, input_shape=3*32*32, num_classes=10):
		super(modelV1, self).__init__()
		
		self.input_shape = input_shape
		self.num_classes = num_classes

		self.autoencoder = autoencoderV1(self.input_shape)
		self.classifier = classifierV1(input_shape=128, num_classes=self.num_classes)


	def forward(self, x):

		x_enc, x_dec = self.autoencoder(x)
		x_cls = self.classifier(x_enc)

		return x_cls, x_dec









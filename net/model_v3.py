import torch
import torch.nn as nn

from net import autoencoderV3, classifierV3


class modelV3(nn.Module):
	"""
	docstring for modelV3

	Input: Image

	Returns: The Classes values from classifier and the Decoded Image
	Both will be used in the Loss function
	"""
	def __init__(self, num_classes=10):
		super(modelV3, self).__init__()
		
		self.num_classes = num_classes

		self.autoencoder = autoencoderV3()
		self.classifier = classifierV3(in_channels=64, num_classes=self.num_classes)


	def forward(self, x):

		x_enc, x_dec = self.autoencoder(x)
		x_cls = self.classifier(x_enc)

		return x_cls, x_dec









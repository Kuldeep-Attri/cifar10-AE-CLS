import torch
import torch.nn as nn

from net import autoencoderV4, classifierV4


class modelV4(nn.Module):
	"""
	docstring for modelV4

	Input: Image

	Returns: The Classes values from classifier and the Decoded Image
	Both will be used in the Loss function
	"""
	def __init__(self, num_classes=10):
		super(modelV4, self).__init__()
		
		self.num_classes = num_classes

		self.autoencoder = autoencoderV4()
		self.classifier = classifierV4(in_channels=32, num_classes=self.num_classes)


	def forward(self, x):

		x_enc, x_dec = self.autoencoder(x)
		x_cls = self.classifier(x_enc)

		return x_cls, x_dec









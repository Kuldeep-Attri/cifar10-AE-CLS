import torch
import torch.nn as nn

from net import autoencoderV2, classifierV2


class modelV2(nn.Module):
	"""
	docstring for modelV2

	Input: Image

	Returns: The Classes values from classifier and the Decoded Image
	Both will be used in the Loss function
	"""
	def __init__(self, num_classes=10):
		super(modelV2, self).__init__()
		
		self.num_classes = num_classes

		self.autoencoder = autoencoderV2()
		self.classifier = classifierV2(input_shape=256, num_classes=self.num_classes)


	def forward(self, x):

		x_enc, x_dec = self.autoencoder(x)
		x_cls = self.classifier(x_enc)

		return x_cls, x_dec









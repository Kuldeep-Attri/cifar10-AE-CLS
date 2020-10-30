import torch
import torch.nn as nn



class classifierV1(nn.Module):
	"""
	docstring for classifierV1
	A simple 2 Linear Layer Classifier.

	Input: Comes from the last layer of encoder from the Autoencoder
		Number of Neurons are hard coded for the hidden features = 128.
		We can play with this number or even remove the layer and just use  1 Linear layer 
	
	Returns: Outputs for each class

	"""

	def __init__(self, input_shape=128, num_classes=10):
		super(classifierV1, self).__init__()

		self.input_shape = input_shape
		self.num_classes = num_classes

		self.fc1 = nn.Linear(in_features=self.input_shape, out_features=128)
		self.fc2 = nn.Linear(in_features=128, out_features=self.num_classes)

		self.relu = nn.ReLU(inplace=True)



	def forward(self, x):

		x = self.fc1(x)
		x = self.relu(x)

		x = self.fc2(x)
		
		return x











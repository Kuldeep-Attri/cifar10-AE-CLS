import torch
import torch.nn as nn
import torch.nn.functional as F


class classifierV3(nn.Module):
	"""
	docstring for classifierV3
	"""

	def __init__(self, in_channels=64, num_classes=10):
		super(classifierV3, self).__init__()

		self.in_channels = in_channels
		self.num_classes = num_classes

		self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=128, kernel_size=1, stride=1, padding=0)
		self.fc = nn.Linear(in_features=128, out_features=self.num_classes)

		self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1)) # Global Avg Pooling layer
		self.relu = nn.ReLU(inplace=True)



	def forward(self, x):

		x = self.conv(x)
		x = self.relu(x)
		
		x = self.gap(x) 
		# x = F.max_pool2d(x, kernel_size=x.size()[2:]) # This is GMP(global max poooling), try this instead of GAP.
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		
		return x





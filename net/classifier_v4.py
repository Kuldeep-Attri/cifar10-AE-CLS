import torch
import torch.nn as nn
import torch.nn.functional as F


class classifierV4(nn.Module):
	"""
	docstring for classifierV4
	"""

	def __init__(self, in_channels=32, num_classes=10):
		super(classifierV4, self).__init__()

		self.in_channels = in_channels
		self.num_classes = num_classes

		self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
		self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

		self.fc1 = nn.Linear(in_features=64*4*4, out_features=512)
		self.fc2 = nn.Linear(in_features=512, out_features=128)
		self.fc3 = nn.Linear(in_features=128, out_features=self.num_classes)

		self.relu = nn.ReLU(inplace=True)
		self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
		self.dropout = nn.Dropout(p=0.3)



	def forward(self, x):

		x = self.conv1(x)
		x = self.relu(x)
		x = self.conv2(x)
		x = self.relu(x)
		x = self.max_pool(x)
		x = self.conv3(x)
		x = self.relu(x)
		x = self.max_pool(x)

		x = x.view(x.size(0), -1)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.dropout(x)
		x = self.fc2(x)
		x = self.relu(x)
		x = self.dropout(x)
		x = self.fc3(x)

		return x



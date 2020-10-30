import torch
import torch.nn as nn
import torchvision.models as models


class standardArchs(nn.Module):
	"""
	docstring for standardArchs
	"""

	def __init__(self, name='vgg', num_output=10):
		super(standardArchs, self).__init__()
		
		self.name = name
		self.num_output = num_output

		if 'vgg' in self.name:
		
			vgg = models.vgg16(pretrained=True)
			
			self.block1 = nn.Sequential(
				*list(vgg.features.children())[:10]
			)
			self.block2 = nn.Sequential(
				*list(vgg.features.children())[10:17]
			)
			self.block3 = nn.Sequential(
				*list(vgg.features.children())[17:24]
			)
			self.block4 = nn.Sequential(
				*list(vgg.features.children())[24:]
			)

			self.decoder = nn.Sequential(
				nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2, padding=0),
				nn.ReLU(inplace=True),
				nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0),
				nn.ReLU(inplace=True),
				nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0),
				nn.ReLU(inplace=True),
				nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0),
				nn.ReLU(inplace=True),
				nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2, padding=0),
				# nn.Sigmoid() # nn.Tanh() # If we normalize the original image between the 0 to 1 or -1 to 1
			)

			self.classifier = nn.Sequential(
				nn.Linear(in_features=512*7*7, out_features=4096),
				nn.ReLU(inplace=True),
				nn.Dropout(),
				nn.Linear(in_features=4096, out_features=4096),
				nn.ReLU(inplace=True),
				nn.Dropout(),
				nn.Linear(4096, self.num_output)
			)

		if 'resnet' in self.name:

			resnet = models.resnet50(pretrained=True)

			self.block1 = nn.Sequential(
				resnet.conv1,
				resnet.bn1,
				resnet.relu,
				resnet.maxpool,
				*list(resnet.layer1)
			)
			self.block2 = nn.Sequential(
				*list(resnet.layer2)
			)
			self.block3 = nn.Sequential(
				*list(resnet.layer3)
			)	
			self.block4 = nn.Sequential(
				*list(resnet.layer4)
			)
			self.avgpool = nn.AdaptiveAvgPool2d(output_size(1, 1))
			self.fc = nn.Linear(in_features=2048, out_features=self.num_output)

			self.decoder = nn.Sequential(
				nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2, padding=0),
				nn.ReLU(inplace=True),
				nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0),
				nn.ReLU(inplace=True),
				nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0),
				nn.ReLU(inplace=True),
				nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2, padding=0),
				nn.ReLU(inplace=True),
				nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2, padding=0),
				# nn.Sigmoid() # nn.Tanh() # If we normalize the original image between the 0 to 1 or -1 to 1
			)


		if 'squeezenet' in self.name:

			squeezenet = models.squeezenet1_1(pretrained=True)
			
			self.encoder = squeezenet.features

			self.decoder = nn.Sequential(
				nn.ConvTranspose2d(512, 384, kernel_size=2, stride=2, padding=0),
				nn.ReLU(inplace=True),
				nn.ConvTranspose2d(384, 256, kernel_size=1, stride=1, padding=0),
				nn.ReLU(inplace=True),
				nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0),
				nn.ReLU(inplace=True),
				nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0),
				nn.ReLU(inplace=True),
				nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2, padding=0),
				# nn.Sigmoid() # nn.Tanh() # If we normalize the original image between the 0 to 1 or -1 to 1
			)

			self.classifier = nn.Sequential(
				nn.Dropout(0.5),
				nn.Conv2d(in_channels=512, out_channels=self.num_output, kernel_size=1, stride=1, padding=0),
				nn.ReLU(inplace=True),
				nn.AdaptiveAvgPool2d(output_size=(1, 1))
			)

		if 'mobilenet' in self.name:

			mobilenet = models.mobilenet_v2(pretrained=True)

			self.encoder = mobilenet.features

			self.decoder = nn.Sequential(
				nn.ConvTranspose2d(1280, 576, kernel_size=2, stride=2, padding=0),
				nn.ReLU(inplace=True),
				nn.ConvTranspose2d(576, 384, kernel_size=2, stride=2, padding=0),
				nn.ReLU(inplace=True),
				nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2, padding=0),
				nn.ReLU(inplace=True),
				nn.ConvTranspose2d(192, 64, kernel_size=2, stride=2, padding=0),
				nn.ReLU(inplace=True),
				nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2, padding=0),
				# nn.Sigmoid() # nn.Tanh() # If we normalize the original image between the 0 to 1 or -1 to 1
			)

			self.avgpool = nn.AdaptiveAvgPool2d(output_size(1, 1))
			self.classifier = nn.Sequential(
				nn.Dropout(0.2),
				nn.Linear(1280, self.num_output)
			)



	def forward(self, x):

		if self.name == 'vgg':

			x1 = self.block1(x)
			x2 = self.block2(x1)
			x3 = self.block3(x2)
			x_encoded = self.block4(x3)

			x_decoded = self.decoder(x_encoded)

			x = x_encoded.view(x_encoded.size(0), -1) # Flattening
			x = self.classifier(x)

			return x, x_decoded


		if self.name == 'resnet':

			x1 = self.block1(x)
			x2 = self.block2(x1)
			x3 = self.block3(x2)
			x4 = self.block4(x3)

			x_decoded = self.decoder(x4)

			x = self.avgpool(x4)
			x = x.view(x.size(0), -1)
			x = self.fc(x)

			return x, x_decoded


		if self.name == 'squeezenet':

			x_encoded = self.encoder(x)
			x_decoded = self.decoder(x_encoded)

			x = self.classifier(x_encoded)
			x = x.view(x.size(0), -1)

			return x, x_decoded


		if self.name == 'mobilenet':

			x_encoded = self.encoder(x)
			x_decoded = self.decoder(x_encoded)

			x = self.avgpool(x_encoded)
			x = x.view(x.size(0), -1)
			x = self.classifier(x)

			return x, x_decoded














import os
import argparse
import glob
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms

from net import modelV1, modelV2, modelV3, modelV4, standardArchs



MEAN = [0.485, 0.456, 0.406] 
STD = [0.229, 0.224, 0.225]

##############################
dev='cpu' # Default setting if no GPU, not recommended.
if torch.cuda.is_available():
	print('CUDA is available, will use GPUs :)')
	dev='cuda'
##############################


def construct_image(recon, fname, output_path, version):
	
	image = recon[0]

	if version == 'v1':
		image = image.view(3, 32, 32) # C*H*W

	image_cons = image.new(*image.size())

	image_cons[0, :, :] = image[0, :, :] * STD[0] + MEAN[0]
	image_cons[1, :, :] = image[1, :, :] * STD[1] + MEAN[1]
	image_cons[2, :, :] = image[2, :, :] * STD[2] + MEAN[2]

	image_cons = image_cons * 255

	image_cons = image_cons.cpu().detach().numpy()
	image_cons = image_cons.astype(np.uint8)

	image_cons = np.swapaxes(image_cons, 0, 1) # C*H*W --> H*C*W
	image_cons = np.swapaxes(image_cons, 1, 2) # H*C*W --> H*W*C

	im = Image.fromarray(image_cons)
	im.save(output_path + '/' + fname + '_recon.png')

	return

def img_processing(img_path, version, input_size=32):

	if input_size==32:
		loader = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])
	if input_size==224:
		loader = transforms.Compose([transforms.Scale(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])
	image = Image.open(img_path)
	image = loader(image).float()

	image = image.to(dev)
	if version == 'v1':
		image = image.view(1, -1)
	else:
		image = image.view(1, image.shape[0], image.shape[1], image.shape[2])

	return image


def get_args():

	parser = argparse.ArgumentParser(description='Constructing image for CIFAR10 using autoencoder model.')
	parser.add_argument('-m', '--model', type=str, default='models/epoch40.pth', help='path to the model pth file')
	parser.add_argument('-p', '--path', type=str, default='cifar10/examples', help='path to the dir contain sample images.')
	parser.add_argument('-o', '--output', type=str, default='output/', help='path to the output folder to store const images')
	parser.add_argument('-v', '--version', type=str, default='v1', help='which architecture you wany to use. For detail see description.txt')

	return parser.parse_args()


if __name__ == '__main__':

	args = get_args()

	if args.version == 'v1':
		model = modelV1(input_shape=3*32*32, num_classes=10).to(dev)
	elif args.version == 'v2':
		model = modelV2(num_classes=10).to(dev)
	elif args.version == 'v3':
		model = modelV3(num_classes=10).to(dev)
	elif args.version == 'v4':
		model = modelV4(num_classes=10).to(dev)
	else:
		model = standardArchs(name=args.version, num_output=10).to(dev)

	print(model)

	model.load_state_dict(torch.load(args.model))

	for file in glob.glob(args.path + '/*.png'):
		print(file)
		fname = file.split('/')[-1].split('.')[0]

		img = img_processing(file, args.version, input_size=32)

		pred, recon = model(img)

		construct_image(recon, fname, args.output, args.version)



	print('Finished Constructing Images!!!')



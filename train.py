import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from net import modelV1, modelV2, modelV3, modelV4, standardArchs
from util import data_loader, classifier_loss, autoencoder_loss


##############################
dev='cpu' # Default setting if no GPU, not recommended.
if torch.cuda.is_available():
	print('CUDA is available, will use GPUs :)')
	dev='cuda'
##############################




# Training on the Train Data for each epoch --> Training
def train_epoch(epoch, model, dataloader, optimizer, base_lr, version, alpha=1.0, beta=1.0, gamma=0.001):
	model.train()
	for param in model.parameters():  # Setting complete model to be trainable
		param.requires_grad = True

	lr =  base_lr * (0.1 ** (epoch // 10))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

	total_loss = 0.0
	total_cls_loss = 0.0
	total_auto_loss = 0.0

	for batch_idx, (data, label) in enumerate(dataloader):

		data = data.to(dev)
		if version == 'v1':
			data = data.view(data.shape[0], -1)

		label = label.to(dev)

		optimizer.zero_grad() # Resetting the grads back to zero.

		pred, recon = model(data)

		cls_loss = classifier_loss(pred, label)
		auto_loss = autoencoder_loss(recon, data)

		l2_reg = None
		for param in model.parameters():
			if l2_reg is None:
				l2_reg = 0.5 * torch.sum(param**2)
			else:
				l2_reg = l2_reg + 0.5 * torch.sum(param**2)


		loss = alpha*cls_loss + beta*auto_loss #+ gamma*l2_reg # We can play with alpha and beta value

		loss.backward() # Do backprop

		optimizer.step() # Update parameters

		total_loss += loss.item()
		total_cls_loss += alpha*cls_loss.item()
		total_auto_loss += beta*auto_loss.item()

		if batch_idx%100 == 0:
			print("batch idx: {}, total loss: {:.6f}, cls loss: {:.6f}, auto loss: {:.6f}".format(batch_idx+1, loss, cls_loss, auto_loss))


	total_loss = total_loss/len(dataloader)
	total_cls_loss = total_cls_loss/len(dataloader)
	total_auto_loss = total_auto_loss/len(dataloader)


	print("Training--> epoch number: {}, lr: {:.6f}, total loss: {:.6f}, cls loss: {:.6f}, auto loss: {:.6f}".format(epoch+1, lr, total_loss, total_cls_loss, total_auto_loss))

	return


# Testing on the Test Data for each epoch --> Validation
def test_epoch(epoch, model, dataloader, version, alpha=1.0):
	model.eval()

	total_loss = 0.0
	total_cls_loss = 0.0
	total_auto_loss = 0.0

	correct = 0
	total = 0

	with torch.no_grad():
		for batch_idx, (data, label) in enumerate(dataloader):

			data = data.to(dev)
			if version == 'v1':
				data = data.view(data.shape[0], -1)
			label = label.to(dev)

			pred, recon = model(data)
			
			cls_loss = classifier_loss(pred, label)
			auto_loss = autoencoder_loss(recon, data)

			loss = cls_loss + alpha*auto_loss

			total_loss += loss.item()
			total_cls_loss += cls_loss.item()
			total_auto_loss += auto_loss.item()

			pred = pred.data.max(1)[1]
			correct += pred.eq(label).sum()
			total += label.size(0)


	total_loss = total_loss/len(dataloader)
	total_cls_loss = total_cls_loss/len(dataloader)
	total_auto_loss = total_auto_loss/len(dataloader)

	print("Testing--> epoch number: {}, total loss: {:.6f}, cls loss: {:.6f}, auto loss: {:.6f}".format(epoch+1, total_loss, total_cls_loss, total_auto_loss))
	print('Correct--> ', float(correct), '  Total--> ', total, '  Accuracy --> ',(float(correct)/total))

	return



# Getting the Arguments.
def get_args():

	parser = argparse.ArgumentParser(description='Training for CIFAR10 classifier and autoencoder model.')
	parser.add_argument('-d', '--root_dir', type=str, default='./data', help='path to the root fodler of Cifar10.')
	parser.add_argument('-n', '--num_epochs', type=int, default=40, help='Number of epochs.')
	parser.add_argument('-e2e', '--end2end', type=bool, default=True, help='Bool for end2end training or seprate.')
	parser.add_argument('-o', '--optim', type=str, default='sgd', help='which optimizer we want to use.')
	parser.add_argument('-lr', '--base_lr', type=float, default=0.01, help='Base Learning Rate for the optimizer.')
	parser.add_argument('-v', '--version', type=str, default='v1', help='which architecture you wany to use. For detail see description.txt')

	return parser.parse_args()


# Main Method
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

	train_dataloader = data_loader(args.root_dir + '/train', batch_size=32, train=True)
	test_dataloader = data_loader(args.root_dir + '/test', batch_size=8, train=False)

	if args.optim == 'sgd':
		optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
	if args.optim == 'adam':
		optimizer = optim.Adam(model.parameters(), lr=args.base_lr)
	if args.optim == 'adagrad':
		optimizer = optim.Adagrad(model.parameters(), lr=args.base_lr)
	#We can use more if need to


	for epoch in range(args.num_epochs):
		train_epoch(epoch, model, train_dataloader, optimizer, args.base_lr, args.version)
		test_epoch(epoch, model, test_dataloader, args.version)



	torch.save(model.state_dict(), "models/epoch" + str(args.num_epochs) + '_' + str(args.optim) + '_' + str(args.version) + ".pth")

	print('Finished training!!!')











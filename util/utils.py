import torch
import torch.nn as nn


cls_criterion = nn.CrossEntropyLoss()
auto_criterion = nn.MSELoss()


def classifier_loss(pred, label):

	return cls_criterion(pred, label)


def autoencoder_loss(recon, data):

	return auto_criterion(recon, data)

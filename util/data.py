import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def data_loader(path, input_size=32, batch_size=32, num_workers=1, pin_memory=True, train=True):
	'''
		Loading both train and test dataset.
		if train: True, laod training dataset
	'''


	if input_size == 32:
		if train:
			return data.DataLoader(
				datasets.ImageFolder(path,
					transforms.Compose([
						transforms.RandomHorizontalFlip(),
						transforms.ToTensor(),
						transforms.Normalize(mean=MEAN, std=STD),
					])),
				batch_size=batch_size,
				shuffle=True,
				num_workers=num_workers,
				pin_memory=pin_memory
			)

		else:
			return data.DataLoader(
				datasets.ImageFolder(path,
					transforms.Compose([
						transforms.ToTensor(),
						transforms.Normalize(mean=MEAN, std=STD),
					])),
				batch_size=batch_size,
				shuffle=False,
				num_workers=num_workers,
				pin_memory=pin_memory
			)

	else:
		if train:
			return data.DataLoader(
				datasets.ImageFolder(path,
					transforms.Compose([
						transforms.Scale(256),
						transforms.RandomCrop(224),
						transforms.RandomHorizontalFlip(),
						transforms.ToTensor(),
						transforms.Normalize(mean=MEAN, std=STD),
					])),
				batch_size=batch_size,
				shuffle=True,
				num_workers=num_workers,
				pin_memory=pin_memory
			)

		else:
			return data.DataLoader(
				datasets.ImageFolder(path,
					transforms.Compose([
						transforms.Scale(256),
						transforms.CenterCrop(224),
						transforms.ToTensor(),
						transforms.Normalize(mean=MEAN, std=STD),
					])),
				batch_size=batch_size,
				shuffle=False,
				num_workers=num_workers,
				pin_memory=pin_memory
			)




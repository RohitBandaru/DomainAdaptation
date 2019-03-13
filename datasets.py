import torch
from torchvision.datasets import MNIST
from usps import USPS
from torchvision import datasets, transforms

mnist_tr = torch.utils.data.DataLoader(MNIST('../data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])),batch_size=64, shuffle=True)

mnist_te = torch.utils.data.DataLoader(MNIST('../data', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])),batch_size=64, shuffle=True)

usps_tr = torch.utils.data.DataLoader(USPS('../data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(28, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])),batch_size=64, shuffle=True)

usps_te = torch.utils.data.DataLoader(USPS('../data', train=False, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(28, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])),batch_size=64, shuffle=True)

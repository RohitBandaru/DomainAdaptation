import torch
from torchvision.datasets import MNIST
from usps import USPS
from torchvision import datasets, transforms

rot = 20
trans = .1

mnist_tr = torch.utils.data.DataLoader(MNIST('../data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.RandomAffine(rot, translate=(trans,trans)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])),batch_size=64, shuffle=True)

mnist_te = torch.utils.data.DataLoader(MNIST('../data', train=False, download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.RandomAffine(rot, translate=(trans,trans)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])),batch_size=64, shuffle=True)

usps_tr = torch.utils.data.DataLoader(USPS('../data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.RandomAffine(rot, translate=(trans,trans)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])),batch_size=64, shuffle=True)

usps_te = torch.utils.data.DataLoader(USPS('../data', train=False, download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.RandomAffine(rot, translate=(trans,trans)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])),batch_size=64, shuffle=True)

svhn_tr = torch.utils.data.DataLoader(datasets.SVHN('../data/SVHN/', split="train", download=True,
                transform=transforms.Compose([
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ])), batch_size=64, shuffle=True)

svhn_te = torch.utils.data.DataLoader(datasets.SVHN('../data/SVHN/', split="train", download=True,
                transform=transforms.Compose([
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ])), batch_size=64, shuffle=True)


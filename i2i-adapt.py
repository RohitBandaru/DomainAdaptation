import numpy as np
from datasets import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
hidden_dims = 500

weights = {
    "c": 1.0,
    "z": 0.05,
    "id": 0.1,
    "tr": 0.02,
    "cyc": 0.1,
    "trc": 0.1,
}

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout2d()


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.maxpool1(x))
        x = self.conv2(x)
        x = F.relu(self.maxpool2(x))
        x = self.dropout(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv1 = nn.ConvTranspose2d(50, 20, 9, 1)
        self.conv2 = nn.ConvTranspose2d(20, 10, 9, 1)
        self.conv3 = nn.ConvTranspose2d(10, 1, 9, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        return x

class ClassClassifier(nn.Module):
    def __init__(self):
        super(ClassClassifier, self).__init__()
        self.fc = nn.Linear(4*4*50,hidden_dims)
        self.fc1 = nn.Linear(hidden_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, 10)

    def forward(self, x):
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc(x))
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)

class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()

        self.fc = nn.Linear(4*4*50,hidden_dims)
        self.fc1 = nn.Linear(hidden_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, 2)

    def forward(self, x):
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc(x))
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout2d()

        self.fc = nn.Linear(4*4*50,hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.maxpool1(x))
        x = self.conv2(x)
        x = F.relu(self.maxpool2(x))
        x = self.dropout(x)

        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)

def test(encoder, classifier, test_loader, source=True):
    encoder.eval()
    classifier.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data, target
            output = classifier(encoder(data))
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    typeData = "Source" if source else "Target"
    print('\n'+typeData+' test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))),


def qid_loss(source_encoding, class_classifier, data_s, target_s):
    output = class_classifier(source_encoding)
    return F.nll_loss(output, target_s)

def train(source_encoder, source_decoder, target_encoder, target_decoder,
    class_classifier, domain_classifier, source_discriminator, target_discriminator,
    source_train_loader, target_train_loader, source_test_loader, target_test_loader,
    epochs=1):

    source_encoder.train()
    source_decoder.train()
    target_encoder.train()
    target_decoder.train()
    class_classifier.train()
    domain_classifier.train()
    source_discriminator.train()
    target_discriminator.train()

    optimizer = optim.SGD(
        list(source_encoder.parameters()) + list(source_decoder.parameters()) +
        list(target_encoder.parameters()) + list(target_decoder.parameters()) +
        list(class_classifier.parameters()) + list(domain_classifier.parameters())
        , lr=.1, momentum=.1)

    for i in range(epochs):

        data_zip = zip(target_train_loader, source_train_loader)
        for batch_idx, ((data_t, _), (data_s, target_s)) in enumerate(data_zip):
            source_encoding = source_encoder(data_s)
            source_decoding = source_decoder(source_encoding)

            target_encoding = target_encoder(data_t)
            target_decoding = target_decoder(target_encoding)

            source_domain = domain_classifier(source_encoding)
            target_domain = domain_classifier(target_encoding)

            source_translation = target_decoder(source_encoding)
            target_translation = source_decoder(target_encoding)
            source_translation_d = target_discriminator(source_translation)
            target_translation_d = source_discriminator(target_translation)

            source_target_encoding = target_encoder(source_translation)
            source_cycle = source_decoder(source_target_encoding)
            target_cycle = target_decoder(source_encoder(target_translation))

            # 1 source classification
            c_loss = weights["c"] * qid_loss(source_encoding, class_classifier, data_s, target_s)

            # 2 decoder
            id_loss = weights["id"] * nn.MSELoss()(data_s, source_decoding) + nn.MSELoss()(data_t, target_decoding)

            # 3 domain classification loss
            z_out = torch.cat((Variable(source_domain), Variable(target_domain)),0)
            z_label = torch.cat((Variable(torch.ones(source_domain.size()[0]).long()),
                             Variable(torch.zeros(target_domain.size()[0]).long())),0)
            z_loss = weights["z"] * F.nll_loss(z_out.float().requires_grad_(), z_label)

            # 4 translation discriminate
            tr_out = torch.cat((Variable(source_translation_d), Variable(target_translation_d)),0)
            tr_label = torch.cat((Variable(torch.ones(source_translation_d.size()[0]).long()),
                             Variable(torch.zeros(target_translation_d .size()[0]).long())),0)
            tr_loss = weights["tr"] * F.nll_loss(tr_out.float().requires_grad_(), tr_label)

            # 5 Cycle consistency
            cyc_loss = weights["cyc"] * nn.MSELoss()(data_s, source_cycle) + nn.MSELoss()(data_t, target_cycle)

            # 6 Source to target classification loss
            output_trc = class_classifier(source_target_encoding)
            trc_loss = weights["trc"] * F.nll_loss(output_trc, target_s)

            loss = c_loss + id_loss + z_loss + tr_loss + cyc_loss + trc_loss
            loss.backward()
            optimizer.step()

    return source_encoder, source_decoder, target_encoder, target_decoder, class_classifier, domain_classifier

def load_mnist():
    transform = transforms.Compose([
                            transforms.Resize(size=(32,32)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,)),
                            ])
    trainset = datasets.MNIST('data/MNIST/', download=True, train=True, transform=transform)
    testset = datasets.MNIST('data/MNIST/', download=True, train=False, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True)

    return trainloader, testloader

def load_svhn():
    transform = transforms.Compose([
                            transforms.Grayscale(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,)),
                            ])
    trainset = datasets.SVHN('data/SVHN/', download=True, split="train", transform=transform)
    testset = datasets.SVHN('data/SVHN/', download=True, split="test", transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True)

    return trainloader, testloader

if __name__ == '__main__':
    source_encoder = Encoder()
    source_decoder = Decoder()

    target_encoder = Encoder()
    target_decoder = Decoder()

    class_classifier = ClassClassifier()
    domain_classifier = DomainClassifier()

    source_discriminator = Discriminator()
    target_discriminator = Discriminator()

    mnist_tr, mnist_te = load_mnist()
    mnist_te, usps_te = load_svhn()

    source_encoder, source_decoder, target_encoder, target_decoder, class_classifier, domain_classifier = train(source_encoder, source_decoder, target_encoder, target_decoder,
        class_classifier, domain_classifier, source_discriminator, target_discriminator,
        mnist_tr, usps_tr, mnist_te, usps_te)


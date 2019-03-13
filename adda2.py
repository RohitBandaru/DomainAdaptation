import numpy as np
from datasets import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d()
        self.fc = nn.Linear(4*4*50,500)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.maxpool1(x))
        x = self.conv2(x)
        x = F.relu(self.maxpool2(x))
        x = self.dropout(x)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc(x))
        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(500, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
                                nn.Linear(500, 500),
                                nn.ReLU(),
                                nn.Linear(500, 500),
                                nn.ReLU(),
                                nn.Linear(500, 2))

    def forward(self, x):
        return F.log_softmax(self.discriminator(x), dim=1)

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

def pretrain(encoder, classifier, source_train_loader, source_test_loader, target_test_loader):
    pretrain_epochs = 5

    encoder.train()
    classifier.train()

    pretrain_optimizer = optim.SGD(list(encoder.parameters()) +
                                         list(classifier.parameters()), lr=.1, momentum=.1)
    # Step 1 pretrain
    for i in range(pretrain_epochs):
        for batch_idx, (data, target) in enumerate(source_train_loader):
            data, target = data, target
            pretrain_optimizer.zero_grad()
            output = classifier(encoder(data).detach())
            classifcation_loss = F.nll_loss(output, target)
            classifcation_loss.backward()
            pretrain_optimizer.step()
            if batch_idx % 100 == 0:
                print('Pre-Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    i, batch_idx * len(data), len(source_train_loader.dataset),
                    100. * batch_idx / len(source_train_loader), classifcation_loss.item()))


    # Step 2 discriminator
    print("PRETRAIN")
    test(encoder, classifier, source_test_loader, True)
    test(encoder, classifier, target_test_loader, False)
    return encoder, classifier


def adapt(source_encoder, target_encoder, discriminator, source_train_loader, target_train_loader, source_test_loader, target_test_loader):
    optimizer_g = optim.Adam(target_encoder.parameters(), lr=1e-4, betas=(0.5,0.9))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5,0.9))

    target_encoder.train()
    discriminator.train()
    adapt_epochs = 10
    for i in range(3):

        data_zip = zip(target_train_loader, source_train_loader)
        for batch_idx, ((data_t, _), (data_s, _)) in enumerate(data_zip):
            # d loss
            data_s = Variable(data_s)
            data_t = Variable(data_t)

            encoding_s = source_encoder(data_s)
            encoding_t = target_encoder(data_t)

            discrim_s = discriminator(encoding_s)
            discrim_t = discriminator(encoding_t)

            encoding_concat = torch.cat((encoding_s, encoding_t), 0)
            discrim_out = discriminator(encoding_concat.detach())

            #discrim_out = torch.cat((Variable(discrim_s), Variable(discrim_t)),0)
            discrim_label = torch.cat((Variable(torch.ones(discrim_s.size()[0]).long()),
                             Variable(torch.zeros(discrim_t.size()[0]).long())),0)

            optimizer_d.zero_grad()
            d_loss = F.nll_loss(discrim_out, discrim_label)
            discriminator.zero_grad()
            d_loss.backward()
            optimizer_d.step()

    for i in range(adapt_epochs):

        data_zip = zip(target_train_loader, source_train_loader)
        for batch_idx, ((data_t, _), (data_s, _)) in enumerate(data_zip):
            # d loss
            data_s = Variable(data_s)
            data_t = Variable(data_t)

            encoding_s = source_encoder(data_s)
            encoding_t = target_encoder(data_t)

            discrim_s = discriminator(encoding_s)
            discrim_t = discriminator(encoding_t)

            encoding_concat = torch.cat((encoding_s, encoding_t), 0)
            discrim_out = discriminator(encoding_concat.detach())

            #discrim_out = torch.cat((Variable(discrim_s), Variable(discrim_t)),0)
            discrim_label = torch.cat((Variable(torch.ones(discrim_s.size()[0]).long()),
                             Variable(torch.zeros(discrim_t.size()[0]).long())),0)

            optimizer_d.zero_grad()
            d_loss = F.nll_loss(discrim_out, discrim_label)
            discriminator.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            # g loss
            discrim_t = discriminator(encoding_t)
            optimizer_g.zero_grad()
            g_loss = F.nll_loss(discrim_t, Variable(torch.zeros(discrim_t.size()[0]).long()))
            target_encoder.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            if batch_idx % 100 == 0:
                print('Adapt Epoch: {} Discriminator Loss: {:.6f}\tGenerator Loss: {:.6f}'.format(
                    i, d_loss.item(), g_loss.item()))
                #test(target_encoder, classifier, target_test_loader, False)

    print("ADAPTED")
    test(source_encoder, classifier, source_test_loader, True)
    test(target_encoder, classifier, target_test_loader, False)

    return target_encoder, discriminator


if __name__ == '__main__':
    source_encoder = Encoder()
    classifier = Classifier()
    source_encoder, classifier, pretrain(source_encoder, classifier, mnist_tr, mnist_te, usps_te)
    # copy weights
    target_encoder = Encoder()
    target_encoder.load_state_dict(source_encoder.state_dict())
    discriminator = Discriminator()
    adapt(source_encoder, target_encoder, discriminator, mnist_tr, usps_tr, mnist_te, usps_te)

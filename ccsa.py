import numpy as np
from datasets import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.init import xavier_normal_
hidden_dims = 500

'''
One net
'''

def weights_init(m):
    if isinstance(m,nn.Conv2d):
        xavier_normal_(m.weight.data)
        m.bias.data.fill_(0)
    elif isinstance(m,nn.Linear):
        xavier_normal_(m.weight.data)
        m.bias.data.fill_(0)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d()
        self.fc = nn.Linear(4*4*50,hidden_dims)

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

        self.fc1 = nn.Linear(hidden_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
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


def maximum_mean_discrepancy(x, y):
    cost = torch.mean(x,0) - torch.mean(y,0)
    cost = torch.mean(torch.abs(cost))
    return cost

def contrastive_loss(y_true, y_pred):
    margin = 1
    return torch.mean(y_true * torch.square(y_pred) + (1 - y_true) * torch.square(torch.maximum(margin - y_pred, 0)))

def train(encoder, classifier, source_train_loader, target_train_loader, source_test_loader, target_test_loader):
    optimizer = optim.SGD(list(encoder.parameters()) +
                                         list(classifier.parameters()), lr=.1, momentum=.1)
    encoder.train()
    classifier.train()
    train_epochs = 3

    alpha = 0.35
    for i in range(train_epochs):

        data_zip = zip(target_train_loader, source_train_loader)
        for batch_idx, ((data_t, target_t), (data_s, target_s)) in enumerate(data_zip):
            # d loss
            data_s = Variable(data_s)
            data_t = Variable(data_t)

            encod_s = encoder(data_s)
            encod_t = encoder(data_t)

            output = classifier(encod_s)
            optimizer.zero_grad()
            classification_loss = F.nll_loss(output, target_s)
            discrim_loss = maximum_mean_discrepancy(encod_s, encod_t)

            loss = alpha*classification_loss + (1-alpha)*discrim_loss
            loss.backward()
            optimizer.step()

        print('Train Epoch: {} \tLoss: {:.6f}\tDLoss: {:.6f}'.format(
            i, classification_loss.item(), discrim_loss.item()))

    test(encoder, classifier, source_test_loader, True)
    test(encoder, classifier, target_test_loader, False)

    return encoder, classifier


if __name__ == '__main__':
    encoder = Encoder()
    classifier = Classifier()
    encoder, classifier = train(encoder, classifier, mnist_tr, usps_tr, mnist_te, usps_te)

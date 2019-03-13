import numpy as np
from datasets import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

def init_weights(layer):
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.source_encoder = nn.Sequential(nn.Conv2d(1, 20, 5, 1),
                                  nn.MaxPool2d(2, 2),
                                  nn.ReLU(),
                                  nn.Conv2d(20, 50, 5, 1),
                                  nn.Dropout2d(),
                                  nn.MaxPool2d(2, 2),
                                  nn.ReLU())

        self.target_encoder = nn.Sequential(nn.Conv2d(1, 20, 5, 1),
                                  nn.MaxPool2d(2, 2),
                                  nn.ReLU(),
                                  nn.Conv2d(20, 50, 5, 1),
                                  nn.Dropout2d(),
                                  nn.MaxPool2d(2, 2),
                                  nn.ReLU())

        self.discriminator = nn.Sequential(
                                nn.Linear(4*4*50, 500),
                                nn.ReLU(),
                                nn.Linear(500, 500),
                                nn.ReLU(),
                                nn.Linear(500, 2))

        self.classifier = nn.Sequential(
                                nn.Linear(4*4*50, 100),
                                nn.ReLU(),
                                nn.Dropout2d(),
                                nn.Linear(100, 10))


    def forward_source(self, x):
        x = self.source_encoder(x)
        x_en = x.view(-1, 4*4*50)
        x = F.relu(self.classifier(x_en))
        return x_en, F.log_softmax(x, dim=1)

    def forward_target(self, x):
        x = self.target_encoder(x)
        x_en = x.view(-1, 4*4*50)
        x = F.relu(self.classifier(x_en))
        return x_en, F.log_softmax(x, dim=1)

    def forward_source_D(self, x):
        x = self.source_encoder(x)
        x_en = x.view(-1, 4*4*50)
        x = F.log_softmax(self.discriminator(x_en), dim=1)
        return x_en, x

    def forward_target_D(self, x):
        x = self.target_encoder(x)
        x_en = x.view(-1, 4*4*50)
        x = F.log_softmax(self.discriminator(x_en), dim=1)
        return x_en, x

def test(model, test_loader, source=True):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data, target
            _ , output = model.forward_source(data) if source else model.forward_target(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    typeData = "Source" if source else "Target"
    print('\n'+typeData+' test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))),

def train(model, source_train_loader, target_train_loader, source_test_loader, target_test_loader):
    pretrain_epochs = 1
    adapt_epochs = 2

    model.train()
    pretrain_optimizer = optim.SGD(list(model.source_encoder.parameters()) +
                                         list(model.classifier.parameters()), lr=.1, momentum=.1)
    # Step 1 pretrain
    for i in range(pretrain_epochs):
        for batch_idx, (data, target) in enumerate(source_train_loader):
            data, target = data, target
            pretrain_optimizer.zero_grad()
            _, output = model.forward_source(data)
            classifcation_loss = F.nll_loss(output, target)
            classifcation_loss.backward()
            pretrain_optimizer.step()
            if batch_idx % 10 == 0:
                print('Pre-Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    i, batch_idx * len(data), len(source_train_loader.dataset),
                    100. * batch_idx / len(source_train_loader), classifcation_loss.item()))

    # copy weights
    model.target_encoder.load_state_dict(model.source_encoder.state_dict())

    # Step 2 discriminator
    print("PRETRAIN")
    test(model, source_test_loader, True)
    test(model, target_test_loader, False)
    #   return model

    #def adapt(model, source_train_loader, target_train_loader, source_test_loader, target_test_loader):
    optimizer_g = optim.Adam(model.target_encoder.parameters(), lr=1e-4, betas=(0.5,0.9))
    optimizer_d = optim.Adam(model.discriminator.parameters(), lr=1e-4, betas=(0.5,0.9))

    for i in range(adapt_epochs):

        data_zip = zip(target_train_loader, source_train_loader)
        for batch_idx, ((data_t, _), (data_s, _)) in enumerate(data_zip):
            # d loss
            output_s, discrim_s = model.forward_source_D(data_s)
            output_t, discrim_t = model.forward_target_D(data_t)

            discrim_out = torch.cat((Variable(discrim_s), Variable(discrim_t)),0)
            discrim_label = torch.cat((Variable(torch.ones(discrim_s.size()[0]).long()),
                             Variable(torch.zeros(discrim_t.size()[0]).long())),0)

            optimizer_d.zero_grad()
            d_loss = F.nll_loss(discrim_out.float().requires_grad_(), discrim_label)
            model.discriminator.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            # g loss
            optimizer_g.zero_grad()
            g_loss = F.nll_loss(discrim_t.requires_grad_(), Variable(torch.ones(discrim_t.size()[0]).long()))
            model.target_encoder.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            if batch_idx % 100 == 0:
                print('Adapt Epoch: {} Discriminator Loss: {:.6f}\tGenerator Loss: {:.6f}'.format(
                    i, d_loss.item(), g_loss.item()))
                test(model, target_test_loader, False)

    print("ADAPTED")
    test(model, source_test_loader, True)
    test(model, target_test_loader, False)


if __name__ == '__main__':
    model = Net()
    model.apply(init_weights)
    train(model, mnist_tr, usps_tr, mnist_te, usps_te)

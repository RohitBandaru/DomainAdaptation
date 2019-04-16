import torch
import torch.nn as nn
import torch.nn.functional as F

hidden_dims = 500
encoding_size = 120

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(16, encoding_size, kernel_size=(5, 5))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.maxpool1(x))
        x = self.conv2(x)
        x = F.relu(self.maxpool2(x))
        x = self.conv3(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv1 = nn.ConvTranspose2d(50, 20, 9, 1)

        # source
        self.conv2s = nn.ConvTranspose2d(20, 10, 9, 1)
        self.conv3s = nn.ConvTranspose2d(10, 1, 9, 1)

        # target
        self.conv2t = nn.ConvTranspose2d(20, 10, 9, 1)
        self.conv3t = nn.ConvTranspose2d(10, 1, 9, 1)

    def forward(self, x, source):
        x = F.relu(self.conv1(x))
        if(source):
            x = F.relu(self.conv2s(x))
            x = torch.tanh(self.conv3s(x))
        else:
            x = F.relu(self.conv2t(x))
            x = torch.tanh(self.conv3t(x))
        return x

class ClassClassifier(nn.Module):
    def __init__(self):
        super(ClassClassifier, self).__init__()
        self.fc = nn.Linear(encoding_size,hidden_dims)
        self.fc1 = nn.Linear(hidden_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, 10)

    def forward(self, x):
        x = x.view(-1, encoding_size)
        x = F.relu(self.fc(x))
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)

class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()

        self.fc = nn.Linear(encoding_size,hidden_dims)
        self.fc1 = nn.Linear(hidden_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, 2)

    def forward(self, x):
        x = x.view(-1, encoding_size)
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

        self.fc = nn.Linear(encoding_size,hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.maxpool1(x))
        x = self.conv2(x)
        x = F.relu(self.maxpool2(x))
        x = self.dropout(x)

        x = x.view(-1, encoding_size)
        x = F.relu(self.fc(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)

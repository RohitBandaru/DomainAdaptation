import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))

import numpy as np
import torch.optim as optim
from torch.autograd import Variable

losses = {
    "c": 1.0,
    "z": 0.05,
    "idx": 0.0,
    "idy": 0.0,
    "tr": 0.0,
    "cyc": 0.0,
    "trc": 0.0,
}

def train(encoder, decoder, class_classifier, domain_classifier,
    source_discriminator, target_discriminator,
    dataloaders, epochs=100):

    source_train_loader, target_train_loader, source_test_loader, target_test_loader = dataloaders

    encoder.train()
    decoder.train()
    class_classifier.train()
    domain_classifier.train()
    source_discriminator.train()
    target_discriminator.train()

    c_optimizer = optim.Adam(list(encoder.parameters()) + list(class_classifier.parameters()) +
        list(decoder.parameters()), lr=0.0002, betas=(0.5,0.999))

    zd_optimizer = optim.Adam(domain_classifier.parameters(), lr=0.0002, betas=(0.5,0.999))

    for i in range(epochs):

        data_zip = zip(source_train_loader, target_train_loader)
        for batch_idx, ((data_s, target_s), (data_t, _)) in enumerate(data_zip):

            if(use_cuda):
              data_s, target_s, data_t = data_s.cuda(), target_s.cuda(), data_t.cuda()

            if(losses['idy'] or losses['z']):
                target_encoding = encoder(data_t)

            if(losses['c'] or losses['z'] or losses['idx']):
                source_encoding = encoder(data_s)
                output = class_classifier(source_encoding)

            # train encoder
            c_optimizer.zero_grad()
            c_loss = losses['c'] * F.nll_loss(output, target_s)

            if(losses['z']):
                source_domain = domain_classifier(encoder(data_s))
                target_domain = domain_classifier(target_encoding)

                z_optimizer.zero_grad()
                z_out = torch.cat((Variable(source_domain), Variable(target_domain)),0)
                z_label = torch.cat((Variable(torch.ones(source_domain.size()[0]).long()),
                                 Variable(torch.ones(target_domain.size()[0]).long())),0)
                z_loss = losses["z"] * F.nll_loss(z_out.float().requires_grad_(), z_label.cuda())

                c_loss = z_loss + c_loss

            if(losses['idy']):
                target_decoding = decoder(target_encoding, False)
                idy_loss = losses['idy'] * nn.MSELoss()(data_t, target_decoding)
                c_loss = c_loss + idy_loss

            if(losses['idx']):
                source_decoding = decoder(source_encoding, True)
                idx_loss = losses['idx'] * nn.MSELoss()(data_s, source_decoding)
                c_loss = c_loss + idx_loss

            c_loss.backward()
            c_optimizer.step()

            # 3 domain classification loss
            if(losses['z']):
                source_domain = domain_classifier(encoder(data_s))
                target_domain = domain_classifier(encoder(data_t))

                zd_optimizer.zero_grad()
                z_out = torch.cat((Variable(source_domain), Variable(target_domain)),0)
                z_label = torch.cat((Variable(torch.ones(source_domain.size()[0]).long()),
                                 Variable(torch.zeros(target_domain.size()[0]).long())),0)
                z_loss = losses["z"] * F.nll_loss(z_out.float().requires_grad_(), z_label.cuda())

                z_loss.backward()
                zd_optimizer.step()

        if(i%1==0):
            test(encoder, class_classifier, source_test_loader, True, i)
            test(encoder, class_classifier, target_test_loader, False, i)

    return encoder, decoder, class_classifier, domain_classifier

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()

    encoder = Encoder()
    decoder = Decoder()

    class_classifier = ClassClassifier()
    domain_classifier = DomainClassifier()

    source_discriminator = Discriminator()
    target_discriminator = Discriminator()

    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

        class_classifier = class_classifier.cuda()
        domain_classifier = domain_classifier.cuda()

        source_discriminator = source_discriminator.cuda()
        target_discriminator = target_discriminator.cuda()
    # dataset tasks
    mnist2usps = (mnist_tr, usps_tr, mnist_te, usps_te)
    usps2mnist = (usps_tr, mnist_tr, usps_te, mnist_te)
    mnist2mnist = (mnist_tr, mnist_tr, mnist_te, mnist_te)
    #mnist2svhn = (mnist_tr, svhn_tr, mnist_te, svhn_te)
    #svhn2mnist = (svhn_tr, mnist_tr, svhn_te, mnist_te)

    encoder, decoder, class_classifier, domain_classifier = train(encoder, decoder,
        class_classifier, domain_classifier, source_discriminator, target_discriminator, usps2mnist)

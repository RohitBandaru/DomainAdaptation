import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from datasets import *
from model import *
from test import *

import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

losses = {
    "c": 1.0,
    "z": 0.0,
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

    c_optimizer = optim.Adam(list(encoder.parameters()) + list(class_classifier.parameters()), lr=1e-4, betas=(0.5,0.9))
    # z
    g_optimizer = optim.Adam(encoder.parameters(), lr=1e-4, betas=(0.5,0.9))
    z_optimizer = optim.Adam(encoder.parameters(), lr=1e-4, betas=(0.5,0.9))
    zd_optimizer = optim.Adam(class_classifier.parameters(), lr=1e-4, betas=(0.5,0.9))

    id_optimizer = optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=1e-4, betas=(0.5,0.9))

    for i in range(epochs):

        data_zip = zip(source_train_loader, target_train_loader)
        for batch_idx, ((data_s, target_s), (data_t, _)) in enumerate(data_zip):
            if(use_cuda):
                data_s, target_s, data_t = data_s.cuda(), target_s.cuda(), data_t.cuda()

            if(losses['c'] or losses['z'] or losses['idx']):
                source_encoding = encoder(data_s)
                output = class_classifier(source_encoding)

            if(losses['idx']):
                source_decoding = source_decoder(source_encoding)

            if(losses['idy'] or losses['z']):
                target_encoding = encoder(data_t)

            if(losses['idy']):
                target_decoding = decoder(target_encoding, False)

            if(losses['z']):
                source_domain = domain_classifier(source_encoding)
                target_domain = domain_classifier(target_encoding)
            '''
            source_translation = target_decoder(source_encoding)
            target_translation = source_decoder(target_encoding)
            source_translation_d = target_discriminator(source_translation)
            target_translation_d = source_discriminator(target_translation)

            source_target_encoding = encoder(source_translation)
            source_cycle = source_decoder(source_target_encoding)
            target_cycle = target_decoder(encoder(target_translation))
            '''
            # 1 source classification
            if(losses['c']):
                c_optimizer.zero_grad()
                c_loss = losses['c'] * F.nll_loss(output, target_s)
                c_loss = c_loss
                c_loss.backward(retain_graph=True)
                c_optimizer.step()

            # 2 decoder
            idx_loss = 0
            idy_loss = 0

            if(losses['idx']):
                idx_loss = losses['idx'] * nn.MSELoss()(data_s, source_decoding)
            if(losses['idy']):
                idy_loss = losses['idy'] * nn.MSELoss()(data_t, target_decoding)

            if(losses['idx'] or losses['idy']):
                id_loss = idx_loss + idy_loss
                id_loss.backward(retain_graph=True)
                id_optimizer.step()

            # 3 domain classification loss
            if(losses['z']):
                z_optimizer.zero_grad()
                z_out = torch.cat((Variable(source_domain), Variable(target_domain)),0)
                z_label = torch.cat((Variable(torch.ones(source_domain.size()[0]).long()),
                                 Variable(torch.zeros(target_domain.size()[0]).long())),0)
                z_loss = losses["z"] * F.nll_loss(z_out.float().requires_grad_(), z_label)

                z_loss.backward(retain_graph=True)
                z_optimizer.step()

                zd_optimizer.zero_grad()
                z_loss.backward(retain_graph=True)
                zd_optimizer.step()
            '''
            g_optimizer.zero_grad()
            z_loss2 = losses["z"] * F.nll_loss(z_out.float().requires_grad_(), z_label)
            z_loss2.backward(retain_graph=True)
            g_optimizer.step()'''
            '''
            # 4 translation discriminate
            tr_out = torch.cat((Variable(source_translation_d), Variable(target_translation_d)),0)
            tr_label = torch.cat((Variable(torch.ones(source_translation_d.size()[0]).long()),
                             Variable(torch.zeros(target_translation_d .size()[0]).long())),0)
            tr_loss = losses["tr"] * F.nll_loss(tr_out.float().requires_grad_(), tr_label)

            # 5 Cycle consistency
            cyc_loss = losses["cyc"] * nn.MSELoss()(data_s, source_cycle) + nn.MSELoss()(data_t, target_cycle)

            # 6 Source to target classification loss
            output_trc = class_classifier(source_target_encoding)
            trc_loss = losses["trc"] * F.nll_loss(output_trc, target_s)
            '''
            #loss = c_loss + z_loss#+ id_loss  + tr_loss + cyc_loss + trc_loss
            #loss.backward()
            #optimizer.step()

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
    mnist2svhn = (mnist_tr, svhn_tr, mnist_te, svhn_te)
    svhn2mnist = (svhn_tr, mnist_tr, svhn_te, mnist_te)

    encoder, decoder, class_classifier, domain_classifier = train(encoder, decoder,
        class_classifier, domain_classifier, source_discriminator, target_discriminator, mnist2usps)

# -*- coding: utf-8 -*-
# --------------------------------------------------
#
# train.py
#
# Written by cetinsamet -*- cetin.samet@metu.edu.tr
# September, 2021
# --------------------------------------------------
# The parts related to generative model are taken/adapted from https://github.com/akshitac8/tfvaegan
# --------------------------------------------------
from __future__ import print_function
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import random
# --------------------
from classifiers import *
from config import opt
from solvers import *
from data import *

FN = torch.from_numpy
F = torch.nn.functional


def print_hps():
    print('HYPER-PARAMETERS', flush=True)
    for k, v in vars(opt).items():
        print('{:30s}:{}'.format(k, v), flush=True) 
    if not opt.validation:
        print("- BE SURE THAT THE HYPERPARAMETERS (INCLUDING THE NUMBER OF ITERATIONS) ARE TUNED ON VALIDATION SET FOR THE FINAL TEST RUN.", flush=True)

def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x+1e-12, x.detach(),size_average=False)
    BCE = BCE.sum()/ x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())/ x.size(0)
    return (BCE + KLD)
           
def sample():
    batch_feature, batch_att = data.next_seen_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)

def WeightedL1(pred, gt):
    wt = (pred-gt).pow(2)
    wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0),wt.size(1))
    loss = wt * (pred-gt).abs()
    return loss.sum()/loss.size(0)
    
def generate_syn_feature(generator,classes, attribute,num,netF=None,netDec=None):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num) 
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        syn_noisev = Variable(syn_noise,volatile=True)
        syn_attv = Variable(syn_att,volatile=True)
        fake = generator(syn_noisev,c=syn_attv)
        if netF is not None:
            dec_out = netDec(fake) # only to call the forward function of decoder
            dec_hidden_feat = netDec.getLayersOutDet() #no detach layers
            feedback_out = netF(dec_hidden_feat)
            fake = generator(syn_noisev, a1=opt.a2, c=syn_attv, feedback_layers=feedback_out)
        output = fake
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label

def calc_gradient_penalty(netD,real_data, fake_data, input_att):
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates, Variable(input_att))
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty


# ------------------------------------------ #
# hyperparameters
# ------------------------------------------ #
print_hps()

# ------------------------------------------ #
# seed and cuda
# ------------------------------------------ #
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# ------------------------------------------ #
# load data
# ------------------------------------------ #
data = DATA_LOADER(opt)

# ------------------------------------------ #
# init modules of generative model
# ------------------------------------------ #
netE = Encoder(opt)
netG = Generator(opt)
netD = Discriminator(opt)
netDec = AttDec(opt,opt.attSize)
netF = None
if opt.feedback_loop == 2:
    netF = Feedback(opt)

# ------------------------------------------ #
# init tensors
# ------------------------------------------ #
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
if opt.sample_probing:
    noise_meta = torch.FloatTensor(opt.n_syn, opt.nz)
one = torch.FloatTensor([1])
mone = one * -1

# ------------------------------------------ #
# cuda
# ------------------------------------------ #
if opt.cuda:
    netD.cuda()
    netE.cuda()
    if opt.feedback_loop == 2:
        netF.cuda()
    netG.cuda()
    netDec.cuda()
    input_res = input_res.cuda()
    input_att = input_att.cuda()
    noise = noise.cuda()
    one = one.cuda()
    mone = mone.cuda()

# ------------------------------------------ #
# init optimizers
# ------------------------------------------ #
optimizer = optim.Adam(netE.parameters(), lr=opt.lr)
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
if opt.feedback_loop == 2:
    optimizerF = optim.Adam(netF.parameters(), lr=opt.feed_lr, betas=(opt.beta1, 0.999))
optimizerDec = optim.Adam(netDec.parameters(), lr=opt.dec_lr, betas=(opt.beta1, 0.999))

# ------------------------------------------ #
# track best result on validation set
# ------------------------------------------ #
if opt.validation:
    best_H = 0.
    best_iter = None

##################################################################
############################## MAIN ##############################
##################################################################
for epoch in range(opt.nepoch):
    for loop in range(opt.feedback_loop):
        for i in range(0, data.ntrain, opt.batch_size):
            # ------------------------------
            # GENERATIVE MODEL TRAINING
            # ------------------------------
            ######### Discriminator Training ##############
            for p in netD.parameters(): #unfreeze discrimator
                p.requires_grad = True
            for p in netDec.parameters(): #unfreeze deocder
                p.requires_grad = True

            # Train Discriminator and Decoder (and Decoder Discriminator)
            gp_sum = 0 #lAMBDA VARIABLE
            for iter_d in range(opt.critic_iter):
                sample()
                netD.zero_grad()          
                input_resv = Variable(input_res)
                input_attv = Variable(input_att)

                netDec.zero_grad()
                recons = netDec(input_resv)
                R_cost = opt.recons_weight*WeightedL1(recons, input_attv) 
                R_cost.backward()
                optimizerDec.step()
                criticD_real = netD(input_resv, input_attv)
                criticD_real = opt.gammaD*criticD_real.mean()
                criticD_real.backward(mone)
                if opt.encoded_noise: # --> True for CUB        
                    means, log_var = netE(input_resv, input_attv)
                    std = torch.exp(0.5 * log_var)
                    eps = torch.randn([opt.batch_size, opt.latent_size]).cpu()
                    eps = Variable(eps.cuda())
                    z = eps * std + means
                else:
                    noise.normal_(0, 1)
                    z = Variable(noise)

                if loop == 1:
                    fake = netG(z, c=input_attv)
                    dec_out = netDec(fake)
                    dec_hidden_feat = netDec.getLayersOutDet()
                    feedback_out = netF(dec_hidden_feat)
                    fake = netG(z, a1=opt.a1, c=input_attv, feedback_layers=feedback_out)
                else:
                    fake = netG(z, c=input_attv)

                criticD_fake = netD(fake.detach(), input_attv)
                criticD_fake = opt.gammaD*criticD_fake.mean()
                criticD_fake.backward(one)
                gradient_penalty = opt.gammaD*calc_gradient_penalty(netD, input_res, fake.data, input_att) 
                gp_sum += gradient_penalty.data
                gradient_penalty.backward()         
                Wasserstein_D = criticD_real - criticD_fake
                D_cost = criticD_fake - criticD_real + gradient_penalty
                optimizerD.step()

            gp_sum /= (opt.gammaD*opt.lambda1*opt.critic_iter)
            if (gp_sum > 1.05).sum() > 0:
                opt.lambda1 *= 1.1
            elif (gp_sum < 1.001).sum() > 0:
                opt.lambda1 /= 1.1

            ############# Generator Training ##############
            # Train Generator and Decoder
            for p in netD.parameters(): # freeze discrimator
                p.requires_grad = False
            if opt.recons_weight > 0 and opt.freeze_dec:
                for p in netDec.parameters(): # freeze decoder
                    p.requires_grad = False

            netE.zero_grad()
            netG.zero_grad()
            if opt.feedback_loop == 2:
                netF.zero_grad()
            input_resv = Variable(input_res)
            input_attv = Variable(input_att)
            means, log_var = netE(input_resv, input_attv)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn([opt.batch_size, opt.latent_size]).cpu()
            eps = Variable(eps.cuda())
            z = eps * std + means
            if loop == 1:
                recon_x = netG(z, c=input_attv)
                dec_out = netDec(recon_x)
                dec_hidden_feat = netDec.getLayersOutDet()
                feedback_out = netF(dec_hidden_feat)
                recon_x = netG(z, a1=opt.a1, c=input_attv, feedback_layers=feedback_out)
            else:
                recon_x = netG(z, c=input_attv)

            vae_loss_seen = loss_fn(recon_x, input_resv, means, log_var)
            errG = vae_loss_seen
            
            if opt.encoded_noise:
                criticG_fake = netD(recon_x,input_attv).mean()
                fake = recon_x 
            else:
                noise.normal_(0, 1)
                noisev = Variable(noise)
                if loop == 1:
                    fake = netG(noisev, c=input_attv)
                    dec_out = netDec(recon_x) #Feedback from Decoder encoded output
                    dec_hidden_feat = netDec.getLayersOutDet()
                    feedback_out = netF(dec_hidden_feat)
                    fake = netG(noisev, a1=opt.a1, c=input_attv, feedback_layers=feedback_out)
                else:
                    fake = netG(noisev, c=input_attv)
                criticG_fake = netD(fake,input_attv).mean()

            # ------------------------------
            # CLOSED-FORM SAMPLE PROBING
            # ------------------------------
            if opt.sample_probing:
                sample_probing_loss = 0. # init sample probing loss
                if opt.closed_form_model_type == 'eszsl':
                    eszsl = ESZSL(d_ft=opt.resSize, d_attr=opt.attSize, alpha=opt.alpha, gamma=opt.gamma) # init ESZSL
                # ------------------------------
                # META DATASET & ITERATOR INITIALIZATION
                # ------------------------------
                meta_dset = MetaDataset(data=data, opt=opt) # init meta-dataset
                meta_iterator = torch.utils.data.DataLoader(meta_dset, batch_size=1) # init meta-iterator
                # ------------------------------
                for x_probe_train, y_probe_train, x_probe_val, y_probe_val in meta_iterator:
                    # ----------------------------------------------------------------
                    # CREATE A SYNTHETIC DATASET USING PROBE TRAIN CLASSES
                    # ----------------------------------------------------------------
                    unq_probe_train_classes = FN(np.unique(y_probe_train[0].numpy()))
                    probe_train_classes_onehot = torch.diag(torch.ones(opt.n_probe_train))
                    X, Y = [], []
                    for idx, c in enumerate(unq_probe_train_classes):
                        noisev = Variable(noise_meta.normal_(0, 1).cuda()) # init noise
                        av = Variable(data.attribute[[c]].repeat(opt.n_syn, 1).cuda()) # attribute
                        if loop == 1: # loop = 1 means that feedback loop is ON
                            x_syn = netG(noisev, c=av)
                            dec_out = netDec(x_syn)
                            dec_hidden_feat = netDec.getLayersOutDet()
                            feedback_out = netF(dec_hidden_feat)
                            x_syn = netG(noisev, a1=opt.a1, c=av, feedback_layers=feedback_out)
                        else:  x_syn = netG(noisev, c=av)
                        y_syn = Variable(probe_train_classes_onehot[[idx]].repeat(opt.n_syn, 1).cuda())
                        X.append(x_syn); Y.append(y_syn) # store
                    Xv = torch.cat(X) # synthetic image features
                    Yv = torch.cat(Y) # synthetic image labels
                    # ----------------------------------------------------------------
                    # COMPUTE W WITH SYNTHETIC DATASET
                    # ----------------------------------------------------------------
                    if opt.closed_form_model_type == 'eszsl':
                        Sv = Variable(data.attribute[unq_probe_train_classes].cuda())
                    else:  Sv = Variable(data.attribute[unq_probe_train_classes].repeat(opt.n_syn, 1).cuda())
                    # compute W
                    if opt.closed_form_model_type == 'eszsl':
                        W = eszsl.find_solution(Xv, Yv, Sv) 
                    elif opt.closed_form_model_type == 'vis2sem':
                        W = VIS2SEM(Xv, Sv, lamb=opt.lamb)
                    elif opt.closed_form_model_type == 'sem2vis':
                        W = SEM2VIS(Sv, Xv, lamb=opt.lamb)
                    # ----------------------------------------------------------------
                    # COMPUTE SAMPLE PROBING LOSS WITH PROBE VALIDATION SAMPLES
                    # ----------------------------------------------------------------
                    task_loss = 0.
                    for i in range(opt.n_subset):
                        unq_probe_val_classes = FN(np.unique(y_probe_val[i].numpy()))
                        if opt.sample_probing_loss_type == 'gzsl':
                            unq_classes = torch.sort(torch.cat((unq_probe_train_classes, unq_probe_val_classes)))[0]
                        elif opt.sample_probing_loss_type == 'zsl':
                            unq_classes = unq_probe_val_classes
                        x = Variable(x_probe_val[i][0].cuda())
                        if opt.closed_form_model_type == 'eszsl':
                            s = Variable(data.attribute[unq_classes].cuda())
                            y = Variable(FN(np.searchsorted(unq_classes.numpy(), y_probe_val[i][0].numpy())).cuda())
                            y_ = eszsl.solve(x, W, s)
                            task_loss += torch.nn.functional.cross_entropy(y_, y) # compute loss
                        elif opt.closed_form_model_type == 'vis2sem':
                            s = Variable(data.attribute[y_probe_val[i][0]].cuda())
                            task_loss += vis2sem_loss(x, s, W, lamb=opt.lamb)
                        elif opt.closed_form_model_type == 'sem2vis':
                            s = Variable(data.attribute[y_probe_val[i][0]].cuda())
                            task_loss += sem2vis_loss(s, x, W, lamb=opt.lamb)
                    # ------------------------------ 
                    task_loss /= opt.n_subset # compute average loss
                    sample_probing_loss += task_loss
                    # ------------------------------
                # ------------------------------
                sample_probing_loss /= opt.n_task # compute average sample probing loss
                SLoss = opt.sample_probing_loss_weight * sample_probing_loss
                SLoss.backward() # backprop using sample probing loss
                # ------------------------------
            # ------------------------------

            G_cost = -criticG_fake
            errG += opt.gammaG*G_cost
            netDec.zero_grad()
            recons_fake = netDec(fake)
            R_cost = WeightedL1(recons_fake, input_attv)
            errG += opt.recons_weight * R_cost
            errG.backward()
            optimizer.step()
            optimizerG.step()
            if loop == 1:
                optimizerF.step()
            if opt.recons_weight > 0 and not opt.freeze_dec: # not train decoder at feedback time
                optimizerDec.step() 
        
    print('[%d/%d]  Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist:%.4f, vae_loss_seen:%.4f' % (
        epoch+1, opt.nepoch, 
        D_cost.data[0], 
        G_cost.data[0], 
        Wasserstein_D.data[0],
        vae_loss_seen.data[0]),
        end=" ", flush=True)

    # ------------------------------------------ #
    # EVALUATION
    # ------------------------------------------ #
    netG.eval()
    netDec.eval()
    if opt.feedback_loop == 2:
        netF.eval()
    syn_feature, syn_label = generate_syn_feature(netG,data.unseenclasses, data.attribute, opt.syn_num,netF=netF,netDec=netDec)

    # ------------------------------------------ #
    # Generalized Zero-Shot Learning (GZSL)
    # ------------------------------------------ #
    if opt.gzsl:   
        # Concatenate real seen features with synthesized unseen features
        train_X = torch.cat((data.train_feature, syn_feature), dim=0)
        train_Y = torch.cat((data.train_label, syn_label), dim=0)
        nclass = opt.nclass_all
        # Train GZSL classifier
        gzsl_cls = CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, \
                25, opt.syn_num, generalized=True, netDec=netDec, dec_size=opt.attSize, dec_hidden_size=4096)
        print('GZSL: seen=%.4f, unseen=%.4f, h=%.4f' % (gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H), end=" ", flush=True)

        if opt.validation:
            if gzsl_cls.H > best_H:
                best_H = gzsl_cls.H
                best_iter = epoch + 1

    # ------------------------------------------ #
    # Zero-Shot Learning (ZSL)
    # ------------------------------------------ #
    # Train ZSL classifier
    zsl_cls = CLASSIFIER(syn_feature, map_label(syn_label, data.unseenclasses), \
                    data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, \
                    generalized=False, netDec=netDec, dec_size=opt.attSize, dec_hidden_size=4096)
    acc = zsl_cls.acc
    print('ZSL: unseen accuracy=%.4f' % (acc), flush=True)
    
    # ------------------------------------------ #
    # reset G to training mode
    # ------------------------------------------ #
    netG.train()
    netDec.train()
    if opt.feedback_loop == 2:
        netF.train()
    # ------------------------------------------ #

if opt.validation:
    print("Best h-score = {:.4f} (obtained at iteration-{})".format(best_H, best_iter))
    print("- BE SURE TO SELECT THE MODEL TRAINED FOR {} ITERATIONS ON TEST SET AS FINAL MODEL.".format(best_iter))
    print(flush=True)

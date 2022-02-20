# -*- coding: utf-8 -*-
# --------------------------------------------------
#
# config.py
#
# Written by cetinsamet -*- cetin.samet@metu.edu.tr
# Parts are adapted from https://github.com/akshitac8/tfvaegan
# September, 2021
# --------------------------------------------------
import argparse


parser = argparse.ArgumentParser()

# ------------------------------------------ #
# GENERATIVE MODEL
# ------------------------------------------ #
parser.add_argument('--dataset', default='FLO', help='FLO')
parser.add_argument('--dataroot', default='../data', help='path to dataset')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=False, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=1024, help='size of the hidden units in discriminator')
parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate to train GANs ')
parser.add_argument('--feed_lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--dec_lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--encoded_noise', action='store_true', default=False, help='enables validation mode')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
parser.add_argument('--validation', action='store_true', default=False, help='enables validation mode')
parser.add_argument('--encoder_layer_sizes', type=list, default=[8192, 4096])
parser.add_argument('--decoder_layer_sizes', type=list, default=[4096, 8192])
parser.add_argument('--gammaD', type=int, default=1000, help='weight on the W-GAN loss')
parser.add_argument('--gammaG', type=int, default=1000, help='weight on the W-GAN loss')
parser.add_argument('--latent_size', type=int, default=312)
parser.add_argument('--conditional', action='store_true',default=True)
parser.add_argument('--a1', type=float, default=1.0)
parser.add_argument('--a2', type=float, default=1.0)
parser.add_argument('--recons_weight', type=float, default=1.0, help='recons_weight for decoder')
parser.add_argument('--feedback_loop', type=int, default=1, help="enable feedback loop. feedback_loop == 1: feedback loop is OFF | feedback_loop == 2: feedback loop is ON")
parser.add_argument('--freeze_dec', action='store_true', default=False, help='Freeze Decoder for fake samples')

# ------------------------------------------ #
# CLOSED-FORM SAMPLE PROBING
# ------------------------------------------ #
parser.add_argument('--sample_probing', action='store_true', default=False, help='enables sample probing')
parser.add_argument('--n_task', type=int, default=1, help='number of meta learning tasks trained in every iteration')
parser.add_argument('--n_probe_train', type=int, default=1, help='number of probe train classes')
parser.add_argument('--n_probe_val', type=int, default=1, help='number of probe val classes')
parser.add_argument('--k_probe_train', type=int, default=1, help='number of probe train class samples')
parser.add_argument('--k_probe_val', type=int, default=1, help='number of probe val class samples')
parser.add_argument('--n_subset', type=int, default=1, help='number of probe val class subsets selected for one probe train sample subset')
parser.add_argument('--alpha', type=float, help='alpha parameter of the ESZSL')
parser.add_argument('--gamma', type=float, help='gamma parameter of the ESZSL')
parser.add_argument('--lamb', type=float, help="lambda parameter of W regularizer in sem2vis and vis2sem")
parser.add_argument('--n_syn', type=int, help='number of synthetic samples generated for closed form model training')
parser.add_argument('--sample_probing_loss_weight', type=float, default=1.0, help='sample probing loss weight')
parser.add_argument('--sample_probing_loss_type', type=str, choices=['zsl', 'gzsl'], help='sample probing loss type')
parser.add_argument('--closed_form_model_type', type=str, choices=['eszsl', 'vis2sem', 'sem2vis'], help="closed-form model type")
parser.add_argument('--finetune', action='store_true', default=False, help='use finetuned features')

# ------------------------------------------ #
opt = parser.parse_args()
opt.lambda2 = opt.lambda1
opt.encoder_layer_sizes[0] = opt.resSize
opt.decoder_layer_sizes[-1] = opt.resSize
opt.latent_size = opt.attSize
# ------------------------------------------ #

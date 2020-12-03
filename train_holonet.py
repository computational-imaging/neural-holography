"""
Neural holography:

This is the main script used for training the Holonet

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

@article{Peng:2020:NeuralHolography,
author = {Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein},
title = {{Neural Holography with Camera-in-the-loop Training}},
journal = {ACM Trans. Graph. (SIGGRAPH Asia)},
year = {2020},
}

Usage
-----

$ python train_holonet.py --channel=1 --run_id=experiment_1


"""
import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import configargparse
from tensorboardX import SummaryWriter

import utils.utils as utils
import utils.perceptualloss as perceptualloss

from propagation_model import ModelPropagate
from holonet import *
from utils.augmented_image_loader import ImageLoader


# Command line argument processing
p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')
p.add_argument('--channel', type=int, default=1, help='red:0, green:1, blue:2, rgb:3')
p.add_argument('--run_id', type=str, default='', help='Experiment name', required=True)
p.add_argument('--proptype', type=str, default='ASM', help='Ideal propagation model')
p.add_argument('--generator_path', type=str, default='', help='Torch save of Holonet, start from pre-trained gen.')
p.add_argument('--model_path', type=str, default='./models', help='Torch save CITL-calibrated model')
p.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
p.add_argument('--batch_size', type=int, default=1, help='Size of minibatch')
p.add_argument('--lr', type=float, default=1e-3, help='learning rate of Holonet weights')
p.add_argument('--scale_output', type=float, default=0.95,
               help='Scale of output applied to reconstructed intensity from SLM')
p.add_argument('--loss_fun', type=str, default='vgg-low', help='Options: mse, l1, si_mse, vgg, vgg-low')
p.add_argument('--purely_unet', type=utils.str2bool, default=False, help='Use U-Net for entire estimation if True')
p.add_argument('--model_lut', type=utils.str2bool, default=True, help='Activate the lut of model')
p.add_argument('--disable_loss_amp', type=utils.str2bool, default=True, help='Disable manual amplitude loss')
p.add_argument('--num_latent_codes', type=int, default=2, help='Number of latent codes of trained prop model.')
p.add_argument('--step_lr', type=utils.str2bool, default=False, help='Use of lr scheduler')
p.add_argument('--perfect_prop_model', type=utils.str2bool, default=False,
               help='Use ideal ASM as the loss function')
p.add_argument('--manual_aberr_corr', type=utils.str2bool, default=True,
               help='Divide source amplitude manually, (possibly apply inverse zernike of primal domain')

# parse arguments
opt = p.parse_args()
channel = opt.channel
run_id = opt.run_id
loss_fun = opt.loss_fun
warm_start = opt.generator_path != ''
chan_str = ('red', 'green', 'blue')[channel]

# tensorboard setup and file naming
time_str = str(datetime.now()).replace(' ', '-').replace(':', '-')
writer = SummaryWriter(f'runs/{run_id}_{chan_str}_{time_str}')


##############
# Parameters #
##############

# units
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9

# Propagation parameters
prop_dist = (20 * cm, 20 * cm, 20 * cm)[channel]
wavelength = (638 * nm, 520 * nm, 450 * nm)[channel]
feature_size = (6.4 * um, 6.4 * um)  # SLM pitch
homography_res = (880, 1600)  # for CITL crop, see ImageLoader

# Training parameters
device = torch.device('cuda')
use_mse_init = False  # first 500 iters will be MSE regardless of loss_fun

# Image data for training
data_path = '/data/train1080'  # path for training data

if opt.model_path == '':
    opt.model_path = f'./models/{chan_str}.pth'

# resolutions need to be divisible by powers of 2 for unet
if opt.purely_unet:
    image_res = (1024, 2048)  # 10 down layers
else:
    image_res = (1072, 1920)  # 4 down layers


###############
# Load models #
###############

# re-use parameters from CITL-calibrated model for out Holonet.
if opt.perfect_prop_model:
    final_phase_num_in = 2

    # set model instance as naive ASM
    model_prop = ModelPropagate(distance=prop_dist, feature_size=feature_size, wavelength=wavelength,
                                target_field=False, num_gaussians=0, num_coeffs_fourier=0,
                                use_conv1d_mlp=False, num_latent_codes=[0],
                                norm=None, blur=None, content_field=False, proptype=opt.proptype).to(device)

    zernike_coeffs = None
    source_amplitude = None
    latent_codes = None
    u_t = None
else:
    if opt.manual_aberr_corr:
        final_phase_num_in = 2 + opt.num_latent_codes
    else:
        final_phase_num_in = 4
    blur = utils.make_kernel_gaussian(0.849, 3)

    # load camera model and set it into eval mode
    model_prop = ModelPropagate(distance=prop_dist,
                                feature_size=feature_size,
                                wavelength=wavelength,
                                blur=blur).to(device)
    model_prop.load_state_dict(torch.load(opt.model_path, map_location=device))

    # Here, we crop model parameters to match the Holonet resolution, which is slightly different from 1080p.
    # parameters for CITL model
    zernike_coeffs = model_prop.coeffs_fourier
    source_amplitude = model_prop.source_amp
    latent_codes = model_prop.latent_code
    latent_codes = utils.crop_image(latent_codes, target_shape=image_res, pytorch=True, stacked_complex=False)

    # content independent target field (See Section 5.1.1.)
    u_t_amp = utils.crop_image(model_prop.target_constant_amp, target_shape=image_res, stacked_complex=False)
    u_t_phase = utils.crop_image(model_prop.target_constant_phase, target_shape=image_res, stacked_complex=False)
    real, imag = utils.polar_to_rect(u_t_amp, u_t_phase)
    u_t = torch.complex(real, imag)

    # match the shape of forward model parameters (1072, 1920)

    # If you make it nn.Parameter, the Holonet will include these parameters in the torch.save files
    model_prop.latent_code = nn.Parameter(latent_codes.detach(), requires_grad=False)
    model_prop.target_constant_amp = nn.Parameter(u_t_amp.detach(), requires_grad=False)
    model_prop.target_constant_phase = nn.Parameter(u_t_phase.detach(), requires_grad=False)

    # But as these parameters are already in the CITL-calibrated models,
    # you can load these from those models avoiding duplicated saves.

model_prop.eval()  # ensure freezing propagation model

# create new phase generator object
if opt.purely_unet:
    phase_generator = PhaseOnlyUnet(num_features_init=32).to(device)
else:
    phase_generator = HoloNet(
        distance=prop_dist,
        wavelength=wavelength,
        zernike_coeffs=zernike_coeffs,
        source_amplitude=source_amplitude,
        initial_phase=InitialPhaseUnet(4, 16),
        final_phase_only=FinalPhaseOnlyUnet(4, 16, num_in=final_phase_num_in),
        manual_aberr_corr=opt.manual_aberr_corr,
        target_field=u_t,
        latent_codes=latent_codes,
        proptype=opt.proptype).to(device)

if warm_start:
    phase_generator.load_state_dict(torch.load(opt.generator_path, map_location=device))

phase_generator.train()  # generator to be trained


###################
# Set up training #
###################

# loss function
loss_fun = ['mse', 'l1', 'si_mse', 'vgg', 'ssim', 'vgg-low', 'l1-vgg'].index(loss_fun.lower())

if loss_fun == 0:        # MSE loss
    loss = nn.MSELoss()
elif loss_fun == 1:      # L1 loss
    loss = nn.L1Loss()
elif loss_fun == 2:      # scale invariant MSE loss
    loss = nn.MSELoss()
elif loss_fun == 3:      # vgg perceptual loss
    loss = perceptualloss.PerceptualLoss()
elif loss_fun == 5:
    loss = perceptualloss.PerceptualLoss(lambda_feat=0.025)
    loss_fun = 3

mse_loss = nn.MSELoss()

# upload to GPU
loss = loss.to(device)
mse_loss = mse_loss.to(device)

if loss_fun == 2:
    scaleLoss = torch.ones(1)
    scaleLoss = scaleLoss.to(device)
    scaleLoss.requires_grad = True

    optvars = [scaleLoss, *phase_generator.parameters()]
else:
    optvars = phase_generator.parameters()

# set aside the VGG loss for the first num_mse_epochs
if loss_fun == 3:
    vgg_loss = loss
    loss = mse_loss

# create optimizer
if warm_start:
    opt.lr /= 10
optimizer = optim.Adam(optvars, lr=opt.lr)

# loads images from disk, set to augment with flipping
image_loader = ImageLoader(data_path,
                           channel=channel,
                           batch_size=opt.batch_size,
                           image_res=image_res,
                           homography_res=homography_res,
                           shuffle=True,
                           vertical_flips=True,
                           horizontal_flips=True)

num_mse_iters = 500
num_mse_epochs = 1 if use_mse_init else 0
opt.num_epochs += 1 if use_mse_init else 0

# learning rate scheduler
if opt.step_lr:
    scheduler = optim.lr_scheduler.StepLR(optimizer, 500, 0.5)


#################
# Training loop #
#################

for i in range(opt.num_epochs):
    # switch to actual loss function from mse when desired
    if i == num_mse_epochs:
        if loss_fun == 3:
            loss = vgg_loss

    for k, target in enumerate(image_loader):
        # cap iters on the mse epoch(s)
        if i < num_mse_epochs and k == num_mse_iters:
            break

        # get target image
        target_amp, target_res, target_filename = target
        target_amp = target_amp.to(device)

        # cropping to desired region for loss
        if homography_res is not None:
            target_res = homography_res
        else:
            target_res = target_res[0]  # use resolution of first image in batch

        optimizer.zero_grad()

        # forward model
        slm_amp, slm_phase = phase_generator(target_amp)
        output_complex = model_prop(slm_phase)

        if loss_fun == 2:
            output_complex = output_complex * scaleLoss

        output_lin_intensity = torch.sum(output_complex.abs()**2 * opt.scale_output, dim=1, keepdim=True)

        output_amp = torch.pow(output_lin_intensity, 0.5)

        # VGG assumes RGB input, we just replicate
        if loss_fun == 3:
            target_amp = target_amp.repeat(1, 3, 1, 1)
            output_amp = output_amp.repeat(1, 3, 1, 1)

        # ignore the cropping and do full-image loss
        if 'nocrop' in run_id:
            loss_nocrop = loss(output_amp, target_amp)

        # crop outputs to the region we care about
        output_amp = utils.crop_image(output_amp, target_res, stacked_complex=False)
        target_amp = utils.crop_image(target_amp, target_res, stacked_complex=False)

        # force equal mean amplitude when checking loss
        if 'force_scale' in run_id:
            print('scale forced equal', end=' ')
            with torch.no_grad():
                scaled_out = output_amp * target_amp.mean() / output_amp.mean()
            output_amp = output_amp + (scaled_out - output_amp).detach()

        # loss and optimize
        loss_main = loss(output_amp, target_amp)
        if warm_start or opt.disable_loss_amp:
            loss_amp = 0
        else:
            # extra loss term to encourage uniform amplitude at SLM plane
            # helps some networks converge properly initially
            loss_amp = mse_loss(slm_amp.mean(), slm_amp)

        loss_val = ((loss_nocrop if 'nocrop' in run_id else loss_main)
                    + 0.1 * loss_amp)
        loss_val.backward()
        optimizer.step()

        if opt.step_lr:
            scheduler.step()

        # print and output to tensorboard
        ik = k + i * len(image_loader)
        if use_mse_init and i >= num_mse_epochs:
            ik += num_mse_iters - len(image_loader)
        print(f'iteration {ik}: {loss_val.item()}')

        with torch.no_grad():
            writer.add_scalar('Loss', loss_val, ik)

            if ik % 50 == 0:
                # write images and loss to tensorboard
                writer.add_image('Target Amplitude', target_amp[0, ...], ik)

                # normalize reconstructed amplitude
                output_amp0 = output_amp[0, ...]
                maxVal = torch.max(output_amp0)
                minVal = torch.min(output_amp0)
                tmp = (output_amp0 - minVal) / (maxVal - minVal)
                writer.add_image('Reconstruction Amplitude', tmp, ik)

                # normalize SLM phase
                writer.add_image('SLM Phase', (slm_phase[0, ...] + math.pi) / (2 * math.pi), ik)

            if loss_fun == 2:
                writer.add_scalar('Scale factor', scaleLoss, ik)

    # save trained model
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    torch.save(phase_generator.state_dict(),
               f'checkpoints/{run_id}_{chan_str}_{time_str}_{i+1}.pth')

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 20:37:33 2023

@author: varsh
"""

"""
Neural holography:

This is the main executive script used for the phase generation using Holonet/UNET or
                                                     optimization using (GS/DPAC/SGD) + camera-in-the-loop (CITL).

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

-----

$ python main.py --channel=0 --algorithm=HOLONET --root_path=./phases --generator_dir=./pretrained_models
"""

import os
#import sys
import cv2
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft
import torch.nn.functional as func
import torch.nn.modules.loss as ll

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import configargparse
import random
import numpy as np
from imageio import imread
from skimage.transform import resize
#from torch.utils.tensorboard import SummaryWriter

import utils.utils as utils
#from utils.augmented_image_loader import ImageLoader
#from propagation_model import ModelPropagate
#from utils.modules import SGD, GS #, PhysicalProp
#from propagation_ASM import propagation_ASM



def propagation_ASM(u_in, feature_size, wavelength, z, linear_conv=True,
                    padtype='zero', return_H=False, precomped_H=None,
                    return_H_exp=False, precomped_H_exp=None,
                    dtype=torch.float32):
    """Propagates the input field using the angular spectrum method

    Inputs
    ------
    u_in: PyTorch Complex tensor (torch.cfloat) of size (num_images, 1, height, width) -- updated with PyTorch 1.7.0
    feature_size: (height, width) of individual holographic features in m
    wavelength: wavelength in m
    z: propagation distance
    linear_conv: if True, pad the input to obtain a linear convolution
    padtype: 'zero' to pad with zeros, 'median' to pad with median of u_in's
        amplitude
    return_H[_exp]: used for precomputing H or H_exp, ends the computation early
        and returns the desired variable
    precomped_H[_exp]: the precomputed value for H or H_exp
    dtype: torch dtype for computation at different precision

    Output
    ------
    tensor of size (num_images, 1, height, width, 2)
    """

    if linear_conv:
        # preprocess with padding for linear conv.
        input_resolution = u_in.size()[-2:]
        conv_size = [i * 2 for i in input_resolution]
        if padtype == 'zero':
            padval = 0
        elif padtype == 'median':
            padval = torch.median(torch.pow((u_in**2).sum(-1), 0.5))
        u_in = utils.pad_image(u_in, conv_size, padval=padval, stacked_complex=False)

    if precomped_H is None and precomped_H_exp is None:
        # resolution of input field, should be: (num_images, num_channels, height, width, 2)
        field_resolution = u_in.size()

        # number of pixels
        num_y, num_x = field_resolution[2], field_resolution[3]

        # sampling inteval size
        dy, dx = feature_size

        # size of the field
        y, x = (dy * float(num_y), dx * float(num_x))

        # frequency coordinates sampling
        fy = np.linspace(-1 / (2 * dy) + 0.5 / (2 * y), 1 / (2 * dy) - 0.5 / (2 * y), num_y)
        fx = np.linspace(-1 / (2 * dx) + 0.5 / (2 * x), 1 / (2 * dx) - 0.5 / (2 * x), num_x)

        # momentum/reciprocal space
        FX, FY = np.meshgrid(fx, fy)

        # transfer function in numpy (omit distance)
        HH = 2 * math.pi * np.sqrt(1 / wavelength**2 - (FX**2 + FY**2))

        # create tensor & upload to device (GPU)
        H_exp = torch.tensor(HH, dtype=dtype).to(u_in.device)

        ###
        # here one may iterate over multiple distances, once H_exp is uploaded on GPU

        # reshape tensor and multiply
        H_exp = torch.reshape(H_exp, (1, 1, *H_exp.size()))

    # handle loading the precomputed H_exp value, or saving it for later runs
    elif precomped_H_exp is not None:
        H_exp = precomped_H_exp

    if precomped_H is None:
        # multiply by distance
        H_exp = torch.mul(H_exp, z)

        # band-limited ASM - Matsushima et al. (2009)
        fy_max = 1 / np.sqrt((2 * z * (1 / y))**2 + 1) / wavelength
        fx_max = 1 / np.sqrt((2 * z * (1 / x))**2 + 1) / wavelength
        H_filter = torch.tensor(((np.abs(FX) < fx_max) & (np.abs(FY) < fy_max)).astype(np.uint8), dtype=dtype)

        # get real/img components
        H_real, H_imag = utils.polar_to_rect(H_filter.to(u_in.device), H_exp)

        H = torch.stack((H_real, H_imag), 4)
        H = utils.ifftshift(H)
        H = torch.view_as_complex(H)
    else:
        H = precomped_H

    # return for use later as precomputed inputs
    if return_H_exp:
        return H_exp
    if return_H:
        return H

    # For who cannot use Pytorch 1.7.0 and its Complex tensors support:
    # # angular spectrum
    # U1 = torch.fft(utils.ifftshift(u_in), 2, True)
    #
    # # convolution of the system
    # U2 = utils.mul_complex(H, U1)
    #
    # # Fourier transform of the convolution to the observation plane
    # u_out = utils.fftshift(torch.ifft(U2, 2, True))

    U1 = torch.fft.fftn(utils.ifftshift(u_in), dim=(-2, -1), norm='ortho')

    U2 = H * U1

    u_out = utils.fftshift(torch.fft.ifftn(U2, dim=(-2, -1), norm='ortho'))

    if linear_conv:
        # return utils.crop_image(u_out, input_resolution) # using stacked version
        return utils.crop_image(u_out, input_resolution, pytorch=True, stacked_complex=False)  # using complex tensor
    else:
        return u_out

def gerchberg_saxton(init_phase, target_amp, num_iters, prop_dist, wavelength, feature_size=6.4e-6,
                     phase_path=None, prop_model='ASM', propagator=None,
                     writer=None, dtype=torch.float32, precomputed_H_f=None, precomputed_H_b=None):
    """
    Given the initial guess, run the SGD algorithm to calculate the optimal phase pattern of spatial light modulator

    :param init_phase: a tensor, in the shape of (1,1,H,W), initial guess for the phase.
    :param target_amp: a tensor, in the shape of (1,1,H,W), the amplitude of the target image.
    :param num_iters: the number of iterations to run the GS.
    :param prop_dist: propagation distance in m.
    :param wavelength: wavelength in m.
    :param feature_size: the SLM pixel pitch, in meters, default 6.4e-6
    :param phase_path: path to save the results.
    :param prop_model: string indicating the light transport model, default 'ASM'. ex) 'ASM', 'fresnel', 'model'
    :param propagator: predefined function or model instance for the propagation.
    :param writer: tensorboard writer
    :param dtype: torch datatype for computation at different precision, default torch.float32.
    :param precomputed_H_f: A Pytorch complex64 tensor, pre-computed kernel for forward prop (SLM to image)
    :param precomputed_H_b: A Pytorch complex64 tensor, pre-computed kernel for backward propagation (image to SLM)

    Output
    ------
    :return: a tensor, the optimized phase pattern at the SLM plane, in the shape of (1,1,H,W)
    """

    # initial guess; random phase
    real, imag = utils.polar_to_rect(torch.ones_like(init_phase), init_phase)
    slm_field = torch.complex(real, imag)

    # run the GS algorithm
    for k in range(num_iters):
        print(k)
        # SLM plane to image plane
        recon_field = utils.propagate_field(slm_field, propagator, prop_dist, wavelength, feature_size,
                                            prop_model, dtype, precomputed_H_f)

        # write to tensorboard / write phase image
        # Note that it takes 0.~ s for writing it to tensorboard
        if False:#k > 0 and k % 10 == 0:
            utils.write_gs_summary(slm_field, recon_field, target_amp, k, writer, prefix='test')

        # replace amplitude at the image plane
        recon_field = utils.replace_amplitude(recon_field, target_amp)

        # image plane to SLM plane
        slm_field = utils.propagate_field(recon_field, propagator, -prop_dist, wavelength, feature_size,
                                          prop_model, dtype, precomputed_H_b)

        # amplitude constraint at the SLM plane
        slm_field = utils.replace_amplitude(slm_field, torch.ones_like(target_amp))

    # return phases
    return slm_field.angle()


# 2. SGD
def stochastic_gradient_descent(init_phase, target_amp, num_iters, prop_dist, wavelength, feature_size,
                                roi_res=None, phase_path=None, prop_model='ASM', propagator=None,
                                loss=nn.MSELoss(), lr=0.01, lr_s=0.003, s0=1.0,
                                writer=None, dtype=torch.float32, precomputed_H=None):

    """
    Given the initial guess, run the SGD algorithm to calculate the optimal phase pattern of spatial light modulator.

    Input
    ------
    :param init_phase: a tensor, in the shape of (1,1,H,W), initial guess for the phase.
    :param target_amp: a tensor, in the shape of (1,1,H,W), the amplitude of the target image.
    :param num_iters: the number of iterations to run the SGD.
    :param prop_dist: propagation distance in m.
    :param wavelength: wavelength in m.
    :param feature_size: the SLM pixel pitch, in meters, default 6.4e-6
    :param roi_res: a tuple of integer, region of interest, like (880, 1600)
    :param phase_path: a string, for saving intermediate phases
    :param prop_model: a string, that indicates the propagation model. ('ASM' or 'MODEL')
    :param propagator: predefined function or model instance for the propagation.
    :param loss: loss function, default L2
    :param lr: learning rate for optimization variables
    :param lr_s: learning rate for learnable scale
    :param s0: initial scale
    :param writer: Tensorboard writer instance
    :param dtype: default torch.float32
    :param precomputed_H: A Pytorch complex64 tensor, pre-computed kernel shape of (1,1,2H,2W) for fast computation.

    Output
    ------
    :return: a tensor, the optimized phase pattern at the SLM plane, in the shape of (1,1,H,W)
    """

    device = init_phase.device
    s = torch.tensor(s0, requires_grad=True, device=device)

    # phase at the slm plane
    slm_phase = init_phase.requires_grad_(True)

    # optimization variables and adam optimizer
    optvars = [{'params': slm_phase}]
    if lr_s > 0:
        optvars += [{'params': s, 'lr': lr_s}]
    optimizer = optim.Adam(optvars, lr=lr)

    # crop target roi
    target_amp = utils.crop_image(target_amp, roi_res, stacked_complex=False)

    # run the iterative algorithm
    for k in range(num_iters):
        print(k)
        optimizer.zero_grad()
        # forward propagation from the SLM plane to the target plane
        real, imag = utils.polar_to_rect(torch.ones_like(slm_phase), slm_phase)
        slm_field = torch.complex(real, imag)

        recon_field = utils.propagate_field(slm_field, propagator, prop_dist, wavelength, feature_size,
                                            prop_model, dtype, precomputed_H)

        # get amplitude
        recon_amp = recon_field.abs()

        # crop roi
        recon_amp = utils.crop_image(recon_amp, target_shape=roi_res, stacked_complex=False)


        out_amp = recon_amp

        # calculate loss and backprop
        lossValue = loss(s * out_amp, target_amp)
        lossValue.backward()
        optimizer.step()

        # write to tensorboard / write phase image
        # Note that it takes 0.~ s for writing it to tensorboard
        # with torch.no_grad():
        #     if k % 50 == 0:
        #         utils.write_sgd_summary(slm_phase, out_amp, target_amp, k,
        #                                 writer=writer, path=phase_path, s=s, prefix='test')

    return slm_phase

# 3. My SGD
def stochastic_gradient_descent1(init_phase, target_amp, num_iters, prop_dist, wavelength, feature_size,
                                roi_res=None, phase_path=None, prop_model='ASM', propagator=None,
                                loss=nn.MSELoss(), lr=0.01, lr_s=0.003, s0=1.0,
                                writer=None, dtype=torch.float32, precomputed_H=None):

    """
    Given the initial guess, run the SGD algorithm to calculate the optimal phase pattern of spatial light modulator.

    Input
    ------
    :param init_phase: a tensor, in the shape of (1,1,H,W), initial guess for the phase.
    :param target_amp: a tensor, in the shape of (1,1,H,W), the amplitude of the target image.
    :param num_iters: the number of iterations to run the SGD.
    :param prop_dist: propagation distance in m.
    :param wavelength: wavelength in m.
    :param feature_size: the SLM pixel pitch, in meters, default 6.4e-6
    :param roi_res: a tuple of integer, region of interest, like (880, 1600)
    :param phase_path: a string, for saving intermediate phases
    :param prop_model: a string, that indicates the propagation model. ('ASM' or 'MODEL')
    :param propagator: predefined function or model instance for the propagation.
    :param loss: loss function, default L2
    :param lr: learning rate for optimization variables
    :param lr_s: learning rate for learnable scale
    :param s0: initial scale
    :param writer: Tensorboard writer instance
    :param dtype: default torch.float32
    :param precomputed_H: A Pytorch complex64 tensor, pre-computed kernel shape of (1,1,2H,2W) for fast computation.

    Output
    ------
    :return: a tensor, the optimized phase pattern at the SLM plane, in the shape of (1,1,H,W)
    """

    device = init_phase.device
    s = torch.tensor(s0, requires_grad=True, device=device)

    # phase at the slm plane
    slm_phase = init_phase.requires_grad_(True)

    # optimization variables and adam optimizer
    optvars = [{'params': slm_phase}]
    if lr_s > 0:
        optvars += [{'params': s, 'lr': lr_s}]
    optimizer = optim.Adam(optvars, lr=lr)

    # crop target roi
    target_amp = utils.crop_image(target_amp, roi_res, stacked_complex=False)

    # run the iterative algorithm
    for k in range(num_iters):
        print(k)
        optimizer.zero_grad()
        # forward propagation from the SLM plane to the target plane
        real, imag = utils.polar_to_rect(torch.ones_like(slm_phase), slm_phase)
        slm_field = torch.complex(real, imag)

        recon_field = utils.propagate_field(slm_field, propagator, prop_dist, wavelength, feature_size,
                                            prop_model, dtype, precomputed_H)

        # get amplitude
        recon_amp = recon_field.abs()

        # crop roi
        recon_amp = utils.crop_image(recon_amp, target_shape=roi_res, stacked_complex=False)


        out_amp = recon_amp

        # calculate loss and backprop
        lossValue = loss(s * out_amp, target_amp)
        lossValue.backward()
        optimizer.step()

        # write to tensorboard / write phase image
        # Note that it takes 0.~ s for writing it to tensorboard
        # with torch.no_grad():
        #     if k % 50 == 0:
        #         utils.write_sgd_summary(slm_phase, out_amp, target_amp, k,
        #                                 writer=writer, path=phase_path, s=s, prefix='test')

    return slm_phase


class GS(nn.Module):
    """Classical Gerchberg-Saxton algorithm

    Class initialization parameters
    -------------------------------
    :param prop_dist: propagation dist between SLM and target, in meters
    :param wavelength: the wavelength of interest, in meters
    :param feature_size: the SLM pixel pitch, in meters
    :param num_iters: the number of iteration, default 500
    :param phase_path: path to write intermediate results
    :param loss: loss function, default L2
    :param prop_model: chooses the propagation operator ('ASM': propagation_ASM,
        'model': calibrated model). Default 'ASM'.
    :param propagator: propagator instance (function / pytorch module)
    :param device: torch.device

    Usage
    -----
    Functions as a pytorch module:

    >>> gs = GS(...)
    >>> final_phase = gs(target_amp, init_phase)

    target_amp: amplitude at the target plane, with dimensions [batch, 1, height, width]
    init_phase: initial guess of phase of phase-only slm
    final_phase: optimized phase-only representation at SLM plane, same dimensions
    """
    def __init__(self, prop_dist, wavelength, feature_size, num_iters, phase_path=None,
                 prop_model='ASM', propagator=None, writer=None, device=torch.device('cuda')):
        super(GS, self).__init__()

        # Setting parameters
        self.prop_dist = prop_dist
        self.wavelength = wavelength
        self.feature_size = feature_size
        self.phase_path = phase_path
        self.precomputed_H_f = None
        self.precomputed_H_b = None
        self.prop_model = prop_model
        self.prop = propagator
        self.num_iters = num_iters
        self.writer = writer
        self.dev = device

    def forward(self, target_amp, init_phase=None):
        # Pre-compute propagataion kernel only once
        if self.precomputed_H_f is None and self.prop_model == 'ASM':
            self.precomputed_H_f = self.prop(torch.empty(*init_phase.shape, dtype=torch.complex64), self.feature_size,
                                             self.wavelength, self.prop_dist, return_H=True)
            self.precomputed_H_f = self.precomputed_H_f.to(self.dev).detach()
            self.precomputed_H_f.requires_grad = False

        if self.precomputed_H_b is None and self.prop_model == 'ASM':
            self.precomputed_H_b = self.prop(torch.empty(*init_phase.shape, dtype=torch.complex64), self.feature_size,
                                             self.wavelength, -self.prop_dist, return_H=True)
            self.precomputed_H_b = self.precomputed_H_b.to(self.dev).detach()
            self.precomputed_H_b.requires_grad = False

        # Run algorithm
        final_phase = gerchberg_saxton(init_phase, target_amp, self.num_iters, self.prop_dist,
                                       self.wavelength, self.feature_size,
                                       phase_path=self.phase_path,
                                       prop_model=self.prop_model, propagator=self.prop,
                                       precomputed_H_f=self.precomputed_H_f, precomputed_H_b=self.precomputed_H_b,
                                       writer=self.writer)
        return final_phase

    @property
    def phase_path(self):
        return self._phase_path

    @phase_path.setter
    def phase_path(self, phase_path):
        self._phase_path = phase_path


class SGD(nn.Module):
    """Proposed Stochastic Gradient Descent Algorithm using Auto-diff Function of PyTorch

    Class initialization parameters
    -------------------------------
    :param prop_dist: propagation dist between SLM and target, in meters
    :param wavelength: the wavelength of interest, in meters
    :param feature_size: the SLM pixel pitch, in meters
    :param num_iters: the number of iteration, default 500
    :param roi_res: region of interest to penalize the loss
    :param phase_path: path to write intermediate results
    :param loss: loss function, default L2
    :param prop_model: chooses the propagation operator ('ASM': propagation_ASM,
        'model': calibrated model). Default 'ASM'.
    :param propagator: propagator instance (function / pytorch module)
    :param lr: learning rate for phase variables
    :param lr_s: learning rate for the learnable scale
    :param s0: initial scale
    :param writer: SummaryWrite instance for tensorboard
    :param device: torch.device

    Usage
    -----
    Functions as a pytorch module:

    >>> sgd = SGD(...)
    >>> final_phase = sgd(target_amp, init_phase)

    target_amp: amplitude at the target plane, with dimensions [batch, 1, height, width]
    init_phase: initial guess of phase of phase-only slm
    final_phase: optimized phase-only representation at SLM plane, same dimensions
    """
    def __init__(self, prop_dist, wavelength, feature_size, num_iters, roi_res, phase_path=None, prop_model='ASM',
                 propagator=None, loss=nn.MSELoss(), lr=0.01, lr_s=0.003, s0=1.0,
                 writer=None, device=torch.device('cuda')):
        super(SGD, self).__init__()

        # Setting parameters
        self.prop_dist = prop_dist
        self.wavelength = wavelength
        self.feature_size = feature_size
        self.roi_res = roi_res
        self.phase_path = phase_path
        self.precomputed_H = None
        self.prop_model = prop_model
        self.prop = propagator

        self.num_iters = num_iters
        self.lr = lr
        self.lr_s = lr_s
        self.init_scale = s0

        self.writer = writer
        self.dev = device
        self.loss = loss.to(device)

    def forward(self, target_amp, init_phase=None):
        # Pre-compute propagataion kernel only once
        if self.precomputed_H is None and self.prop_model == 'ASM':
            self.precomputed_H = self.prop(torch.empty(*init_phase.shape, dtype=torch.complex64), self.feature_size,
                                           self.wavelength, self.prop_dist, return_H=True)
            self.precomputed_H = self.precomputed_H.to(self.dev).detach()
            self.precomputed_H.requires_grad = False

        # Run algorithm
        final_phase = stochastic_gradient_descent(init_phase, target_amp, self.num_iters, self.prop_dist,
                                                  self.wavelength, self.feature_size,
                                                  roi_res=self.roi_res, phase_path=self.phase_path,
                                                  prop_model=self.prop_model, propagator=self.prop,
                                                  loss=self.loss, lr=self.lr, lr_s=self.lr_s, s0=self.init_scale,
                                                  writer=self.writer,
                                                  precomputed_H=self.precomputed_H)
        return final_phase

    @property
    def init_scale(self):
        return self._init_scale

    @init_scale.setter
    def init_scale(self, s):
        self._init_scale = s

    @property
    def citl_hardwares(self):
        return self._citl_hardwares

    @citl_hardwares.setter
    def citl_hardwares(self, citl_hardwares):
        self._citl_hardwares = citl_hardwares

    @property
    def phase_path(self):
        return self._phase_path

    @phase_path.setter
    def phase_path(self, phase_path):
        self._phase_path = phase_path

    @property
    def prop(self):
        return self._prop

    @prop.setter
    def prop(self, prop):
        self._prop = prop

class ImageLoader:
    """Loads images a folder with augmentation for generator training

    Class initialization parameters
    -------------------------------
    data_path: folder containing images
    channel: color channel to load (0, 1, 2 for R, G, B, None for all 3),
        default None
    batch_size: number of images to pass each iteration, default 1
    image_res: 2d dimensions to pad/crop the image to for final output, default
        (1080, 1920)
    homography_res: 2d dims to scale the image to before final crop to image_res
        for consistent resolutions (crops to preserve input aspect ratio),
        default (880, 1600)
    shuffle: True to randomize image order across batches, default True
    vertical_flips: True to augment with vertical flipping, default True
    horizontal_flips: True to augment with horizontal flipping, default True
    idx_subset: for the iterator, skip all but these images. Given as a list of
        indices corresponding to sorted filename order. Forces shuffle=False and
        batch_size=1. Defaults to None to not subset at all.
    crop_to_homography: if True, only crops the image instead of scaling to get
        to target homography resolution, default False

    Usage
    -----
    To be used as an iterator:

    >>> image_loader = ImageLoader(...)
    >>> for ims, input_resolutions, filenames in image_loader:
    >>>     ...

    ims: images in the batch after transformation and conversion to linear
        amplitude, with dimensions [batch, channel, height, width]
    input_resolutions: list of length batch_size containing tuples of the
        original image height/width before scaling/cropping
    filenames: list of input image filenames, without extension

    Alternatively, can be used to manually load a single image:

    >>> ims, input_resolutions, filenames = image_loader.load_image(idx)

    idx: the index for the image to load, indices are alphabetical based on the
        file path.
    """

    def __init__(self, data_path, channel=None, batch_size=1,
                 image_res=(1080, 1920), homography_res=(880, 1600),
                 shuffle=True, vertical_flips=True, horizontal_flips=True,
                 idx_subset=None, crop_to_homography=False):
        if not os.path.isdir(data_path):
            raise NotADirectoryError(f'Data folder: {data_path}')
        self.data_path = data_path
        self.channel = channel
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_res = image_res
        self.homography_res = homography_res
        self.subset = idx_subset
        self.crop_to_homography = crop_to_homography

        self.augmentations = []
        if vertical_flips:
            self.augmentations.append(self.augment_vert)
        if horizontal_flips:
            self.augmentations.append(self.augment_horz)
        # store the possible states for enumerating augmentations
        self.augmentation_states = [fn() for fn in self.augmentations]

        self.im_names = get_image_filenames(data_path)
        self.im_names.sort()

        # if subsetting indices, force no randomization and batch size 1
        if self.subset is not None:
            self.shuffle = False
            self.batch_size = 1

        # create list of image IDs with augmentation state
        self.order = ((i,) for i in range(len(self.im_names)))
        for aug_type in self.augmentations:
            states = aug_type()  # empty call gets possible states
            # augment existing list with new entry to states tuple
            self.order = ((*prev_states, s)
                          for prev_states in self.order
                          for s in states)
        self.order = list(self.order)

    def __iter__(self):
        self.ind = 0
        if self.shuffle:
            random.shuffle(self.order)
        return self

    def __next__(self):
        if self.subset is not None:
            while self.ind not in self.subset and self.ind < len(self.order):
                self.ind += 1

        if self.ind < len(self.order):
            batch_ims = self.order[self.ind:self.ind+self.batch_size]
            self.ind += self.batch_size
            return self.load_batch(batch_ims)
        else:
            raise StopIteration

    def __len__(self):
        if self.subset is None:
            return len(self.order)
        else:
            return len(self.subset)

    def load_batch(self, images):
        im_res_name = [self.load_image(*im_data) for im_data in images]
        ims = torch.stack([im for im, _, _ in im_res_name], 0)
        return (ims,
                [res for _, res, _ in im_res_name],
                [name for _, _, name in im_res_name])

    def load_image(self, filenum, *augmentation_states):
        im = imread(self.im_names[filenum])

        if len(im.shape) < 3:
            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)  # augment channels for gray images

        if self.channel is None:
            im = im[..., :3]  # remove alpha channel, if any
        else:
            # select channel while keeping dims
            im = im[..., self.channel, np.newaxis]

        im = utils.im2float(im, dtype=np.float64)  # convert to double, max 1

        # linearize intensity and convert to amplitude
        low_val = im <= 0.04045
        im[low_val] = 25 / 323 * im[low_val]
        im[np.logical_not(low_val)] = ((200 * im[np.logical_not(low_val)] + 11)
                                       / 211) ** (12 / 5)
        im = np.sqrt(im)  # to amplitude

        # move channel dim to torch convention
        im = np.transpose(im, axes=(2, 0, 1))

        # apply data augmentation
        for fn, state in zip(self.augmentations, augmentation_states):
            im = fn(im, state)

        # normalize resolution
        input_res = im.shape[-2:]
        im = resize_keep_aspect(im, self.image_res)
        # if self.crop_to_homography:
        #     im = pad_crop_to_res(im, self.homography_res)
        # else:
        #     im = resize_keep_aspect(im, self.homography_res)
        im = pad_crop_to_res(im, self.image_res)

        return (torch.from_numpy(im).float(),
                input_res,
                os.path.splitext(self.im_names[filenum])[0])

    def augment_vert(self, image=None, flip=False):
        if image is None:
            return (True, False)  # return possible augmentation values

        if flip:
            return image[..., ::-1, :]
        return image

    def augment_horz(self, image=None, flip=False):
        if image is None:
            return (True, False)  # return possible augmentation values

        if flip:
            return image[..., ::-1]
        return image


def get_image_filenames(dir):
    """Returns all files in the input directory dir that are images"""
    image_types = ('jpg', 'jpeg', 'tiff', 'tif', 'png', 'bmp', 'gif')
    files = os.listdir(dir)
    exts = (os.path.splitext(f)[1] for f in files)
    images = [os.path.join(dir, f)
              for e, f in zip(exts, files)
              if e[1:] in image_types]
    return images


def resize_keep_aspect(image, target_res, pad=False):
    """Resizes image to the target_res while keeping aspect ratio by cropping

    image: an 3d array with dims [channel, height, width]
    target_res: [height, width]
    pad: if True, will pad zeros instead of cropping to preserve aspect ratio
    """
    im_res = image.shape[-2:]

    # finds the resolution needed for either dimension to have the target aspect
    # ratio, when the other is kept constant. If the image doesn't have the
    # target ratio, then one of these two will be larger, and the other smaller,
    # than the current image dimensions
    resized_res = (int(np.ceil(im_res[1] * target_res[0] / target_res[1])),
                   int(np.ceil(im_res[0] * target_res[1] / target_res[0])))

    # only pads smaller or crops larger dims, meaning that the resulting image
    # size will be the target aspect ratio after a single pad/crop to the
    # resized_res dimensions
    if pad:
        image = utils.pad_image(image, resized_res, pytorch=False)
    else:
        image = utils.crop_image(image, resized_res, pytorch=False)

    # switch to numpy channel dim convention, resize, switch back
    image = np.transpose(image, axes=(1, 2, 0))
    image = resize(image, target_res, mode='reflect')
    return np.transpose(image, axes=(2, 0, 1))


def pad_crop_to_res(image, target_res):
    """Pads with 0 and crops as needed to force image to be target_res

    image: an array with dims [..., channel, height, width]
    target_res: [height, width]
    """
    return utils.crop_image(utils.pad_image(image,
                                            target_res, pytorch=False),
                            target_res, pytorch=False)




# Command line argument processing
p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--channel', type=int, default=1, help='Red:0, green:1, blue:2')
p.add_argument('--method', type=str, default='SGD', help='Type of algorithm, GS/SGD')
p.add_argument('--prop_model', type=str, default='ASM', help='Type of propagation model, ASM or model')
p.add_argument('--root_path', type=str, default='./phases', help='Directory where optimized phases will be saved.')
p.add_argument('--data_path', type=str, default='./data', help='Directory for the dataset')
# p.add_argument('--generator_dir', type=str, default='./pretrained_networks',
#                help='Directory for the pretrained holonet/unet network')
# p.add_argument('--prop_model_dir', type=str, default='./calibrated_models',
#                help='Directory for the CITL-calibrated wave propagation models')
p.add_argument('--experiment', type=str, default='', help='Name of experiment')
p.add_argument('--lr', type=float, default=8e-3, help='Learning rate for phase variables (for SGD)')
p.add_argument('--lr_s', type=float, default=2e-3, help='Learning rate for learnable scale (for SGD)')
p.add_argument('--num_iters', type=int, default=500, help='Number of iterations (GS, SGD)')

# parse arguments
opt = p.parse_args()
run_id = f'{opt.experiment}_{opt.method}_{opt.prop_model}'  # {algorithm}_{prop_model} format


channel = opt.channel  # Red:0 / Green:1 / Blue:2
chan_str = ('red', 'green', 'blue')[channel]

print(f'   - optimizing phase with {opt.method}/{opt.prop_model} ... ')

# Hyperparameters setting
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
prop_dist = (20 * cm, 20 * cm, 20 * cm)[channel]  # propagation distance from SLM plane to target plane
wavelength = (638 * nm, 520 * nm, 450 * nm)[channel]  # wavelength of each color
feature_size = (6.4 * um, 6.4 * um)  # SLM pitch
slm_res = (1080, 1920)  # resolution of SLM
image_res = (1080, 1920)
roi_res = (880, 1600)  # regions of interest (to penalize for SGD)
dtype = torch.float32  # default datatype (Note: the result may be slightly different if you use float64, etc.)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # The gpu you are using

# Options for the algorithm
loss = nn.MSELoss().to(device)  # loss functions to use (try other loss functions!)
s0 = 1.0  # initial scale

root_path = os.path.join(opt.root_path, run_id, chan_str)  # path for saving out optimized phases

# Tensorboard writer
summaries_dir = os.path.join(root_path, 'summaries')
utils.cond_mkdir(summaries_dir)
writer = None;#SummaryWriter(summaries_dir)

camera_prop = None

# Simulation model
if opt.prop_model == 'ASM':
    propagator = propagation_ASM  # Ideal model

# elif opt.prop_model.upper() == 'MODEL':
#     blur = utils.make_kernel_gaussian(0.85, 3)
#     propagator = ModelPropagate(distance=prop_dist,  # Parameterized wave propagation model
#                                 feature_size=feature_size,
#                                 wavelength=wavelength,
#                                 blur=blur).to(device)

   


# Select Phase generation method, algorithm
if opt.method == 'SGD':
    phase_only_algorithm = SGD(prop_dist, wavelength, feature_size, opt.num_iters, roi_res, root_path,
                               opt.prop_model, propagator, loss, opt.lr, opt.lr_s, s0, writer, device)
elif opt.method == 'GS':
    phase_only_algorithm = GS(prop_dist, wavelength, feature_size, opt.num_iters, root_path,
                              opt.prop_model, propagator, writer, device)


# Augmented image loader (if you want to shuffle, augment dataset, put options accordingly.)
image_loader = ImageLoader(opt.data_path, channel=channel,
                           image_res=image_res, homography_res=roi_res,
                           crop_to_homography=True,
                           shuffle=False, vertical_flips=False, horizontal_flips=False)

# Loop over the dataset
for k, target in enumerate(image_loader):
    # get target image
    target_amp, target_res, target_filename = target
    target_path, target_filename = os.path.split(target_filename[0])
    target_idx = target_filename.split('_')[-1]
    target_amp = target_amp.to(device)
    print(target_idx)

    # if you want to separate folders by target_idx or whatever, you can do so here.
    phase_only_algorithm.init_scale = s0 * utils.crop_image(target_amp, roi_res, stacked_complex=False).mean()
    phase_only_algorithm.phase_path = os.path.join(root_path)

       
    # iterative methods, initial phase: random guess
    init_phase = (-0.5 + 1.0 * torch.rand(1, 1, *slm_res)).to(device)
    final_phase = phase_only_algorithm(target_amp, init_phase)

    print(final_phase.shape)

    # save the final result somewhere.
    phase_out_8bit = utils.phasemap_8bit(final_phase.cpu().detach(), inverted=True)

    utils.cond_mkdir(root_path)
    cv2.imwrite(os.path.join(root_path, f'{target_idx}.png'), phase_out_8bit)

print(f'    - Done, result: --root_path={root_path}')

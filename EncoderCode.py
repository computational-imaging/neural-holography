# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 23:09:29 2023

@author: varsh
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

import matplotlib.pyplot as plt 

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import configargparse
import random
import numpy as np
from imageio import imread
from skimage.transform import resize
#from torch.utils.tensorboard import SummaryWriter

import utils.utils as utils

class ImageLoaderModified:
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
                 image_res=(1080, 1920),
                 shuffle=True, vertical_flips=True, horizontal_flips=True,
                 idx_subset=None):
        if not os.path.isdir(data_path):
            raise NotADirectoryError(f'Data folder: {data_path}')
        self.data_path = data_path
        self.channel = channel
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_res = image_res
        self.subset = idx_subset

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
        im = resize(im, image_res)

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
        # im = resize_keep_aspect(im, self.image_res)
        # im = pad_crop_to_res(im, self.image_res)

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
    
# Creating a DeepAutoencoder class 
class DeepAutoencoder(torch.nn.Module): 
    def __init__(self, dtype,propagator, prop_dist, wavelength, feature_size, prop_model, precomputed_H,inputSize = [1920,1080]):
        self.propagator = propagator;
        self.prop_dist = prop_dist;
        self.wavelength = wavelength;
        self.feature_size = feature_size;
        self.prop_model = prop_model;
        self.precomputed_H = precomputed_H;
        self.inputSize = inputSize;
        self.dtype = dtype;
        inpSize = inputSize[0] * inputSize[1];
        redRatio1 = 1;
        redRatio2 = 1;
        super().__init__()         
        self.encoder = torch.nn.Sequential( 
            torch.nn.Linear(inpSize, inpSize*redRatio1), 
            torch.nn.ReLU(), 
            torch.nn.Linear(inpSize*redRatio1, inpSize*redRatio2),
            torch.nn.Tanh()
        ) 
          
        # self.decoder = torch.nn.Sequential( 
        
        # ) 
  
    def forward(self, x): 
        newX = x.reshape(-1, image_res[0]*image_res[1]) 
        encoded = 0.5*self.encoder(newX)#Output between -0.5 and 0.5
        encoded_reshaped = torch.reshape(encoded, (1, 1, *self.inputSize))
        #decoded = self.decoder(encoded) 
        real, imag = utils.polar_to_rect_just_ang(encoded_reshaped);
        slm_field = torch.complex(real, imag)

        recon_field = utils.propagate_field(slm_field, self.propagator, self.prop_dist, self.wavelength, self.feature_size,
                                            self.prop_model, self.dtype, self.precomputed_H)

        # get amplitude
        recon_amp = recon_field.abs()

        # crop roi
        recon_amp = utils.crop_image(recon_amp, target_shape=self.inputSize, stacked_complex=False)


        out_amp = recon_amp

        # calculate loss and backprop
        #lossValue = loss(s * out_amp, target_amp)
        return out_amp     


data_path = './data'

cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
channels = [0,1,2]
prop_dists = (20 * cm, 20 * cm, 20 * cm)
wavelenghts = (638 * nm, 520 * nm, 450 * nm)
feature_size = (6.4 * um, 6.4 * um)  # SLM pitch
#slm_res = (1080, 1920)  # resolution of SLM
image_res = (54, 96)
#roi_res = (880, 1600)  # regions of interest (to penalize for SGD)
dtype = torch.float32  # default datatype (Note: the result may be slightly different if you use float64, etc.)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # The gpu you are using
propagator = propagation_ASM
prop_model = 'ASM'


num_epochs = 100



for channel in channels:
    prop_dist = prop_dists[channel]  # propagation distance from SLM plane to target plane
    wavelength = wavelenghts[channel]  # wavelength of each color
    
    precomputed_H = propagator(torch.empty((1,1,*image_res), dtype=torch.complex64), feature_size,
                                   wavelength, prop_dist, return_H=True)
    precomputed_H = precomputed_H.to(device).detach()
    precomputed_H.requires_grad = False
  
    # Augmented image loader (if you want to shuffle, augment dataset, put options accordingly.)
    image_loader = ImageLoaderModified(data_path, channel=channel,
                               image_res=image_res,
                               shuffle=False, vertical_flips=False, horizontal_flips=False)
    # Instantiating the model and hyperparameters 
    
    model = DeepAutoencoder(dtype,propagator, prop_dist, wavelength, feature_size, prop_model, precomputed_H,inputSize = image_res) 
    criterion = torch.nn.MSELoss() 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    # List that will store the training loss 
    train_loss = [] 
    
    # Dictionary that will store the 
    # different images and outputs for 
    # various epochs 
    outputs = {} 
    
    batch_size = len(image_loader) 
    
    # Training loop starts 
    for epoch in range(num_epochs): 
    		
    	# Initializing variable for storing 
    	# loss 
    	running_loss = 0
    	
    	# Iterating over the training dataset 
    	for k, target in enumerate(image_loader):
    			
            target_amp, target_res, target_filename = target
            target_path, target_filename = os.path.split(target_filename[0])
            target_idx = target_filename.split('_')[-1]
            #target_amp = target_amp.to(device)
            print(target_idx)
    		# Loading image(s) and 
    		# reshaping it into a 1-d vector
            img = target_amp;
            img1 = torch.fft.fftn(utils.ifftshift(img), dim=(-2, -1), norm='ortho').abs();
    		# Generating output 
            out = model(img1)
    		
    		# Calculating loss 
            loss = criterion(out, img) 
    		
    		# Updating weights according 
    		# to the calculated loss 
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
    		
    		# Incrementing loss 
            running_loss += loss.item() 
    	
    	# Averaging out loss over entire batch 
    	running_loss /= batch_size 
    	train_loss.append(running_loss) 
    	
    	# Storing useful images and 
    	# reconstructed outputs for the last batch 
    	outputs[epoch+1] = {'img': img, 'out': out} 
    
    
    # Plotting the training loss 
    plt.plot(range(1,num_epochs+1),train_loss) 
    plt.xlabel("Number of epochs") 
    plt.ylabel("Training Loss") 
    plt.show()

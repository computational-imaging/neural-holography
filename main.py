"""
Neural holography:

This is the main executive script used for the phase optimization using (GS/DPAC/SGD) + camera-in-the-loop (CITL) optimization.

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

$ python main.py --channel=0 --algorithm=HOLONET --root_path=./phases --model_dir=./pretrained_models
"""

import os
import sys
import cv2
import torch
import torch.nn as nn
import configargparse
from torch.utils.tensorboard import SummaryWriter

import utils.utils as utils
from utils.augmented_image_loader import ImageLoader
from utils.modules import SGD, GS, DPAC
from holonet import HoloNet, InitialPhaseUnet, FinalPhaseOnlyUnet, PhaseOnlyUnet
from propagation_ASM import propagation_ASM

# Command line argument processing
p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--channel', type=int, default=1, help='red:0, green:1, blue:2, rgb:3')
p.add_argument('--method', type=str, default='SGD', help='Type of algorithm')
p.add_argument('--prop_model', type=str, default='ASM', help='Type of propagation model')
p.add_argument('--root_path', type=str, default='./media/data4a/suyeon/results', help='Name of directory where optimized phases will be saved.')
p.add_argument('--model_dir', type=str, default='./pretrained_networks/',
               help='Directory for the pretrained holonet/unet network')

# parse arguments
opt = p.parse_args()
run_id = f'{opt.method}_{opt.prop_model}'  # {algorithm}_{prop_model} format

channel = opt.channel  # Red:0 / Green:1 / Blue:2
chan_str = ('red', 'green', 'blue')[channel]

# Hyperparameters setting
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
prop_dist = (20 * cm, 20 * cm, 20 * cm)[channel]  # propagation distance from SLM plane to target plane
wavelength = (638 * nm, 520 * nm, 450 * nm)[channel]  # wavelength of each color
feature_size = (6.4 * um, 6.4 * um)  # SLM pitch
slm_res = (1080, 1920)  # resolution of SLM
image_res = (1080, 1920)
roi_res = (880, 1600)  # regions of interest (to penalize for SGD)
dtype = torch.float32  # default datatype (Note: the result may be slightly different if you use float64, etc.)
device = torch.device('cuda')  # The gpu you are using

if opt.prop_model == 'ASM':
    propagator = propagation_ASM

print(f'   - optimizing phase with {opt.method}/{opt.prop_model} ... ')

# Options for the algorithm
num_iters = 500  # the number of iterations
lr = 0.02  # learning rate for phase variables
lr_s = 0.005  # learning rate for the learnable scale
loss = nn.MSELoss().to(device)  # loss functions to use (try other loss functions!)
s0 = 0.75  # initial scale

# Data path
data_path = './data'
root_path = os.path.join(opt.root_path, run_id, chan_str)  # path for saving out optimized phases

# Tensorboard writer
summaries_dir = os.path.join(root_path, 'summaries')
utils.cond_mkdir(summaries_dir)
writer = SummaryWriter(summaries_dir)

# algorithm
if opt.method == 'SGD':
    phase_only_algorithm = SGD(prop_dist, wavelength, feature_size, num_iters, roi_res, root_path,
                               opt.prop_model, propagator, loss, lr, lr_s, s0, writer, device)
elif opt.method == 'GS':
    phase_only_algorithm = GS(prop_dist, wavelength, feature_size, num_iters, root_path,
                              opt.prop_model, propagator, writer, device)
elif opt.method == 'DPAC':
    phase_only_algorithm = DPAC(prop_dist, wavelength, feature_size, opt.prop_model, propagator, device)
elif opt.method == 'HOLONET':
    phase_only_algorithm = HoloNet(prop_dist, wavelength, feature_size, initial_phase=InitialPhaseUnet(4, 16),
                                   final_phase_only=FinalPhaseOnlyUnet(4, 16, num_in=2)).to(device)
    model_path = os.path.join(opt.model_dir, f'holonet20_{chan_str}.pth')
    image_res = (1072, 1920)
elif opt.method == 'UNET':
    phase_only_algorithm = PhaseOnlyUnet(num_features_init=32).to(device)
    model_path = os.path.join(opt.model_dir, f'unet20_{chan_str}.pth')
    image_res = (1024, 2048)

if 'NET' in opt.method:
    checkpoint = torch.load(model_path)
    phase_only_algorithm.load_state_dict(checkpoint)
    phase_only_algorithm.eval()


# Augmented image loader (if you want to shuffle, augment dataset, put options accordingly.)
image_loader = ImageLoader(data_path, channel=channel,
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

    # if you want to separate folders by target_idx or whatever, you can do so here.
    phase_only_algorithm.phase_path = os.path.join(root_path)

    # run algorithm (See algorithm_modules.py and algorithms.py)
    if opt.method in ['DPAC', 'HOLONET', 'UNET']:
        # direct methods
        _, final_phase = phase_only_algorithm(target_amp)
    else:
        # iterative methods, initial phase: random guess
        init_phase = (-0.5 + 1.0 * torch.rand(1, 1, *slm_res)).to(device)
        final_phase = phase_only_algorithm(target_amp, init_phase)

    # save the final result somewhere.
    phase_out_8bit = utils.phasemap_8bit(final_phase.cpu().detach(), inverted=True)

    utils.cond_mkdir(root_path)
    cv2.imwrite(os.path.join(root_path, f'{target_idx}.png'), phase_out_8bit)

print(f'    - Done, result: --root_path={root_path}')

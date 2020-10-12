"""
This is the script that is used for evaluating phases for physical or simulation forward model

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Technical Paper:
Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein. Neural Holography with Camera-in-the-loop Training. ACM TOG (SIGGRAPH Asia), 2020.

-----

$ python eval.py --channel=[0 or 1 or 2 or 3] --root_path=[some path]

"""

import torch
import imageio
import os
import skimage.io
import scipy.io as sio
import sys
import numpy as np
import configargparse

from propagation_ASM import propagation_ASM
from utils.augmented_image_loader import ImageLoader
import utils.utils as utils

# Command line argument processing
p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--channel', type=int, default=1, help='red:0, green:1, blue:2, rgb:3')
p.add_argument('--prop_model', type=str, default='ASM', help='Type of propagation model.')
p.add_argument('--root_path', type=str, default='./phases', help='Name of directory where test phases are being stored.')

# Parse
opt = p.parse_args()
channel = opt.channel
chs = range(channel) if channel == 3 else [channel]  # retrieve all channels if channel is 3
run_id = opt.root_path.split('/')[-1]  # {algorithm}_{prop_model}

# Hyperparameters setting
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
chan_strs = ('red', 'green', 'blue', 'rgb')
prop_dist = 20 * cm
wavelengths = (638 * nm, 520 * nm, 450 * nm)  # wavelength of each color
feature_size = (6.4 * um, 6.4 * um)  # SLM pitch

# Resolutions
slm_res = (1080, 1920)  # resolution of SLM
if 'HOLONET' in run_id.upper():
    slm_res = (1072, 1920)
elif 'UNET' in run_id.upper():
    slm_res = (1024, 2048)

image_res = (1080, 1920)
roi_res = (880, 1600)  # regions of interest (to penalize)
dtype = torch.float32  # default datatype (Note: the result may be slightly different if you use float64, etc.)
device = torch.device('cuda')  # The gpu you are using

# You can pre-compute kernels for fast-computation
precomputed_H = [None] * 3
if opt.prop_model == 'ASM':
    propagator = propagation_ASM

    for c in chs:
        precomputed_H[c] = propagator(torch.empty(1, 1, *slm_res, 2), feature_size,
                                      wavelengths[c], prop_dist, return_H=True).to(device)

#### The code for this part will be released during SIGGRAPH Asia. ####

# elif opt.prop_model is 'CAMERA':
#     camera, slm, alc, calibrator = utils_citl.setup_hardwares(channel,
#                                                               slm_settle_time=0.1,
#                                                               path_calib_patterns=calib_path,
#                                                               plot_calib_preview=True)
#     propagator = utils_citl.cameraProp(slm, camera, calibrator)
# elif opt.prop_model is 'MODEL':
#     propagator = propagation_model

print(f'  - reconstruction with {opt.prop_model}... ')

# Data path
data_path = './data'
recon_path = './recon'

# Augmented image loader (if you want to shuffle, augment dataset, put options accordingly.)
image_loader = ImageLoader(data_path, channel=channel if channel < 3 else None,
                           image_res=image_res, homography_res=roi_res,
                           crop_to_homography=True,
                           shuffle=False, vertical_flips=False, horizontal_flips=False)

# Placeholders for metrics
psnrs = {'amp':[], 'lin':[], 'srgb':[]}
ssims = {'amp':[], 'lin':[], 'srgb':[]}
idxs = []

# Loop over the dataset
for k, target in enumerate(image_loader):
    # get target image
    target_amp, target_res, target_filename = target
    target_path, target_filename = os.path.split(target_filename[0])
    target_idx = target_filename.split('_')[-1]
    target_amp = target_amp.to(device)

    print(f'    - running for img_{target_idx}...')

    # crop to ROI
    target_amp = utils.crop_image(target_amp, target_shape=roi_res, stacked_complex=False).to(device)

    recon_amp = []

    # for each channel, propagate wave from the SLM plane to the image plane and get the reconstructed image.
    for c in chs:
        # load and invert phase (our SLM setup)
        phase_filename = os.path.join(f'{opt.root_path}/{chan_strs[c]}', f'{target_idx}.png')
        slm_phase = skimage.io.imread(phase_filename) / 255.
        slm_phase = torch.tensor((1 - slm_phase) * 2 * np.pi - np.pi, dtype=dtype).reshape(1, 1, *slm_res).to(device)

        # propagate field
        real, imag = utils.polar_to_rect(torch.ones_like(slm_phase), slm_phase)
        slm_field = torch.stack((real, imag), -1)
        recon_field = utils.propagate_field(slm_field, propagator, prop_dist, wavelengths[c], feature_size,
                                            opt.prop_model, dtype)

        # cartesian to polar coordinate
        recon_amp_c, _ = utils.rect_to_polar(recon_field[..., 0], recon_field[..., 1])

        # crop to ROI
        recon_amp_c = utils.crop_image(recon_amp_c, target_shape=roi_res, stacked_complex=False)

        # append to list
        recon_amp.append(recon_amp_c)

    # list to tensor, scaling
    recon_amp = torch.cat(recon_amp, dim=1)
    recon_amp *= (torch.sum(recon_amp * target_amp, (-2, -1), keepdim=True)
                  / torch.sum(recon_amp * recon_amp, (-2, -1), keepdim=True))

    # tensor to numpy
    recon_amp = recon_amp.squeeze().cpu().detach().numpy()
    target_amp = target_amp.squeeze().cpu().detach().numpy()
    if channel == 3:
        recon_amp = recon_amp.transpose(1, 2, 0)
        target_amp = target_amp.transpose(1, 2, 0)

    # calculate metrics
    psnr_val, ssim_val = utils.get_psnr_ssim(recon_amp, target_amp, multichannel=(channel == 3))

    idxs.append(target_idx)

    for domain in ['amp', 'lin', 'srgb']:
        psnrs[domain].append(psnr_val[domain])
        ssims[domain].append(ssim_val[domain])
        print(f'PSNR({domain}): {psnr_val[domain]},  SSIM({domain}): {ssim_val[domain]:.4f}, ')

    # save reconstructed image in srgb domain
    recon_srgb = utils.srgb_lin2gamma(np.clip(recon_amp**2, 0.0, 1.0))
    utils.cond_mkdir(recon_path)
    imageio.imwrite(os.path.join(recon_path, f'{target_idx}_{run_id}_{chan_strs[channel]}.png'), (recon_srgb * 255).round().astype(np.uint8))

# save it as a .mat file
data_dict = {}
data_dict['img_idx'] = idxs
for domain in ['amp', 'lin', 'srgb']:
    data_dict[f'ssims_{domain}'] = ssims[domain]
    data_dict[f'psnrs_{domain}'] = psnrs[domain]

sio.savemat(os.path.join(recon_path, f'metrics_{run_id}_{chan_strs[channel]}.mat'), data_dict)

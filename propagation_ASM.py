"""
This is the script that is used for the wave propagation using the angular spectrum method (ASM). Refer to 
Goodman, Joseph W. Introduction to Fourier optics. Roberts and Company Publishers, 2005, for principle details.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Technical Paper:
Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein. Neural Holography with Camera-in-the-loop Training. ACM TOG (SIGGRAPH Asia), 2020.
"""

import math
import torch
import numpy as np
import utils.utils as utils
import torch.fft
from aotools.functions import zernikeArray


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


def propagation_ASM_zernike(u_in, feature_size, wavelength, z, linear_conv=True,
                            padtype='zero', coeffs=None, zernike=None, adjoint=False,
                            return_H=False, precomped_H=None, return_H_exp=False,
                            precomped_H_exp=None, dtype=torch.float32):
    """A wrapper around propagation_ASM that applies a Zernike phase correction

    Inputs
    ------
    coeffs: a 1d tensor of Zernike coefficients
    zernike: a precomputed Zernike function basis. Should contain the same
        number of basis functions as number of coeffs, and be the same height
        and width as u_in. Use compute_zernike_basis to compute
    adjoint: if True, propagate then apply zernike. If False, apply then prop

    See propagation_ASM for u_in, feature_size, wavelength, z, linear_conv,
    padtype, return_H, precomped_H, return_H_exp, precomped_H_exp, dtype
    """

    if return_H or return_H_exp or coeffs is None:
        return propagation_ASM(u_in, feature_size, wavelength, z, linear_conv,
                               padtype, return_H, precomped_H, return_H_exp,
                               precomped_H_exp, dtype)

    coeffs = coeffs.reshape(-1, 1, 1)

    if zernike is None:
        # resolution of input field, should be: (num_images, num_channels, height, width, 2)
        zernike = compute_zernike_basis(coeffs.size()[0], u_in.size()[-3:-1])

    if z < 0:
        coeffs = -coeffs

    # create phase
    zernike = combine_zernike_basis(coeffs, zernike)

    # operator order
    if adjoint:
        u_out = propagation_ASM(u_in, feature_size, wavelength, z, linear_conv,
                                padtype, return_H, precomped_H, return_H_exp,
                                precomped_H_exp)
        u_out = utils.mul_complex(zernike, u_out)
    else:
        u_out = utils.mul_complex(zernike, u_in)
        u_out = propagation_ASM(u_out, feature_size, wavelength, z, linear_conv,
                                padtype, return_H, precomped_H, return_H_exp,
                                precomped_H_exp)

    return u_out


def propagation_ASM_zernike_fourier(u_in, feature_size, wavelength, z, linear_conv=True,
                                    padtype='zero', coeffs=None, zernike=None,
                                    return_H=False, precomped_H=None, return_H_exp=False,
                                    precomped_H_exp=None, dtype=torch.float32):
    """A wrapper around propagation_ASM that applies a Zernike phase correction at Fourier plane

    Inputs
    ------
    coeffs: a 1d tensor of Zernike coefficients
    zernike: a precomputed Zernike function basis. Should contain the same
        number of basis functions as number of coeffs, and be the same height
        and width as u_in. Use compute_zernike_basis to compute

    See propagation_ASM for u_in, feature_size, wavelength, z, linear_conv,
    padtype, return_H, precomped_H, return_H_exp, precomped_H_exp, dtype
    """

    if return_H or return_H_exp or coeffs is None:
        return propagation_ASM(u_in, feature_size, wavelength, z, linear_conv,
                               padtype, return_H, precomped_H, return_H_exp,
                               precomped_H_exp, dtype)

    coeffs = coeffs.reshape(-1, 1, 1)

    if zernike is None:
        # resolution of input field, should be: (num_images, num_channels, height, width, 2)
        zernike = compute_zernike_basis(coeffs.size()[0], u_in.size()[-3:-1])

    if z < 0:
        coeffs = -coeffs

    # create phase
    zernike = combine_zernike_basis(coeffs, zernike)
    zernike = utils.ifftshift(zernike)

    precomped_H_new = zernike * precomped_H

    u_out = propagation_ASM(u_in, feature_size, wavelength, z, linear_conv,
                            padtype, return_H, precomped_H_new, return_H_exp,
                            precomped_H_exp)

    return u_out


def combine_zernike_basis(coeffs, basis, return_phase=False):
    """
    Multiplies the Zernike coefficients and basis functions while preserving
    dimensions

    :param coeffs: torch tensor with coeffs, see propagation_ASM_zernike
    :param basis: the output of compute_zernike_basis, must be same length as coeffs
    :param return_phase:
    :return: A Complex64 tensor that combines coeffs and basis.
    """

    if len(coeffs.shape) < 3:
        coeffs = torch.reshape(coeffs, (coeffs.shape[0], 1, 1))

    # combine zernike basis and coefficients
    zernike = (coeffs * basis).sum(0, keepdim=True)

    # shape to [1, len(coeffs), H, W]
    zernike = zernike.unsqueeze(0)

    # convert to Pytorch Complex tensor
    real, imag = utils.polar_to_rect(torch.ones_like(zernike), zernike)
    return torch.complex(real, imag)


def compute_zernike_basis(num_polynomials, field_res, dtype=torch.float32, wo_piston=False):
    """Computes a set of Zernike basis function with resolution field_res

    num_polynomials: number of Zernike polynomials in this basis
    field_res: [height, width] in px, any list-like object
    dtype: torch dtype for computation at different precision
    """

    # size the zernike basis to avoid circular masking
    zernike_diam = int(np.ceil(np.sqrt(field_res[0]**2 + field_res[1]**2)))

    # create zernike functions

    if not wo_piston:
        zernike = zernikeArray(num_polynomials, zernike_diam)
    else:  # 200427 - exclude pistorn term
        idxs = range(2, 2 + num_polynomials)
        zernike = zernikeArray(idxs, zernike_diam)

    zernike = utils.crop_image(zernike, field_res, pytorch=False)

    # convert to tensor and create phase
    zernike = torch.tensor(zernike, dtype=dtype, requires_grad=False)

    return zernike

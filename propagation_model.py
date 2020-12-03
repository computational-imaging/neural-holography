"""
This is the script that is used for the parameterized wave propagation described in the paper.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Technical Paper:
Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein. Neural Holography with Camera-in-the-loop Training. ACM TOG (SIGGRAPH Asia), 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import utils.utils as utils
from propagation_ASM import compute_zernike_basis, combine_zernike_basis, \
    propagation_ASM, propagation_ASM_zernike, propagation_ASM_zernike_fourier

from utils.pytorch_prototyping.pytorch_prototyping import Conv2dSame


class LatentCodedMLP(nn.Module):
    """
    concatenate latent codes in the middle of forward pass as well.
    put latent codes shape of (1, L, H, W) as a parameter for the forward pass.

    num_latent_codes: list of numbers of slices for each layer
    * so the sum of num_latent_codes should be total number of the latent codes channels
    """
    def __init__(self, num_layers=5, num_features=32, norm=None, num_latent_codes=None):
        super(LatentCodedMLP, self).__init__()

        if num_latent_codes is None:
            num_latent_codes = [0] * num_layers

        assert len(num_latent_codes) == num_layers

        self.num_latent_codes = num_latent_codes
        self.idxs = [sum(num_latent_codes[:y]) for y in range(num_layers + 1)]
        self.nets = nn.ModuleList([])
        num_features = [num_features] * num_layers
        num_features[0] = 1

        # define each layer
        for i in range(num_layers - 1):
            net = [nn.Conv2d(num_features[i] + num_latent_codes[i], num_features[i + 1], kernel_size=1)]
            if norm is not None:
                net += [norm(num_groups=4, num_channels=num_features[i + 1], affine=True)]
            net += [nn.LeakyReLU(0.2, True)]
            self.nets.append(nn.Sequential(*net))

        self.nets.append(nn.Conv2d(num_features[-1] + num_latent_codes[-1], 1, kernel_size=1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.05)

    def forward(self, phases, latent_codes=None):

        after_relu = phases
        # concatenate latent codes at each layer and send through the convolutional layers
        for i in range(len(self.num_latent_codes)):
            if latent_codes is not None:
                after_relu = torch.cat((after_relu, latent_codes[:, self.idxs[i]:self.idxs[i + 1], ...]), 1)
            after_relu = self.nets[i](after_relu)

        # residual connection
        return phases - after_relu


class ContentDependentField(nn.Module):
    def __init__(self, num_layers=5, num_features=32, norm=nn.GroupNorm, latent_coords=False):
        """ Simple 5layers CNN modeling content dependent undiffracted light """

        super(ContentDependentField, self).__init__()

        if not latent_coords:
            first_ch = 1
        else:
            first_ch = 3

        net = [Conv2dSame(first_ch, num_features, kernel_size=3)]

        for i in range(num_layers - 2):
            if norm is not None:
                net += [norm(num_groups=2, num_channels=num_features, affine=True)]
            net += [nn.LeakyReLU(0.2, True),
                    Conv2dSame(num_features, num_features, kernel_size=3)]

        if norm is not None:
            net += [norm(num_groups=4, num_channels=num_features, affine=True)]

        net += [nn.LeakyReLU(0.2, True),
                Conv2dSame(num_features, 2, kernel_size=3)]

        self.net = nn.Sequential(*net)

    def forward(self, phases, latent_coords=None):
        if latent_coords is not None:
            input_cnn = torch.cat((phases, latent_coords), dim=1)
        else:
            input_cnn = phases

        return self.net(input_cnn).unsqueeze(4).permute(0, 4, 2, 3, 1)


class ProcessPhase(nn.Module):
    def __init__(self, num_layers=5, num_features=32, num_output_feat=0, norm=nn.BatchNorm2d, num_latent_codes=0):
        super(ProcessPhase, self).__init__()

        # avoid zero
        self.num_output_feat = max(num_output_feat, 1)
        self.num_latent_codes = num_latent_codes

        # a bunch of 1x1 conv layers, set by num_layers
        net = [nn.Conv2d(1 + num_latent_codes, num_features, kernel_size=1)]

        for i in range(num_layers - 2):
            if norm is not None:
                net += [norm(num_groups=2, num_channels=num_features, affine=True)]
            net += [nn.LeakyReLU(0.2, True),
                    nn.Conv2d(num_features, num_features, kernel_size=1)]

        if norm is not None:
            net += [norm(num_groups=2, num_channels=num_features, affine=True)]

        net += [nn.ReLU(True),
                nn.Conv2d(num_features, self.num_output_feat, kernel_size=1)]

        self.net = nn.Sequential(*net)

    def forward(self, phases):
        return phases - self.net(phases)


class SourceAmplitude(nn.Module):
    def __init__(self, num_gaussians=3, init_sigma=None, init_amp=0.7, x_s0=0.0, y_s0=0.0):
        super(SourceAmplitude, self).__init__()

        self.num_gaussians = num_gaussians

        if init_sigma is None:
            init_sigma = [100.] * self.num_gaussians  # default to 100 for all

        # create parameters for source amplitudes
        self.sigmas = nn.Parameter(torch.tensor(init_sigma),
                                   requires_grad=True)
        self.x_s = nn.Parameter(torch.ones(num_gaussians) * x_s0,
                                requires_grad=True)
        self.y_s = nn.Parameter(torch.ones(num_gaussians) * y_s0,
                                requires_grad=True)
        self.amplitudes = nn.Parameter(torch.ones(num_gaussians) / (num_gaussians) * init_amp,
                                       requires_grad=True)

        self.dc_term = nn.Parameter(torch.zeros(1),
                                    requires_grad=True)

        self.x_dim = None
        self.y_dim = None

    def forward(self, phases):
        # create DC term, then add the gaussians
        source_amp = torch.ones_like(phases) * self.dc_term
        for i in range(self.num_gaussians):
            source_amp += self.create_gaussian(phases.shape, i)

        return source_amp

    def create_gaussian(self, shape, idx):
        # create sampling grid if needed
        if self.x_dim is None or self.y_dim is None:
            self.x_dim = torch.linspace(-(shape[-1] - 1) / 2,
                                        (shape[-1] - 1) / 2,
                                        shape[-1], device=self.dc_term.device)
            self.y_dim = torch.linspace(-(shape[-2] - 1) / 2,
                                        (shape[-2] - 1) / 2,
                                        shape[-2], device=self.dc_term.device)

        if self.x_dim.device != self.sigmas.device:
            self.x_dim.to(self.sigmas.device).detach()
            self.x_dim.requires_grad = False
        if self.y_dim.device != self.sigmas.device:
            self.y_dim.to(self.sigmas.device).detach()
            self.y_dim.requires_grad = False

        # offset grid by coordinate and compute x and y gaussian components
        x_gaussian = torch.exp(-0.5 * torch.pow(torch.div(self.x_dim - self.x_s[idx], self.sigmas[idx]), 2))
        y_gaussian = torch.exp(-0.5 * torch.pow(torch.div(self.y_dim - self.y_s[idx], self.sigmas[idx]), 2))

        # outer product with amplitude scaling
        gaussian = torch.ger(self.amplitudes[idx] * y_gaussian, x_gaussian)

        return gaussian


class ModelPropagate(nn.Module):
    """Parameterized light transport model, propagates a SLM phase with multipart propagation, including
    learnable Zernike phase, source amplitude, and phase LUT corrections, etc....

    Class initialization parameters
    -------------------------------
    distance: propagation dist between SLM and target, in meters, default 0.1
    wavelength: the wavelength of interest, in meters, default 520e-9
    feature_size: the SLM pixel pitch, in meters, default 6.4e-6
    num_coeffs: number of Zernike basis function coeffs to learn, default 15
    num_layers: number of layers in phase LUT correction convnet, default 5
    num_features: number of features per layer of LUT convnet, default 32
    num_output_feat: number of "attention" layers, per-pixel parameters, set 0 if not using. default 0
    num_gaussians: number of Gaussians to use in source amp model, default 3
    init_sigma: initial spread of Gaussians, in pixels, default 100
    learn_dist: if True, makes distance a learnable parameter, default False
    init_coeffs: initial value for Zernike coefficients
    use_conv1d_mlp: if False, disable phase LUT correction, default True
    norm: norm (e.g., nn.BatchNorm2d) to use in LUT convnet, default None
    proptype: chooses the propagation operator ('ASM': propagation_ASM,
        'fresnel': propagation_fresnel). Default ASM.
    linear_conv: if True, pads for linear conv for propagation, default True

    Usage
    -----
    Functions as a pytorch module:

    >>> propagate_model = ModelPropagate(...)
    >>> output_complex = propagate_model(slm_phase)

    slm_phase: encoded phase-only representation at SLM plane , with dimensions
        [batch, 1, height, width]
    output_complex: complex field at the target plane, with dimensions [batch,
        1, height, width, 2], where the final dimension is stacked real and
        imaginary values
    """

    def __init__(self, distance=0.1, wavelength=520e-9, feature_size=6.4e-6, image_res=(1080, 1920), learn_dist=False,
                 target_field=True, num_gaussians=3, init_sigma=(1300.0, 1500.0, 1700.0), init_amp=0.9,
                 num_coeffs=0, num_coeffs_fourier=5, init_coeffs=0.0,
                 use_conv1d_mlp=True, num_layers=3, num_features=16, num_latent_codes=None, norm=nn.GroupNorm,
                 blur=None,
                 content_field=True, num_layers_cdp=5, num_feats_cdp=8, latent_coords=False,
                 proptype='ASM', linear_conv=True):
        super(ModelPropagate, self).__init__()

        # Section 5.1.1. Content-independent Source & Target Field variation
        if num_gaussians:
            self.source_amp = SourceAmplitude(num_gaussians, init_sigma, init_amp=init_amp, x_s0=0.0, y_s0=0.0)
        else:
            self.source_amp = None
        if target_field:
            self.target_constant_amp = nn.Parameter(0.07 * torch.ones(1, 1, *image_res), requires_grad=True)
            self.target_constant_phase = nn.Parameter(torch.zeros((1, 1, *image_res)), requires_grad=True)
        else:
            self.target_constant_amp, self.target_constant_phase = None, None

        # Section 5.1.2 Modeling Optical Propagation with Aberrations
        if num_coeffs:
            self.coeffs = nn.Parameter(torch.ones(num_coeffs) * init_coeffs,
                                       requires_grad=True)
        else:
            self.coeffs = None
        if num_coeffs_fourier:
            self.coeffs_fourier = nn.Parameter(torch.ones(num_coeffs_fourier) * init_coeffs,
                                               requires_grad=True)
        else:
            self.coeffs_fourier = None

        # Section 5.1.3. Phase nonlinearity
        if num_latent_codes is None:
            num_latent_codes = [2, 0, 0]

        if use_conv1d_mlp:
            self.process_phase = LatentCodedMLP(num_layers, num_features, norm=norm, num_latent_codes=num_latent_codes)
        else:
            self.process_phase = None

        if sum(num_latent_codes) > 0:
            self.latent_code = nn.Parameter(torch.zeros(1, sum(num_latent_codes), *image_res), requires_grad=True)
        else:
            self.latent_code = None

        # Section 5.1.4. Content-dependent Undiffracted Light
        if content_field:
            self.content_dependent_field = ContentDependentField(num_layers=num_layers_cdp, num_features=num_feats_cdp, norm=nn.GroupNorm, latent_coords=latent_coords)
        else:
            self.content_dependent_field = None

        if latent_coords:
            latent_x = np.linspace(-1.0, 1.0, image_res[1])
            latent_y = np.linspace(-1.0 * image_res[0] / image_res[1],
                                   1.0 * image_res[0] / image_res[1], image_res[0])
            lx, ly = np.meshgrid(latent_x, latent_y)
            self.latent_coords = nn.Parameter(torch.from_numpy(np.stack((lx, ly), 0)).type(torch.float32).reshape(1, 2, *image_res), requires_grad=False)
        else:
            self.latent_coords = None

        self.learn_dist = learn_dist
        if learn_dist:
            self.distance = nn.Parameter(torch.tensor(distance, dtype=torch.float),
                                         requires_grad=True)
        else:
            self.distance = distance

        if blur is not None:
            self.blur = blur
            self.blur = Conv2dSame(1, 1, kernel_size=3, bias=False)
            self.blur.net[1].weight = nn.Parameter(blur, requires_grad=False)
        else:
            self.blur = None

        # propagation parameters
        self.wavelength = wavelength
        self.feature_size = (feature_size
                             if hasattr(feature_size, '__len__')
                             else [feature_size] * 2)

        self.zernike = None
        self.zernike_fourier = None
        self.zernike_eval = None
        self.zernike_eval_fourier = None
        self.precomped_H = None
        self.precomped_H_exp = None

        # change out the propagation operator
        if proptype == 'ASM':
            self.prop = propagation_ASM
            self.prop_zernike = propagation_ASM_zernike
            self.prop_zernike_fourier = propagation_ASM_zernike_fourier

        self.linear_conv = linear_conv

        # set a device for initializing the precomputed objects
        try:
            self.dev = next(self.parameters()).device
        except StopIteration:  # no parameters
            self.dev = torch.device('cpu')

    def forward(self, phases, skip_lut=False, skip_tm=False):

        # Section 5.1.3. Modeling Phase Nonlinearity
        if self.process_phase is not None and not skip_lut:
            if self.latent_code is not None:
                # support mini-batch
                processed_phase = self.process_phase(phases, self.latent_code.repeat(phases.shape[0], 1, 1, 1))
            else:
                processed_phase = self.process_phase(phases)
        else:
            processed_phase = phases

        # Section 5.1.1. Create Source Amplitude (DC + gaussians)
        if self.source_amp is not None:
            source_amp = self.source_amp(processed_phase)
        else:
            source_amp = torch.ones_like(processed_phase)

        # convert phase to real and imaginary
        real, imag = utils.polar_to_rect(source_amp, processed_phase)
        processed_complex = torch.complex(real, imag)

        # Section 5.1.2. precompute the zernike basis only once
        if self.zernike is None and self.coeffs is not None:
            self.zernike = compute_zernike_basis(self.coeffs.size()[0],
                                                 phases.size()[-2:], wo_piston=True)
            self.zernike = self.zernike.to(self.dev).detach()
            self.zernike.requires_grad = False

        if self.zernike_fourier is None and self.coeffs_fourier is not None:
            self.zernike_fourier = compute_zernike_basis(self.coeffs_fourier.size()[0],
                                                         [i * 2 for i in phases.size()[-2:]],
                                                         wo_piston=True)
            self.zernike_fourier = self.zernike_fourier.to(self.dev).detach()
            self.zernike_fourier.requires_grad = False

        if not self.training and self.zernike_eval is None and self.coeffs is not None:
            # sum the phases
            self.zernike_eval = combine_zernike_basis(self.coeffs, self.zernike)
            self.zernike_eval = self.zernike_eval.to(self.coeffs.device).detach()
            self.zernike_eval.requires_grad = False

        if not self.training and self.zernike_eval_fourier is None and self.coeffs_fourier is not None:
            # sum the phases
            self.zernike_eval_fourier = combine_zernike_basis(self.coeffs_fourier, self.zernike_fourier)
            self.zernike_eval_fourier = utils.ifftshift(self.zernike_eval_fourier)
            self.zernike_eval_fourier = self.zernike_eval_fourier.to(self.coeffs_fourier.device).detach()
            self.zernike_eval_fourier.requires_grad = False

        # precompute the kernel only once
        if self.learn_dist and self.training:
            self.precompute_H_exp(processed_complex)
        else:
            self.precompute_H(processed_complex)

        # Section 5.1.2. apply zernike and propagate
        if self.training:
            if self.coeffs_fourier is None:
                output_complex = self.prop_zernike(processed_complex,
                                                   self.feature_size,
                                                   self.wavelength,
                                                   self.distance,
                                                   coeffs=self.coeffs,
                                                   zernike=self.zernike,
                                                   precomped_H=self.precomped_H,
                                                   precomped_H_exp=self.precomped_H_exp,
                                                   linear_conv=self.linear_conv)
            else:
                output_complex = self.prop_zernike_fourier(processed_complex,
                                                           self.feature_size,
                                                           self.wavelength,
                                                           self.distance,
                                                           coeffs=self.coeffs_fourier,
                                                           zernike=self.zernike_fourier,
                                                           precomped_H=self.precomped_H,
                                                           precomped_H_exp=self.precomped_H_exp,
                                                           linear_conv=self.linear_conv)

        else:
            if self.coeffs is not None:
                # in primal domain
                processed_zernike = self.zernike_eval * processed_complex
            else:
                processed_zernike = processed_complex

            if self.coeffs_fourier is not None:
                # in fourier domain
                precomped_H = self.zernike_eval_fourier * self.precomped_H
            else:
                precomped_H = self.precomped_H

            output_complex = self.prop(processed_zernike,
                                       self.feature_size,
                                       self.wavelength,
                                       self.distance,
                                       precomped_H=precomped_H,
                                       linear_conv=self.linear_conv)

        # Section 5.1.1. Content-independent field at target plane
        if self.target_constant_amp is not None:
            real, imag = utils.polar_to_rect(self.target_constant_amp, self.target_constant_phase)
            target_field = torch.complex(real, imag)
            output_complex = output_complex + target_field

        # Section 5.1.4. Content-dependent Undiffracted light
        if self.content_dependent_field is not None:
            if self.latent_coords is not None:
                cdf = self.content_dependent_field(phases, self.latent_coords.repeat(phases.shape[0], 1, 1, 1))
            else:
                cdf = self.content_dependent_field(phases)
            real, imag = utils.polar_to_rect(cdf[..., 0], cdf[..., 1])
            cdf_rect = torch.complex(real, imag)
            output_complex = output_complex + cdf_rect

        amp = output_complex.abs()
        _, phase = utils.rect_to_polar(output_complex.real, output_complex.imag)

        if self.blur is not None:
            amp = self.blur(amp)

        real, imag = utils.polar_to_rect(amp, phase)

        return torch.complex(real, imag)

    def precompute_H(self, processed_complex):
        if self.precomped_H is None:
            self.precomped_H = self.prop(
                processed_complex,
                self.feature_size,
                self.wavelength,
                self.distance,
                return_H=True,
                linear_conv=self.linear_conv)
            self.precomped_H = self.precomped_H.to(self.dev).detach()
            self.precomped_H.requires_grad = False

    def precompute_H_exp(self, processed_complex):
        if self.precomped_H_exp is None:
            self.precomped_H_exp = self.prop(
                processed_complex,
                self.feature_size,
                self.wavelength,
                self.distance,
                return_H_exp=True,
                linear_conv=self.linear_conv)
            self.precomped_H_exp = self.precomped_H_exp.to(self.dev).detach()
            self.precomped_H_exp.requires_grad = False

    def to(self, *args, **kwargs):
        slf = super().to(*args, **kwargs)
        if slf.zernike is not None:
            slf.zernike = slf.zernike.to(*args, **kwargs)
        if slf.zernike_eval is not None:
            slf.zernike_eval = slf.zernike_eval.to(*args, **kwargs)
        if slf.precomped_H is not None:
            slf.precomped_H = slf.precomped_H.to(*args, **kwargs)
        if slf.precomped_H_exp is not None:
            slf.precomped_H_exp = slf.precomped_H_exp.to(*args, **kwargs)
        # try setting dev based on some parameter, default to cpu
        try:
            slf.dev = next(slf.parameters()).device
        except StopIteration:  # no parameters
            device_arg = torch._C._nn._parse_to(*args, **kwargs)[0]
            if device_arg is not None:
                slf.dev = device_arg
        return slf

    # override default training bool so we can detect eval/train switch
    @property
    def training(self):
        return self._training

    @training.setter
    def training(self, mode):
        if mode:
            self.zernike_eval = None  # reset when switching to training
        self._training = mode

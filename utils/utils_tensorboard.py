"""
This is the script containing all utility functions used for tensorboard.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Technical Paper:
Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein. Neural Holography with Camera-in-the-loop Training. ACM TOG (SIGGRAPH Asia), 2020.
"""

import time
import copy
import torch
import numpy as np
import tensorboardX
import matplotlib.pyplot as plt

import utils.utils as utils
from mpl_toolkits.axes_grid1 import make_axes_locatable

from propagation_ASM import compute_zernike_basis, combine_zernike_basis


class SummaryModelWriter(tensorboardX.SummaryWriter):
    """
    Inherited class of tensorboard summarywriter for visualization of model parameters.

    :param model: ModelPropagate instance that is being trained.
    :param writer_dir: directory of this summary
    :param slm_res: resolution of the SLM (1080, 1920)
    :param roi_res: resolution of the region of interest, default (880, 1600)
    :param ch: an integer indicating training channel (red:0, green:1, blue:2)
    :param kw:
    """
    def __init__(self, model, writer_dir, slm_res=(1080, 1920), roi_res=(880, 1600), ch=1, **kw):
        super(SummaryModelWriter, self).__init__(writer_dir, **kw)
        self.model = model
        self.zernike_basis = None
        self.zernike_basis_fourier = None
        self.slm_res = slm_res
        self.roi_res = roi_res
        self.cmap_rgb = (plt.cm.Reds, plt.cm.Greens, plt.cm.Blues)[ch]

    def visualize_model(self, idx=0):
        """
        Visualize the model parameters on Tensorboard

        :param idx: Global step value to record
        """

        self.add_lut_mean_var(idx)
        self.add_source_amplitude_image(idx)
        self.add_source_amplitude_parameters(idx)
        self.add_zernike(idx)
        self.add_target_field(idx)
        self.add_zeroth_order(idx)
        self.add_latent_codes(idx)

    def add_lut_mean_var(self, idx=0, show_identity=True):
        """
        Add phase-non-linearity lookuptable to Tensorboard

        :param idx: Global step value to record
        :param show_identity: show y=x on the graph
        """
        with torch.no_grad():
            num_x = 64
            if self.model.process_phase is not None:
                lut = copy.deepcopy(self.model.process_phase)
                lut.eval()

                test_phase = torch.linspace(-np.pi, np.pi, num_x, dtype=torch.float)
                input_phase = test_phase.numpy()
                test_phase = test_phase.reshape(num_x, 1, 1, 1)
                output_mean, output_std = np.empty(num_x), np.empty(num_x)
                for v in range(num_x):
                    x = test_phase[v, ...].repeat(1, 1, *self.slm_res).to(self.model.dev)
                    output_phase = lut(x, self.model.latent_code).detach().cpu().numpy().squeeze()
                    output_mean[v] = np.mean(output_phase)
                    output_std[v] = np.std(output_phase)

                fig = plt.figure()
                if show_identity:
                    plt.plot(input_phase, output_mean, 'b',
                             input_phase, input_phase - input_phase[int(num_x / 2)] + output_mean[int(num_x / 2)], 'k--')
                    plt.fill_between(input_phase, output_mean - output_std, output_mean + output_std,
                                     alpha=0.5)
                else:
                    plt.plot(input_phase, output_phase, 'b')

                self.add_figure(f'parameters/voltage-to-phase', fig, idx)
                del lut

    def add_source_amplitude_parameters(self, idx=0):
        """
        Add parameters of gaussian source amplitudes to Tensorboard

        :param idx: Global step value to record
        """
        if self.model.source_amp.num_gaussians > 0:
            sa = self.model.source_amp
            self.add_scalar('parameters_SA/DC', sa.dc_term.cpu().numpy(), idx)
            for i in range(self.model.source_amp.num_gaussians):
                self.add_scalar(f'parameters_SA/Amps_{i}', sa.amplitudes.cpu().numpy()[i], idx)
                self.add_scalar(f'parameters_SA/sigmas_{i}', sa.sigmas.cpu().numpy()[i], idx)
                self.add_scalar(f'parameters_SA/x_{i}', sa.x_s.cpu().numpy()[i], idx)
                self.add_scalar(f'parameters_SA/y_{i}', sa.y_s.cpu().numpy()[i], idx)

    def add_source_amplitude_image(self, idx=0):
        """
        Add visualization of gaussian source amplitudes to Tensorboard

        :param idx: Global step value to record
        """
        if self.model.source_amp is not None:
            img = self.model.source_amp(torch.empty(1, 1, *self.slm_res).
                                        to(self.model.dev)).squeeze().cpu().detach().numpy()
            self.add_figure_cmap(f'parameters/source_amp', img, idx, self.cmap_rgb)

    def add_zernike(self, idx=0, domain='fourier', cm=plt.cm.plasma):
        """
        plot Zernike coeffs as a bar plot and
        plot Zernike map visualization

        :param domain: 'fourier' or 'primal'
        :param idx: Global step value to record
        :param cm: colomap for the zernike, default plasma
        """

        if domain == 'fourier':
            zernike_coeffs = self.model.coeffs_fourier
            map_size = [2160, 3840]
        elif domain == 'primal':
            zernike_coeffs = self.model.coeffs
            map_size = [1080, 1920]

        if zernike_coeffs is not None:
            num_coeffs = len(zernike_coeffs)

            # Zernike coeffs visualization
            x = torch.linspace(0, num_coeffs - 1, num_coeffs)
            fig_zernike_coeffs = plt.figure()
            plt.bar(x.numpy(), zernike_coeffs.cpu().numpy(), width=0.5, align='center')
            self.add_figure(f'parameters/Zernike_coeffs_{domain}', fig_zernike_coeffs, idx)

            # Zernike map visualization
            if domain == 'fourier':
                if self.model.zernike_fourier is None:
                    self.model.zernike_fourier = compute_zernike_basis(self.model.coeffs_fourier.size()[0],
                                                                       map_size, wo_piston=True)
                    self.model.zernike_fourier = self.model.zernike_fourier.to(self.model.dev).detach()
                    self.model.zernike_fourier.requires_grad = False
                zernike_basis = self.model.zernike_fourier
            if domain == 'primal':
                if self.zernike_basis is None:
                    self.model.zernike = compute_zernike_basis(self.model.coeffs.size()[0],
                                                               map_size.size()[-2:], wo_piston=True)
                    self.model.zernike = self.model.zernike.to(self.model.dev).detach()
                    self.model.zernike.requires_grad = False
                zernike_basis = self.model.zernike

            basis_rect = combine_zernike_basis(zernike_coeffs, zernike_basis)
            zernike_phase = basis_rect.angle()
            img_phase = zernike_phase.squeeze().cpu().detach().numpy()
            self.add_figure_cmap(f'parameters/Zernike_map_{domain}', img_phase, idx, cm)

    def add_target_field(self, idx=0):
        """
        Plot u_t, content-independent undiffracted field at the target plane

        :param idx: Global step value to record
        """

        if self.model.target_constant_amp is not None:
            amp = self.model.target_constant_amp
            amp = amp.squeeze().unsqueeze(0).cpu().detach().numpy()
            self.add_figure_cmap(f'parameters/Content-independent_target_amp', amp.squeeze(), idx, self.cmap_rgb)
            self.add_image(f'parameters/Content-independent_target_amp_1080p',
                           ((amp - amp.min())
                            / (amp.max() - amp.min() + 1e-6)), idx)
        if self.model.target_constant_phase is not None:
            phase = self.model.target_constant_phase
            phase = phase.squeeze().unsqueeze(0).cpu().detach().numpy()
            self.add_figure_cmap(f'parameters/Content-independent_target_phase', phase.squeeze(), idx, plt.cm.plasma)
            self.add_image(f'parameters/Content-independent_target_phase_1080p',
                           ((phase - phase.min())
                            / (phase.max() - phase.min() + 1e-6)), idx)

    def add_zeroth_order(self, idx=0):
        """
        Plot output of model with zero-phase input.

        :param idx: Global step value to record
        """

        zero_phase = torch.zeros((1, 1, *self.slm_res)).to(self.model.dev)
        recon_field = self.model(zero_phase)
        recon_amp = recon_field.abs()
        recon_amp = utils.crop_image(recon_amp, self.slm_res,
                                     stacked_complex=False).cpu().detach().squeeze().unsqueeze(0)
        self.add_image(f'parameters/zero_input_1080p', (recon_amp - recon_amp.min())
                       / (recon_amp.max() - recon_amp.min()), idx)
        self.add_figure_cmap(f'parameters/zero_input_figure', recon_amp.squeeze(), idx, self.cmap_rgb)

    def add_latent_codes(self, idx=0, chs=(0, 1)):
        """
        plot latent codes (if exists)

        :param idx: Global step value to record
        :param chs: a list of channel indices to visualize
        """
        if self.model.latent_code is not None:
            for ch in chs:
                lc = self.model.latent_code[0, ch, ...]
                self.add_figure_cmap(f'parameters/latent_code/{ch}', lc.cpu().detach().squeeze().numpy(),
                                     idx, plt.cm.plasma)

    def add_content_dependent_field(self, phase, idx=0):
        if self.model.content_dependent_field is not None:
            cdf = self.model.content_dependent_field(phase, self.model.latent_coords)
            cdf_amp, cdf_phase = cdf[..., 0], cdf[..., 1]
            cdf_amp = cdf_amp.cpu().detach().squeeze().unsqueeze(0)
            cdf_phase = cdf_phase.cpu().detach().squeeze().unsqueeze(0)

            self.add_figure_cmap(f'parameters/content_dependent_amp', cdf_amp, idx, self.cmap_rgb)
            self.add_figure_cmap(f'parameters/content_dependent_phase', cdf_phase, idx, plt.cm.plasma)

    def add_figure_cmap(self, title, img, idx, cmap=plt.cm.plasma):
        figure = plt.figure()
        p = plt.imshow(img.squeeze())
        p.set_cmap(cmap)
        colorbar(p)
        self.add_figure(title, figure, idx)


def colorbar(mappable):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

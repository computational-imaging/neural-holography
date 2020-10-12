"""
Some modules for easy use. (No need to calculate kernels explicitly)

"""
import torch
import torch.nn as nn
from algorithms import gerchberg_saxton, stochastic_gradient_descent, double_phase_amplitude_coding


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
            self.precomputed_H_f = self.prop(torch.empty(*init_phase.shape, 2), self.feature_size,
                                             self.wavelength, self.prop_dist, return_H=True)
            self.precomputed_H_f = self.precomputed_H_f.to(self.dev).detach()
            self.precomputed_H_f.requires_grad = False

        if self.precomputed_H_b is None and self.prop_model == 'ASM':
            self.precomputed_H_b = self.prop(torch.empty(*init_phase.shape, 2), self.feature_size,
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
                 propagator=None, loss=nn.MSELoss(), lr=0.01, lr_s=0.003, s0=1.0, writer=None,
                 device=torch.device('cuda')):
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
            self.precomputed_H = self.prop(torch.empty(*init_phase.shape, 2), self.feature_size,
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


class DPAC(nn.Module):
    """Double-phase Amplitude Coding

    Class initialization parameters
    -------------------------------
    :param prop_dist: propagation dist between SLM and target, in meters
    :param wavelength: the wavelength of interest, in meters
    :param feature_size: the SLM pixel pitch, in meters
    :param prop_model: chooses the propagation operator ('ASM': propagation_ASM,
        'model': calibrated model). Default 'ASM'.
    :param propagator: propagator instance (function / pytorch module)
    :param device: torch.device

    Usage
    -----
    Functions as a pytorch module:

    >>> dpac = DPAC(...)
    >>> _, final_phase = dpac(target_amp, target_phase)

    target_amp: amplitude at the target plane, with dimensions [batch, 1, height, width]
    target_amp (optional): phase at the target plane, with dimensions [batch, 1, height, width]
    final_phase: optimized phase-only representation at SLM plane, same dimensions

    """
    def __init__(self, prop_dist, wavelength, feature_size, prop_model='ASM', propagator=None,
                 device=torch.device('cuda')):
        """

        """
        super(DPAC, self).__init__()

        # propagation is from target to SLM plane (one step)
        self.prop_dist = -prop_dist
        self.wavelength = wavelength
        self.feature_size = feature_size
        self.precomputed_H = None
        self.prop_model = prop_model
        self.prop = propagator
        self.dev = device

    def forward(self, target_amp, target_phase=None):
        if target_phase is None:
            target_phase = torch.zeros_like(target_amp)

        if self.precomputed_H is None and self.prop_model == 'ASM':
            self.precomputed_H = self.prop(torch.empty(*target_amp.shape, 2), self.feature_size,
                                           self.wavelength, self.prop_dist, return_H=True)
            self.precomputed_H = self.precomputed_H.to(self.dev).detach()
            self.precomputed_H.requires_grad = False

        final_phase = double_phase_amplitude_coding(target_phase, target_amp, self.prop_dist,
                                                    self.wavelength, self.feature_size,
                                                    prop_model=self.prop_model, propagator=self.prop,
                                                    precomputed_H=self.precomputed_H)
        return None, final_phase

    @property
    def phase_path(self):
        return self._phase_path

    @phase_path.setter
    def phase_path(self, phase_path):
        self._phase_path = phase_path

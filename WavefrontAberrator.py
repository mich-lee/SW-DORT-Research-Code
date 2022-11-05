import warnings
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.Optical_Components.CGH_Component import CGH_Component
from holotorch.utils.Dimensions import HW, TensorDimension
from holotorch.utils.Enumerators import *
from holotorch.utils.Helper_Functions import applyFilterSpaceDomain, ft2, ift2
from holotorch.Optical_Components.SimpleMask import SimpleMask
from holotorch.Optical_Propagators.ASM_Prop import ASM_Prop


class WavefrontAberrator(CGH_Component):
	def __init__(	self,
					model : torch.nn.Module,
					direction_label : str = '',
					parameterDict : dict = None
				) -> None:
		super().__init__()
		self.model = model
		self.direction_label = direction_label
		self.parameterDict = parameterDict

	def forward(self, field : ElectricField) -> ElectricField:
		return self.model(field)


class WavefrontAberratorGenerator:
	def __init__(	self,
					meanFreePath	: float,
					screenSigma		: float,
					numLayers		: float,
					resolution		: list or tuple,
					elementSpacings : list or tuple,
					reusePropagator	: bool = True,
					device			: torch.device = None,
					gpu_no			: int = 0,
					use_cuda		: bool = False
				) -> None:

		def validate2TupleList(l):
			# Not bothering to check if elements of l are numbers.
			if (type(l) is not list) and (type(l) is not tuple):
				return False
			if (len(l) != 0) and (len(l) != 2):
				return False
			return True

		if (not validate2TupleList(resolution)):
			raise Exception("'resolution' must be a tuple or list with two positive integer elements.")
		if (not validate2TupleList(elementSpacings)):
			raise Exception("'elementSpacings' must be a tuple or list with two positive real number elements.")

		if (device != None):
			self.device = device
		else:
			self.device = torch.device("cuda:"+str(gpu_no) if use_cuda else "cpu")
			self.gpu_no = gpu_no
			self.use_cuda = use_cuda

		self.meanFreePath = meanFreePath
		self.screenSigma = screenSigma
		self.numLayers = numLayers
		self.resolution = resolution
		self.elementSpacings = elementSpacings
		self._reusePropagator = reusePropagator

		# This dictionary will be passed to the WavefrontAberrator components that this
		# WavefrontAberratorGenerator object creates.  That way, the generated WavefrontAberrator
		# objects---which, unlike this object, may be included in saved models---will still contain
		# information about the parameters.
		self._parameterDict =	{
									'meanFreePath'		: meanFreePath,
									'screenSigma'		: screenSigma,
									'numLayers'			: numLayers,
									'resolution'		: resolution,
									'elementSpacings'	: elementSpacings,
									'reusePropagator'	: reusePropagator,
									'device'			: device,
									'gpu_no'			: gpu_no,
									'use_cuda'			: use_cuda
								}

		self._initializeModel()

	def _initializeModel(self):
		self._initializeFrequencyGrids()

		if (self._reusePropagator):
			prop = ASM_Prop(init_distance=self.meanFreePath)

		model = torch.nn.Sequential()
		for i in range(self.numLayers):
			model.append(self._generatePhaseScreen(self.screenSigma))
			if (self._reusePropagator):
				model.append(prop)
			else:
				model.append(ASM_Prop(init_distance=self.meanFreePath))
		model.append(self._generatePhaseScreen(self.screenSigma))

		modelReversed = torch.nn.Sequential()
		for i in range(len(model) - 1, -1, -1):
			modelReversed.append(model[i])

		self._modelSequential = model
		self._modelReversedSequential = modelReversed
		self.model = WavefrontAberrator(model=model, direction_label='normal', parameterDict=self._parameterDict)
		self.modelReversed = WavefrontAberrator(model=modelReversed, direction_label='reverse', parameterDict=self._parameterDict)

	def _initializeFrequencyGrids(self):
		def create_normalized_grid(H, W, device):
			# precompute frequency grid for ASM defocus kernel
			with torch.no_grad():
				# Creates the frequency coordinate grid in x and y direction
				kx = (torch.linspace(0, H - 1, H) - (H // 2)) / H
				ky = (torch.linspace(0, W - 1, W) - (W // 2)) / W
				Kx, Ky = torch.meshgrid(kx, ky)
				return Kx.to(device=device), Ky.to(device=device)
		self._Kx, self._Ky = create_normalized_grid(self.resolution[0], self.resolution[1], self.device)
		self._Kx = 2*np.pi * self._Kx / self.elementSpacings[0]
		self._Ky = 2*np.pi * self._Ky / self.elementSpacings[1]

	@classmethod
	def _generateGaussianInFreq(cls, sigma, Kx, Ky):
		K_transverse = torch.sqrt(Kx**2 + Ky**2)
		return torch.exp((-1/2)*(K_transverse**2)/(sigma**2))

	def _generatePhaseScreen(self, sigma):
		screen = SimpleMask(
								tensor_dimension=HW(self.resolution[0], self.resolution[1]),
								init_type=INIT_TYPE.ZEROS,
								mask_model_type=MASK_MODEL_TYPE.COMPLEX,
								mask_forward_type=MASK_FORWARD_TYPE.MULTIPLICATIVE,
								mask_opt=False
							)
		phaseMask = 2 * np.pi * torch.rand(screen.mask.shape, dtype=torch.float, device=self.device)
		H = WavefrontAberratorGenerator._generateGaussianInFreq(sigma, self._Kx, self._Ky).to(device=self.device)
		h = ift2(H, norm='backward').real
		phaseMask = applyFilterSpaceDomain(h, phaseMask).real
		screen.mask = torch.exp(1j*phaseMask).to(device=self.device)
		return screen

	def get_model(self):
		return self.model

	def get_model_reversed(self):
		return self.modelReversed
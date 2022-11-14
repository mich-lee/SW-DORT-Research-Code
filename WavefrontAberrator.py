import numpy as np
import torch
import matplotlib.pyplot as plt

import warnings
import sys

from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.Optical_Components.CGH_Component import CGH_Component
from holotorch.utils.Dimensions import HW, TensorDimension
from holotorch.Spectra.SpacingContainer import SpacingContainer
from holotorch.utils.Enumerators import *
from holotorch.utils.Helper_Functions import conv, applyFilterSpaceDomain, ft2, ift2
from holotorch.Optical_Components.SimpleMask import SimpleMask
from holotorch.Optical_Propagators.ASM_Prop import ASM_Prop

################################################################################################################################

# This class should not be instantiated manually.  Its subclasses should be instantiated instead.
class WavefrontAberratorGenerator:
	def __init__(	self,
					resolution				: list or tuple,
					elementSpacings 		: list or tuple,
					generateBidirectional	: bool = False,
					device					: torch.device = None,
					gpu_no					: int = 0,
					use_cuda				: bool = False
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

		self.resolution = resolution
		self.elementSpacings = elementSpacings
		self.generateBidirectional = generateBidirectional

		if (device != None):
			self.device = device
		else:
			self.device = torch.device("cuda:"+str(gpu_no) if use_cuda else "cpu")
			self.gpu_no = gpu_no
			self.use_cuda = use_cuda

		# This dictionary will be passed to the WavefrontAberrator components that this
		# WavefrontAberratorGenerator object creates.  That way, the generated WavefrontAberrator
		# objects---which, unlike this object, may be included in saved models---will still contain
		# information about the parameters.
		self._parameterDict =	{
									'modelType'					: self.modelType,
									'resolution'				: resolution,
									'elementSpacings'			: elementSpacings,
									'generateBidirectional'		: generateBidirectional,
									'device'					: device,
									'gpu_no'					: gpu_no,
									'use_cuda'					: use_cuda,
								}
								
		self._initializeFrequencyGrids()
		self._initializeModel()

	def get_model(self):
		return self.model

	def get_model_reversed(self):
		return self.modelReversed

	def _initializeFrequencyGrids(self):
		self._Kx, self._Ky = WavefrontAberratorGenerator._create_normalized_grid(self.resolution[0], self.resolution[1], self.device)
		self._Kx = 2*np.pi * self._Kx / self.elementSpacings[0]
		self._Ky = 2*np.pi * self._Ky / self.elementSpacings[1]

	@classmethod
	def _generateGaussian(cls, sigma, Kx, Ky, domain='space', device='cpu'):
		K_transverse = torch.sqrt(Kx**2 + Ky**2)
		if (domain == 'space'):
			return torch.exp((-1/2)*(K_transverse**2)/(sigma**2)).to(device=device)
		elif (domain == 'frequency'):
			return torch.exp((-1/2)*(sigma**2)*(K_transverse**2)).to(device=device)
		else:
			raise Exception("Invalid value for 'domain' argument.")

	@classmethod
	def _create_normalized_grid(cls, H, W, device):
			# precompute frequency grid for ASM defocus kernel
			with torch.no_grad():
				# Creates the frequency coordinate grid in x and y direction
				kx = (torch.linspace(0, H - 1, H) - (H // 2)) / H
				ky = (torch.linspace(0, W - 1, W) - (W // 2)) / W
				Kx, Ky = torch.meshgrid(kx, ky)
				return Kx.to(device=device), Ky.to(device=device)




class WavefrontAberrator(CGH_Component):
	def __init__(	self,
					direction_label : str = '',
					parameterDict : dict = None
				) -> None:
		super().__init__()
		self.direction_label = direction_label
		self.parameterDict = parameterDict
		self.spacing = SpacingContainer(spacing=list(self.parameterDict['elementSpacings']))
		self.resolution = self.parameterDict['resolution']

	def _check_field_dimensions_valid_bool(self, field : ElectricField) -> bool:
		if not field.spacing.is_equivalent(self.spacing):
			return False
		if (field.data.shape[-2] != self.resolution[0]) or (field.data.shape[-1] != self.resolution[1]):
			return False
		return True

	def _check_field_dimensions_valid(self, field : ElectricField) -> bool:
		if not self._check_field_dimensions_valid_bool(field):
			raise Exception("Input field does not match the resolution and/or spacing specified.")

	# Deprecated.  This method is only here to provide backwards compatibility with a previously saved model.
	def forward(self, field : ElectricField) -> ElectricField:
		if not hasattr(self, '_warnFlag'):
			warnings.warn("Should not be using WavefrontAberrator's forward(...) method.")
			self._warnFlag = True
		self._check_field_dimensions_valid(field)
		return self.model(field)




class RandomPhaseScreen(WavefrontAberrator):
	def __init__(	self,
					model : torch.nn.Module,
					direction_label : str = '',
					parameterDict : dict = None
				) -> None:
		super().__init__(
			direction_label = direction_label,
			parameterDict = parameterDict
		)
		self.model = model

	def forward(self, field : ElectricField) -> ElectricField:
		self._check_field_dimensions_valid(field)
		return self.model(field)




class RandomThicknessScreen(WavefrontAberrator):
	def __init__(	self,
					thicknesses : torch.Tensor,
					n_screen : float = None,
					direction_label : str = '',
					parameterDict : dict = None
				) -> None:
		super().__init__(
			direction_label = direction_label,
			parameterDict = parameterDict
		)
		self.thicknesses = thicknesses
		self.n_screen = n_screen

	def forward(self, field : ElectricField) -> ElectricField:
		self._check_field_dimensions_valid(field)

		# pathLen = 

		return self.model(field)




class RandomThicknessScreenGenerator(WavefrontAberratorGenerator):
	"""
	Description:
		Implements asdasfaasdasdadassdas
	
	References:
		- asdsadasdasdasdasd
	"""
	def __init__(	self,
					resolution				: list or tuple,
					elementSpacings 		: list or tuple,
					meanThickness			: float,
					thicknessVariance		: float,
					correlationLength		: float,
					n_screen				: float = None,
					reuseScreen				: bool = True,
					generateBidirectional	: bool = False,
					device					: torch.device = None,
					gpu_no					: int = 0,
					use_cuda				: bool = False
				) -> None:

		self.modelType = 'RandomThicknessScreen'

		super.__init__(
			resolution				= resolution,
			elementSpacings			= elementSpacings,
			generateBidirectional	= generateBidirectional,
			device					= device,
			gpu_no					= gpu_no,
			use_cuda				= use_cuda
		)

		self.meanThickness = meanThickness
		self.thicknessVariance = thicknessVariance
		self.correlationLength = correlationLength
		self.n_screen = n_screen
		self.reuseScreen = reuseScreen

		self._parameterDict['meanThickness']		= meanThickness
		self._parameterDict['thicknessVariance']	= thicknessVariance
		self._parameterDict['correlationLength']	= correlationLength
		self._parameterDict['n_screen']				= n_screen
		self._parameterDict['reuseScreen']			= reuseScreen


	def _initializeModel(self):
		self._initializeRandomHeights()
		
		if not self.reuseScreens:
			self.model = RandomThicknessScreen(thicknesses=self.thicknesses, n_screen=self.n_screen, direction_label='normal', parameterDict=self._parameterDict)
		else:
			self.model = RandomThicknessScreen(thicknesses=self.thicknesses, n_screen=self.n_screen, direction_label='both', parameterDict=self._parameterDict)

		if self.generateBidirectional:
			if not self.reuseScreen:
				self.modelReversed = RandomThicknessScreen(thicknesses=self.thicknesses, n_screen=self.n_screen, direction_label='reverse', parameterDict=self._parameterDict)
			else:
				self.modelReversed = self.model
		else:
			self.modelReversed = None

	def _initializeRandomHeights(self):
		def generateRandomHeightsHelper1(filterSigma, kx, ky, padding, meanThickness, thicknessVariance, device):
			heights0 = generateRandomHeightsHelper2(filterSigma, kx, ky, thicknessVariance, device)
			heights1 = torch.zeros_like(heights0)
			heights1[padding[0]:-padding[1], padding[2]:-padding[3]] = torch.flip(heights0[padding[0]:-padding[1], padding[2]:-padding[3]], [0,1])

			# Element corresponding to (0,0) not perfectly centered --> need to roll tensor to align (0,0) elements of heights0 and heights1,
			# otherwise autocorrelation will be shifted.
			# Note that if padding is zero (should not occur), one will get wraparound and throw off (probably only slightly) the autocorrelation result.
			heights1 = torch.roll(heights1, (1,1), (0,1))

			# Because of the way 'heights0' and 'heights1' were set up, this should hopefully approximate an unbiased autocorrelation
			ht_autocorr = conv(heights0, heights1)
			ht_autocorr = ht_autocorr[padding[0]:-padding[1], padding[2]:-padding[3]].real

			ht = heights0[padding[0]:-padding[1], padding[2]:-padding[3]]

			# Fixing the variance and removing the mean (again, and somewhat redundantly)
			ht_var = ((ht - ht.mean()) ** 2).sum() / ht.numel()
			ht = (ht - ht.mean()) * np.sqrt(thicknessVariance) / torch.sqrt(ht_var)

			# Adding the mean
			ht = ht + meanThickness
			
			return ht, ht_autocorr
			
		def generateRandomHeightsHelper2(filterSigma, kx, ky, thicknessVariance, device):
			ht = torch.randn(kx.shape[-2], kx.shape[-1], dtype=torch.float, device=device)
			H = WavefrontAberratorGenerator._generateGaussian(filterSigma, kx, ky, domain='frequency', device=self.device)
			h = ift2(H, norm='backward').real
			ht = applyFilterSpaceDomain(h, ht).real

			# Fixing the variance and removing the mean
			ht_var = ((ht - ht.mean()) ** 2).sum() / ht.numel()
			ht = (ht - ht.mean()) * np.sqrt(thicknessVariance) / torch.sqrt(ht_var)

			return ht
		
		pad_x1 = int(np.floor(self.resolution[0] / 2))
		pad_x2 = int(np.ceil(self.resolution[0] / 2))
		pad_y1 = int(np.floor(self.resolution[1] / 2))
		pad_y2 = int(np.ceil(self.resolution[1] / 2))
		padding = (pad_x1, pad_x2, pad_y1, pad_y2)
		H = self.resolution[0] + pad_x1 + pad_x2
		W = self.resolution[1] + pad_y1 + pad_y2

		# Making the grids twice as big to facilitate removing bias in the autocorrelation later
		kxTemp, kyTemp = WavefrontAberratorGenerator._create_normalized_grid(H, W, self.device)
		kxTemp = 2*np.pi * kxTemp / self.elementSpacings[0]
		kyTemp = 2*np.pi * kyTemp / self.elementSpacings[1]

		# xGridNorm, yGridNorm = WavefrontAberratorGenerator._create_normalized_grid(self.resolution[0], self.resolution[1], self.device)
		# xGrid = xGridNorm * self.resolution[0] * self.elementSpacings[0]
		# yGrid = yGridNorm * self.resolution[1] * self.elementSpacings[1]

		# The 'filterSigma' argument is set to self.correlationLength/2.  This was obtained by considering that autocorrelation is
		# |H(e^{j\omega})|^2\Phi_{xx}(e^{j\omega}) in frequency.  By looking at the Fourier transform pair for a Gaussian, it can be
		# seen that letting filterSigma=self.correlationLength/2 results in the autocorrelation dropping to 1/e times its max value
		# at a distance of self.correlationLength from the origin (assuming that X is a random signal drawn from a zero-mean Gaussian distribution).
		# Note that autocorrelation in this context refers to autocorrelation with means removed.
		heights, heightAutocorr = generateRandomHeightsHelper1(self.correlationLength / 2, kxTemp, kyTemp, padding, self.meanThickness, self.thicknessVariance, self.device)
		self.thicknesses = heights

		return




class RandomPhaseScreenGenerator(WavefrontAberratorGenerator):
	"""
	Description:
		Implements a random phase screen model.
	
	References:
		- "Realistic phase screen model for forward multiple-scattering media" by Mu Qiao and Xin Yuan
			- NOTE: Did not implement what was done in the paper; referred to explanation of conventional random phase screens.
		- "Characterization of the angular memory effect of scattered light in biological tissues" by Schott et al.
	"""
	def __init__(	self,
					resolution				: list or tuple,
					elementSpacings 		: list or tuple,
					meanFreePath			: float,
					screenGaussianSigma		: float,
					numLayers				: int,
					reusePropagator			: bool = True,
					generateBidirectional	: bool = False,
					device					: torch.device = None,
					gpu_no					: int = 0,
					use_cuda				: bool = False
				) -> None:

		self.modelType = 'RandomPhaseScreen'

		super.__init__(
			resolution				= resolution,
			elementSpacings			= elementSpacings,
			generateBidirectional	= generateBidirectional,
			device					= device,
			gpu_no					= gpu_no,
			use_cuda				= use_cuda
		)

		self.meanFreePath = meanFreePath
		self.screenGaussianSigma = screenGaussianSigma
		self.numLayers = numLayers
		self.reusePropagator = reusePropagator

		self._parameterDict['meanFreePath']				= meanFreePath
		self._parameterDict['screenGaussianSigma']		= screenGaussianSigma
		self._parameterDict['numLayers']				= numLayers
		self._parameterDict['reusePropagator']			= reusePropagator


	def _initializeModel(self):
		if (self.reusePropagator):
			prop = ASM_Prop(init_distance=self.meanFreePath)

		model = torch.nn.Sequential()
		for i in range(self.numLayers):
			model.append(self._generatePhaseScreen(self.screenGaussianSigma))
			if (self.reusePropagator):
				model.append(prop)
			else:
				model.append(ASM_Prop(init_distance=self.meanFreePath))
		model.append(self._generatePhaseScreen(self.screenGaussianSigma))

		self._modelSequential = model
		self.model = RandomPhaseScreen(model=model, direction_label='normal', parameterDict=self._parameterDict)

		if self.generateBidirectional:
			modelReversed = torch.nn.Sequential()
			for i in range(len(model) - 1, -1, -1):
				modelReversed.append(model[i])

			self._modelReversedSequential = modelReversed
			self.modelReversed = RandomPhaseScreen(model=modelReversed, direction_label='reverse', parameterDict=self._parameterDict)
		else:
			self._modelReversedSequential = None
			self.modelReversed = None


	def _generatePhaseScreen(self, sigma):
		screen = SimpleMask(
								tensor_dimension=HW(self.resolution[0], self.resolution[1]),
								init_type=INIT_TYPE.ZEROS,
								mask_model_type=MASK_MODEL_TYPE.COMPLEX,
								mask_forward_type=MASK_FORWARD_TYPE.MULTIPLICATIVE,
								mask_opt=False
							)
		phaseMask = 2 * np.pi * torch.rand(screen.mask.shape, dtype=torch.float, device=self.device)

		# Want sigma to correspond to width in frequency so setting domain='space'
		H = WavefrontAberratorGenerator._generateGaussian(sigma, self._Kx, self._Ky, domain='space', device=self.device)
		
		h = ift2(H, norm='backward').real
		phaseMask = applyFilterSpaceDomain(h, phaseMask).real
		screen.mask = torch.exp(1j*phaseMask).to(device=self.device)
		return screen
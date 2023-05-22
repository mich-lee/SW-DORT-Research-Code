import numpy as np
import torch
import matplotlib.pyplot as plt

import warnings
import sys

from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.Optical_Components.CGH_Component import CGH_Component
from holotorch.utils.Dimensions import HW, BTPCHW
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
		self.spacing.to(field.data.device)
		if not field.spacing.is_equivalent(self.spacing):
			return False
		if (field.data.shape[-2] != self.resolution[0]) or (field.data.shape[-1] != self.resolution[1]):
			return False
		return True

	def _check_field_dimensions_valid(self, field : ElectricField) -> bool:
		if not self._check_field_dimensions_valid_bool(field):
			raise Exception("Input field does not match the resolution and/or spacing specified.")

	def get_thickness(self):
		raise Exception("The 'getThickness' method should be implemented in a subclass.")

	# Deprecated.  This method is only here to provide backwards compatibility with a previously saved model.
	def forward(self, field : ElectricField) -> ElectricField:
		raise Exception("Should not be using WavefrontAberrator's forward(...) method.")
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

	def get_thickness(self):
		return self.parameterDict['meanFreePath'] * self.parameterDict['numLayers']

	def forward(self, field : ElectricField) -> ElectricField:
		self._check_field_dimensions_valid(field)
		return self.model(field)




class RandomThicknessScreen(WavefrontAberrator):
	def __init__(	self,
					thicknesses : torch.Tensor,
					max_thickness : float,
					n_screen : float,
					n_ambient : float,
					sign_convention : ENUM_PHASE_SIGN_CONVENTION = ENUM_PHASE_SIGN_CONVENTION.TIME_PHASORS_ROTATE_CLOCKWISE,
					direction_label : str = '',
					parameterDict : dict = None
				) -> None:
		super().__init__(
			direction_label = direction_label,
			parameterDict = parameterDict
		)
		if (sign_convention != ENUM_PHASE_SIGN_CONVENTION.TIME_PHASORS_ROTATE_CLOCKWISE):
			raise Exception("The provided sign_convention has not been implemented.")
		self.thicknesses = thicknesses
		self.max_thickness = max_thickness
		self.n_screen = n_screen
		self.n_ambient = n_ambient
		self.sign_convention = sign_convention

	def get_thickness(self):
		return self.max_thickness

	def forward(self, field : ElectricField) -> ElectricField:
		self._check_field_dimensions_valid(field)

		lambda0 = field.wavelengths.data_tensor.view(field.wavelengths.tensor_dimension.get_new_shape(new_dim=BTPCHW))
		thicknesses = self.thicknesses[None, None, None, None, :, :]

		l_screen = thicknesses
		l_ambient = self.max_thickness - thicknesses
		k_screen = 2 * np.pi / (lambda0 / self.n_screen)
		k_ambient = 2 * np.pi / (lambda0 / self.n_ambient)

		# Using the convention that phasors in time rotate clockwise (so delays are positive phase shifts)
		delta_phi_screen = (k_screen * l_screen) + (k_ambient * l_ambient)
		screen = torch.exp(1j * delta_phi_screen)

		fieldOut = ElectricField(
									data = field.data * screen,
									wavelengths = field.wavelengths,
									spacing = field.spacing
								)

		return fieldOut




class RandomThicknessScreenGenerator(WavefrontAberratorGenerator):
	"""
	Description:
		Implements the model described in Sections 7.1.1 and Equation 8.2-2 in Section 8.2 in Statistical Optics (2nd Edition) by Joseph W. Goodman.
			- Note that the transmittance function B(x, y) is NOT implemented.
		Generates random heights using surface roughness statistics.
	
	References:
		- Section 7.1.1 in Statistical Optics (2nd Edition) by Joseph W. Goodman
		- "Optical quality of the eye lens surfaces from roughness and diffusion measurements" by Navarro et al
			- This gives examples of surface roughness parameters
		- https://www.newfor.net/wp-content/uploads/2015/02/DL15-NEWFOR_Roughness_state_of_the_art.pdf
			- This defines correlation length
	"""
	def __init__(	self,
					resolution						: list or tuple,
					elementSpacings 				: list or tuple,
					n_screen						: float,
					surfaceVariationStdDev			: float,
					correlationLength				: float,
					maxThickness					: float,
					minThickness					: float = 0,
					thicknessVariationMaxRange		: float = None,
					n_ambient						: float = 1,	# Assume free space
					doubleSidedRoughness			: bool = True,
					reuseScreenForBidirectional		: bool = True,
					generateBidirectional			: bool = False,
					device							: torch.device = None,
					gpu_no							: int = 0,
					use_cuda						: bool = False
				) -> None:

		self.modelType = 'RandomThicknessScreen'

		super().__init__(
			resolution				= resolution,
			elementSpacings			= elementSpacings,
			generateBidirectional	= generateBidirectional,
			device					= device,
			gpu_no					= gpu_no,
			use_cuda				= use_cuda
		)

		self.n_screen = n_screen
		self.n_ambient = n_ambient
		self.surfaceVariationStdDev = surfaceVariationStdDev
		self.surfaceVariationVariance = surfaceVariationStdDev**2
		self.correlationLength = correlationLength
		self.maxThickness = maxThickness
		self.minThickness = minThickness
		if (thicknessVariationMaxRange is None):
			self.thicknessVariationMaxRange = 10 * surfaceVariationStdDev	# I.e. within +/- 5 standard deviations of the mean
		else:
			self.thicknessVariationMaxRange = thicknessVariationMaxRange
		self.doubleSidedRoughness = doubleSidedRoughness
		self.reuseScreenForBidirectional = reuseScreenForBidirectional

		self._parameterDict['n_screen']							= n_screen
		self._parameterDict['n_ambient']						= n_ambient
		self._parameterDict['surfaceVariationStdDev']			= surfaceVariationStdDev
		self._parameterDict['surfaceVariationVariance']			= self.surfaceVariationVariance
		self._parameterDict['correlationLength']				= correlationLength
		self._parameterDict['maxThickness']						= maxThickness
		self._parameterDict['minThickness']						= minThickness
		self._parameterDict['thicknessVariationMaxRange']		= thicknessVariationMaxRange
		self._parameterDict['doubleSidedRoughness']				= doubleSidedRoughness
		self._parameterDict['reuseScreenForBidirectional']		= reuseScreenForBidirectional

		self._initializeModel()


	def _initializeModel(self):
		self._initializeRandomHeights()
		
		if not self.reuseScreenForBidirectional:
			self.model = RandomThicknessScreen(thicknesses=self.thicknesses, max_thickness=self.maxThickness, n_screen=self.n_screen, n_ambient=self.n_ambient, direction_label='normal', parameterDict=self._parameterDict)
		else:
			self.model = RandomThicknessScreen(thicknesses=self.thicknesses, max_thickness=self.maxThickness, n_screen=self.n_screen, n_ambient=self.n_ambient, direction_label='both', parameterDict=self._parameterDict)

		if self.generateBidirectional:
			if not self.reuseScreenForBidirectional:
				self.modelReversed = RandomThicknessScreen(thicknesses=self.thicknesses, max_thickness=self.maxThickness, n_screen=self.n_screen, n_ambient=self.n_ambient, direction_label='reverse', parameterDict=self._parameterDict)
			else:
				self.modelReversed = self.model
		else:
			self.modelReversed = None

	def _initializeRandomHeights(self):
		# This generates random heights based on the surface roughness parameters.
		# The parameters are the standard deviation of the surface roughness heights and the correlation length.
		# 	- Note that skew in the surface roughness height probability density functions (PDF) is NOT modeled here
		#
		# For more information on surface roughness, see:
		# 	- "Optical quality of the eye lens surfaces from roughness and diffusion measurements" by Navarro et al
		#		- This gives examples of surface roughness parameters
		#	- https://www.newfor.net/wp-content/uploads/2015/02/DL15-NEWFOR_Roughness_state_of_the_art.pdf
		#		- This defines correlation length

		def generateRandomHeightsHelper1(filterSigma, kx, ky, padding, surfaceVariance, device):
			heights0 = generateRandomHeightsHelper2(filterSigma, kx, ky, device)
			heights1 = torch.zeros_like(heights0)
			heights1[padding[0]:-padding[1], padding[2]:-padding[3]] = torch.flip(heights0[padding[0]:-padding[1], padding[2]:-padding[3]], [0,1])

			# Element corresponding to (0,0) not perfectly centered --> need to roll tensor to align (0,0) elements of heights0 and heights1,
			# otherwise autocorrelation will be shifted.
			# Note that if padding is zero (should not occur), one will get wraparound and throw off (probably only slightly) the autocorrelation result.
			heights1 = torch.roll(heights1, (1,1), (0,1))

			# Because of the way 'heights0' and 'heights1' were set up (i.e. heights1 being a truncated and padded version of heights1),
			# this should hopefully approximate an unbiased autocorrelation
			ht_autocorr = conv(heights0, heights1, use_inplace_ffts=True)
			ht_autocorr = ht_autocorr[padding[0]:-padding[1], padding[2]:-padding[3]].real

			ht = heights0[padding[0]:-padding[1], padding[2]:-padding[3]]

			# Fixing the variance and removing the mean
			ht_var = ((ht - ht.mean()) ** 2).sum() / ht.numel()
			ht = (ht - ht.mean()) * np.sqrt(surfaceVariance) / torch.sqrt(ht_var)
			
			return ht, ht_autocorr
			
		def generateRandomHeightsHelper2(filterSigma, kx, ky, device):
			# NOTE: Asymmetry in roughness height PDF is NOT modeled here.
			ht = torch.randn(kx.shape[-2], kx.shape[-1], dtype=torch.float, device=device)
			H = WavefrontAberratorGenerator._generateGaussian(filterSigma, kx, ky, domain='frequency', device=self.device)
			h = ift2(H, norm='backward').real
			ht = applyFilterSpaceDomain(h, ht, use_inplace_ffts=True).real
			return ht
		
		pad_x1 = int(np.floor(self.resolution[0] / 2))
		pad_x2 = int(np.ceil(self.resolution[0] / 2))
		pad_y1 = int(np.floor(self.resolution[1] / 2))
		pad_y2 = int(np.ceil(self.resolution[1] / 2))
		padding = (pad_x1, pad_x2, pad_y1, pad_y2)
		H = self.resolution[0] + pad_x1 + pad_x2
		W = self.resolution[1] + pad_y1 + pad_y2

		# Making the grids twice as big to avoid edge effects when filtering in generateRandomHeightsHelper2(...).
		kxTemp, kyTemp = WavefrontAberratorGenerator._create_normalized_grid(H, W, self.device)
		kxTemp = 2*np.pi * kxTemp / self.elementSpacings[0]
		kyTemp = 2*np.pi * kyTemp / self.elementSpacings[1]

		# xGridNorm, yGridNorm = WavefrontAberratorGenerator._create_normalized_grid(self.resolution[0], self.resolution[1], self.device)
		# xGrid = xGridNorm * self.resolution[0] * self.elementSpacings[0]
		# yGrid = yGridNorm * self.resolution[1] * self.elementSpacings[1]

		heights = 0
		for _ in range(2):
			# The 'filterSigma' argument is set to self.correlationLength/2.  This was obtained by considering that autocorrelation is
			# |H(e^{j\omega})|^2\Phi_{xx}(e^{j\omega}) in frequency, and the filter H is Gausssian in this case.
			#	-	Note that the Phi_{xx}(...) terms should be a constant since the autocorrelation of uncorrelated stationary noise should (I believe)
			#		be an impulse in time/space (and hence a constant in frequency)
			# 	-	By looking at the Fourier transform pair for a Gaussian, it can be seen that letting filterSigma=self.correlationLength/2 results
			# 		in the autocorrelation dropping to 1/e times its max value at a distance of self.correlationLength from the origin
			# 		(assuming that X is a random signal drawn from a zero-mean Gaussian distribution).
			#			- The 1/e comes from the definition of correlation length as defined in https://www.newfor.net/wp-content/uploads/2015/02/DL15-NEWFOR_Roughness_state_of_the_art.pdf
			#	-	Note that the autocorrelation of the unfiltered phases will essentially be a delta function (i.e. a constant in frequency) so
			#		the autocorrelation's shape will more-or-less be determined by H(e^{j\omega}).
			#	-	Note that autocorrelation in this context refers to autocorrelation with means removed.
			heightsTemp, heightAutocorrTemp = generateRandomHeightsHelper1(self.correlationLength / 2, kxTemp, kyTemp, padding, self.surfaceVariationVariance, self.device)

			# Can test to make sure that the correlation length is okay using this code:
				# N = 35
				# heightsTemp, heightAutocorrTemp = generateRandomHeightsHelper1(N*self.elementSpacings[0] / 2, kxTemp, kyTemp, padding, self.surfaceVariationVariance, self.device)
				# plt.imshow((heightAutocorrTemp * (heightAutocorrTemp >= heightAutocorrTemp.max() / np.exp(1))).cpu())
			# Should see (max autocorrelation)*1/e reached at around N units from the center on the plot
			# Behavior less reliable for larger N, however

			heightVariationCutoff = self.thicknessVariationMaxRange / 2
			heightsTemp = torch.clamp(heightsTemp, -heightVariationCutoff, heightVariationCutoff)

			heights = heights + heightsTemp

			if not self.doubleSidedRoughness:
				break
		
		heights = heights + (self.maxThickness - heights.max())
		heights = torch.max(heights, torch.tensor(self.minThickness))
		
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

		super().__init__(
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

		self._initializeModel()


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
from numbers import Number
from typing import Union
import warnings

import torch
import matplotlib.pyplot as plt

from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.Optical_Components.CGH_Component import CGH_Component
from holotorch.utils.Dimensions import HW, TensorDimension
from holotorch.utils.Enumerators import *
from holotorch.utils.Helper_Functions import generateGrid
from holotorch.Optical_Components.SimpleMask import SimpleMask


class Scatterer:
	def __init__(self, location_x, location_y, diameter, scatteringResponse):
		self.location_x = location_x
		self.location_y = location_y
		self.diameter = diameter
		self.scatteringResponse = scatteringResponse


class ScattererModel(CGH_Component):
	def __init__(	self,
					scatterers : Union[list, tuple, Scatterer] = None
				) -> None:

		super().__init__()
		self.initialize_scatterer_list(scatterers)
		self._gridMaskInitFlag = False
		

	def initialize_scatterer_list(self, scatterers):
		self.scatterers = []

		if scatterers is None:
			return

		if (not isinstance(scatterers, list)) and (not isinstance(scatterers, tuple)) and (not isinstance(scatterers, Scatterer)):
			raise Exception("Invalid argument for 'scatterers'.  Must be either a Scatterer instance or a list/tuple of Scatterer instances.")

		if (isinstance(scatterers, Scatterer)):
			self.scatterers.append(scatterers)
			return

		for s in scatterers:
			if (not isinstance(s, Scatterer)):
				raise Exception("Invalid argument for 'scatterers'.  Must be either a Scatterer instance or a list/tuple of Scatterer instances.")
			self.scatterers.append(s)


	def updateGridsAndMask(self, field):
		if (field.spacing.data_tensor.numel() != 2):
			raise Exception("'ScattererModel' class does not support multiple spacings.")

		resolution = tuple(field.data.shape[-2:])
		grid_spacing = tuple(field.spacing.data_tensor.squeeze().tolist())		# Technically, it's not necessary to convert to a tuple here---a list can work too.

		if self._gridMaskInitFlag:
			if ((resolution == self.resolution) and (grid_spacing == self.grid_spacing)):
				# Same resolution and spacing as before so no need to initialize new grids and a new mask
				return
		else:
			self._gridMaskInitFlag = True

		self.resolution = resolution
		self.grid_spacing = grid_spacing

		[self.xGrid, self.yGrid] = generateGrid(self.resolution, self.grid_spacing[0], self.grid_spacing[1])
		self.scatterer_mask = SimpleMask(
											tensor_dimension=HW(self.resolution[0], self.resolution[1]),
											init_type=INIT_TYPE.ZEROS,
											mask_model_type=MASK_MODEL_TYPE.COMPLEX,
											mask_forward_type=MASK_FORWARD_TYPE.MULTIPLICATIVE,
											mask_opt=False
										)
		self.scatterer_mask.mask = self.scatterer_mask.mask.to(field.data.device)

		for s in self.scatterers:
			x = s.location_x
			y = s.location_y
			d = s.diameter
			a = s.scatteringResponse

			tempInds = torch.sqrt(((self.xGrid - x)**2) + ((self.yGrid - y)**2)) <= (d/2)
			self.scatterer_mask.mask[tempInds] = a

			if (tempInds*1.0).sum() == 0:
				warnings.warn("A scatterer's diameter was small enough that no points on the mask got set.")


	def forward(self, field : ElectricField) -> ElectricField:
		self.updateGridsAndMask(field)
		return self.scatterer_mask(field)
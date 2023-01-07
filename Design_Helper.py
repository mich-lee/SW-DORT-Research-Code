import numpy as np
import torch
import matplotlib.pyplot as plt

import sys

sys.path.append("holotorch-lib/")
sys.path.append("holotorch-lib/holotorch")

from typing import NamedTuple
from collections import namedtuple

from holotorch.utils.units import * # E.g. to get nm, um, mm etc.
from holotorch.utils.Helper_Functions import generateGrid_MultiRes
from holotorch.Spectra.SpacingContainer import SpacingContainer

################################################################################################################################

class BasicSetup(NamedTuple):
	simulation_resolution			: tuple[int]
	simulation_sample_spacing		: float
	input_resolution				: tuple[int]
	input_sample_spacing			: float

################################################################################################################################

def get_pixel_grids(	
						res						: tuple[int],
						delta					: float,
						centerGrids				: bool = True,
						centerCoordsAroundZero	: bool = True,
						device					: torch.device = None
					):
	xGrid, yGrid = generateGrid_MultiRes(	res=res, spacingTensor=SpacingContainer(delta).data_tensor,
											centerGrids=centerGrids, centerCoordsAroundZero=centerCoordsAroundZero, device=device	)
	return xGrid, yGrid

################################################################################################################################

# Assumes square input pixels
# 	Coordinates might not be *exactly* in-line with the simulations---should be close though.
def FT_Lens_Setup_Helper(
							setup					: BasicSetup,
							wavelength				: float,
							focal_length			: float,
							device					: torch.device = None
						):
	res = setup.simulation_resolution
	inputRes = setup.input_resolution
	dx = setup.simulation_sample_spacing
	dxInput = setup.input_sample_spacing
	wavelen = wavelength
	f = focal_length

	xGrid, yGrid = get_pixel_grids(res=res, delta=dx, centerGrids=True, centerCoordsAroundZero=True, device=device)
	xGridInput, yGridInput = get_pixel_grids(res=inputRes, delta=dxInput, centerGrids=True, centerCoordsAroundZero=False, device=device)
	xGrid = xGrid.squeeze()
	yGrid = yGrid.squeeze()
	xGridInput = xGridInput.squeeze()
	yGridInput = yGridInput.squeeze()

	simulation_bw = 1 / dx
	sinc_envelope_first_null_freq = 1 / dxInput
	ft_scale_factor = wavelen * f

	ft_out_width = simulation_bw * ft_scale_factor
	ft_out_dimensions = (ft_out_width, ft_out_width)

	sinc_envelope_first_null = sinc_envelope_first_null_freq * ft_scale_factor

	approxHalfAngle = np.arctan(wavelen / dxInput)
	approxHalfWidth = f * np.tan(approxHalfAngle)
	minX = xGridInput[0,0] - approxHalfWidth
	minY = yGridInput[0,0] - approxHalfWidth
	maxX = xGridInput[-1,-1] + approxHalfWidth
	maxY = yGridInput[-1,-1] + approxHalfWidth
	intermediate_field_size_estimate = (float(maxX-minX), float(maxY-minY))

	retStruct = namedtuple("Return_Data", "ft_out_dimensions sinc_envelope_first_null intermediate_field_size_estimate")
	return retStruct(ft_out_dimensions, sinc_envelope_first_null, intermediate_field_size_estimate)

################################################################################################################################

setup = BasicSetup	(
						simulation_resolution = (4096,4096),
						simulation_sample_spacing = 3.2*um,
						input_resolution = (256, 256),
						input_sample_spacing = 3*(6.4*um)
					)

asdf = FT_Lens_Setup_Helper(setup, 532*nm, 100*mm)
print(asdf.ft_out_dimensions)
print(asdf.sinc_envelope_first_null)
print(asdf.intermediate_field_size_estimate)

pass
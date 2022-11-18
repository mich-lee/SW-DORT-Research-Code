import numpy as np
import torch
import matplotlib.pyplot as plt

import warnings
import sys
import copy
import datetime
import os

sys.path.append("holotorch-lib/")
sys.path.append("holotorch-lib/holotorch")

import holotorch.utils.Dimensions as Dimensions
from holotorch.utils.Enumerators import *
from holotorch.utils.units import * # E.g. to get nm, um, mm etc.
from holotorch.Optical_Components.CGH_Component import CGH_Component
# from holotorch.LightSources.CoherentSource import CoherentSource
from holotorch.Spectra.WavelengthContainer import WavelengthContainer
from holotorch.Spectra.SpacingContainer import SpacingContainer
from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.Optical_Propagators.ASM_Prop import ASM_Prop
# from holotorch.Optical_Propagators.ASM_Prop_Legacy import ASM_Prop_Legacy
# from holotorch.Sensors.Detector import Detector
from holotorch.Optical_Components.FT_Lens import FT_Lens
from holotorch.Optical_Components.Thin_Lens import Thin_Lens
# from holotorch.Optical_Components.SimpleMask import SimpleMask
from holotorch.Optical_Components.Field_Padder_Unpadder import Field_Padder_Unpadder
from holotorch.Optical_Components.Field_Resampler import Field_Resampler
from holotorch.Optical_Setups.Ideal_Imaging_Lens import Ideal_Imaging_Lens
from holotorch.Miscellaneous_Components.Memory_Reclaimer import Memory_Reclaimer

from ScattererModel import Scatterer, ScattererModel
from TransferMatrixProcessor import TransferMatrixProcessor
from WavefrontAberrator import RandomThicknessScreenGenerator, RandomThicknessScreen

from holotorch.utils.Helper_Functions import applyFilterSpaceDomain
from holotorch.utils.Field_Utils import get_field_slice
from MiscHelperFunctions import getSequentialModelComponentSequence, addSequentialModelOutputHooks, getSequentialModelOutputSequence, plotModelOutputSequence

import holotorch.utils.Memory_Utils as Memory_Utils


################################################################################################################################

def printSimulationSize(sz : tuple, spacing : float or tuple, prefixStr : str = ''):
	nx = sz[0]
	ny = sz[1]
	if (type(spacing) is tuple):
		dx = spacing[0]
		dy = spacing[1]
	else:
		dx = spacing
		dy = spacing
	lx_mm = nx * dx * 1e3
	ly_mm = ny * dy * 1e3
	dx_um = dx * 1e6
	dy_um = dy * 1e6
	print(prefixStr + "Simulation Size: %.3fx%.3f mm\t|\t(dx, dy): (%.3fum, %.3fum)" % (lx_mm, ly_mm, dx_um, dy_um))

class QuickFlip(CGH_Component):
	def __init__(self) -> None:
		super().__init__()
	
	def forward(self, field : ElectricField) -> ElectricField:
		newField = ElectricField(
			data        = torch.flip(field.data, [-2, -1]),
			spacing     = field.spacing,
			wavelengths = field.wavelengths            
		)
		return newField

################################################################################################################################


use_cuda = True
gpu_no = 0
device = torch.device("cuda:"+str(gpu_no) if use_cuda else "cpu")

Memory_Utils.initialize(RESERVED_MEM_CLEAR_CACHE_THRESHOLD_INIT=0.5, ALLOC_TO_RESERVED_RATIO_CLEAR_CACHE_THRESHOLD_INIT=0.8)


################################################################################################################################


syntheticWavelength = 0.1*mm
lambda1 = 550*nm
lambda2 = lambda1 * syntheticWavelength / (syntheticWavelength - lambda1)

wavelengths = [lambda1, lambda2]
# wavelengths = [lambda1]

inputRes = (384, 384)
inputSpacing = 6.4*um

intermediateRes = (4096, 4096)	# (int(8*inputRes[0]), int(8*inputRes[0]))
intermediateSpacing = inputSpacing / 2

outputRes = (3036, 3036)
outputSpacing = 1.85*um


################################################################################################################################


inputBoolMask = TransferMatrixProcessor.getUniformSampleBoolMask(inputRes[0], inputRes[1], 64, 64)
outputBoolMask = TransferMatrixProcessor.getUniformSampleBoolMask(outputRes[0], outputRes[1], 64, 64)

centerXInd = int(np.floor((inputRes[0] - 1) / 2))
centerYInd = int(np.floor((inputRes[1] - 1) / 2))

if np.isscalar(wavelengths):
	wavelengthContainer = WavelengthContainer(wavelengths=wavelengths, tensor_dimension=Dimensions.C(n_channel=1))
elif isinstance(wavelengths, list):
	wavelengthContainer = WavelengthContainer(wavelengths=wavelengths, tensor_dimension=Dimensions.C(n_channel=np.size(wavelengths)))
spacingContainer = SpacingContainer(spacing=inputSpacing)


fieldData = torch.zeros(4,1,1,wavelengthContainer.data_tensor.numel(),inputRes[0],inputRes[1],device=device)
# fieldData[...,centerXInd:centerXInd+1,centerYInd:centerYInd+1] = 1
fieldData[... , 0:12, 0:12] = 1
# fieldData[...,centerXInd-7:centerXInd+8,centerYInd-7:centerYInd+8] = 1
# fieldData[...,centerXInd-3:centerXInd+4,centerYInd-3:centerYInd+4] = 1
# fieldData[...,:,:] = 1
# fieldData[...,0,0] = 1
# fieldData[...,-1,0] = 1
# fieldData[...,0,-1] = 1
# fieldData[...,-1,-1] = 1
# fieldData[...,:,:] = torch.exp(1j*10*(2*np.pi/res[0])*torch.tensor(range(res[0])).repeat([res[1],1]))
# fieldData = torch.rand(fieldData.shape, device=device)
fieldData = fieldData + 0j

fieldIn = ElectricField(data=fieldData, wavelengths=wavelengthContainer, spacing=spacingContainer)
fieldIn.wavelengths.to(device=device)
fieldIn.spacing.to(device=device)

printSimulationSize(inputRes, inputSpacing, 'Simulation Input\t|\t')
printSimulationSize(intermediateRes, intermediateSpacing, 'Intermediate Calcs\t|\t')
printSimulationSize(outputRes, outputSpacing, 'Simulation Output\t|\t')
print()


################################################################################################################################


scattererList = [
					# Scatterer(location_x=0, location_y=0.065*mm, diameter=0.015*mm, scatteringResponse=1),
					# Scatterer(location_x=0.2*mm, location_y=0.2*mm, diameter=0.015*mm, scatteringResponse=1),
					# Scatterer(location_x=0, location_y=0.07*mm, diameter=0.015*mm, scatteringResponse=1),
					# Scatterer(location_x=0*mm, location_y=0*mm, diameter=0.3*mm, scatteringResponse=1),
					# Scatterer(location_x=0.75*mm, location_y=-1*mm, diameter=0.1*mm, scatteringResponse=1),
					# Scatterer(location_x=-1*mm, location_y=-1*mm, diameter=0.1*mm, scatteringResponse=1),
					# Scatterer(location_x=1.2*mm, location_y=1.2*mm, diameter=0.1*mm, scatteringResponse=1),
					# Scatterer(location_x=2.45*mm, location_y=2.45*mm, diameter=0.1*mm, scatteringResponse=1),

					# Scatterer(location_x=1.44*mm, location_y=1.44*mm, diameter=0.1*mm, scatteringResponse=1),
					
					Scatterer(location_x=-0.75*1.44*mm, location_y=-0.75*1.44*mm, diameter=0.08*mm, scatteringResponse=0.7),
					Scatterer(location_x=0.75*1.44*mm, location_y=0.75*1.44*mm, diameter=0.1*mm, scatteringResponse=0.8),
				]

# wavefrontAberratorGen = RandomThicknessScreenGenerator(	surfaceVariationStdDev = 1.3*um,
# 														correlationLength = 8.8*um,
# 														maxThickness = 200*um,
# 														n_screen = 1.52,
# 														generateBidirectional = True,
# 														resolution = intermediateRes,
# 														elementSpacings = [intermediateSpacing, intermediateSpacing],
# 														device = device
# 													)
# wavefrontAberrator = wavefrontAberratorGen.get_model()
# wavefrontAberratorReverse = wavefrontAberratorGen.get_model_reversed()

do_ffts_inplace = True

inputResampler = Field_Resampler(outputHeight=intermediateRes[0], outputWidth=intermediateRes[1], outputPixel_dx=intermediateSpacing, outputPixel_dy=intermediateSpacing, device=device)
# asmProp1 = ASM_Prop(init_distance=275/3*mm, do_ffts_inplace=do_ffts_inplace)
# asmProp2 = ASM_Prop(init_distance=110*mm, do_ffts_inplace=do_ffts_inplace)
# # asmProp2 = ASM_Prop(init_distance=(110-20)*mm, do_ffts_inplace=do_ffts_inplace)
# # asmProp3 = ASM_Prop(init_distance=20*mm, do_ffts_inplace=do_ffts_inplace)
# thinLens = Thin_Lens(focal_length=50*mm)
scattererModel = ScattererModel(scattererList)
memoryReclaimer = Memory_Reclaimer(device=device, clear_cuda_cache=True, collect_garbage=True,
										print_cleaning_actions=False, print_memory_status=False, print_memory_status_printType=2)
outputResampler = Field_Resampler(outputHeight=outputRes[0], outputWidth=outputRes[1], outputPixel_dx=outputSpacing, outputPixel_dy=outputSpacing, device=device)

model = torch.nn.Sequential	(
								inputResampler,
								FT_Lens(focal_length=50*mm),
								scattererModel,
								# FT_Lens(focal_length=50*mm),
								outputResampler
							)

# model = torch.nn.Sequential	(
# 								inputResampler,
# 								Ideal_Imaging_Lens(focal_length=50*mm, object_dist=275/3*mm, device=device),
# 								scattererModel,
# 								Ideal_Imaging_Lens(focal_length=50*mm, object_dist=110*mm, device=device),
# 								outputResampler
# 							)

# model = torch.nn.Sequential	(
# 								inputResampler,
# 								# memoryReclaimer,
# 								asmProp1,
# 								thinLens,
# 								asmProp2,
# 								# wavefrontAberrator,
# 								# asmProp3,
# 								scattererModel,
# 								# asmProp3,
# 								# wavefrontAberratorReverse,
# 								asmProp2,
# 								thinLens,
# 								asmProp1,
# 								# memoryReclaimer,
# 								outputResampler,
# 								# memoryReclaimer
# 							)


# asmProp1 = ASM_Prop(init_distance=33.333333333333*mm, do_ffts_inplace=do_ffts_inplace)
# asmProp2 = ASM_Prop(init_distance=50*mm, do_ffts_inplace=do_ffts_inplace)
# asmProp3 = ASM_Prop(init_distance=75*mm, do_ffts_inplace=do_ffts_inplace)
# # asmProp3 = ASM_Prop(init_distance=((75-0.2)/2)*mm, do_ffts_inplace=do_ffts_inplace)
# thinLens = Thin_Lens(focal_length=50*mm)
# model = torch.nn.Sequential	(
# 								inputResampler,
# 								memoryReclaimer,
# 								asmProp1,
# 								thinLens,
# 								asmProp2,
# 								thinLens,
# 								# asmProp3,
# 								memoryReclaimer,
# 								# wavefrontAberrator,
# 								asmProp3,
# 								scattererModel,
# 								asmProp3,
# 								# wavefrontAberratorReverse,
# 								memoryReclaimer,
# 								# asmProp3,
# 								thinLens,
# 								asmProp2,
# 								thinLens,
# 								asmProp1,
# 								memoryReclaimer,
# 								outputResampler
# 							)


# asmProp1 = ASM_Prop(init_distance=12.5*mm)
# asmProp2 = ASM_Prop(init_distance=25*mm)
# asmProp3 = ASM_Prop(init_distance=50*mm)
# thinLens = Thin_Lens(focal_length=25*mm)
# model = torch.nn.Sequential	(
# 								inputResampler,
# 								memoryReclaimer,
# 								asmProp1,
# 								thinLens,
# 								asmProp2,
# 								thinLens,
# 								asmProp3,
# 								scattererModel,
# 								asmProp3,
# 								thinLens,
# 								asmProp2,
# 								thinLens,
# 								asmProp1,
# 								outputResampler
# 							)


################################################################################################################################

modelComponentSequence = getSequentialModelComponentSequence(model=model, recursive=False)
addSequentialModelOutputHooks(model=model, recursive=False)
fieldOut = model(fieldIn)
outputs = getSequentialModelOutputSequence(model=model, recursive=False)

# plotModelOutputSequence(outputs=outputs, inputField=fieldIn, channel_inds_range=0, rescale_factor=1, plot_xlims=(-0.075,0.075), plot_ylims=(-0.075,0.075))
plotModelOutputSequence(outputs=outputs, inputField=fieldIn, componentSequenceList=modelComponentSequence, channel_inds_range=0, rescale_factor=0.25)

pass
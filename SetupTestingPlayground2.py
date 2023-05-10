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
from holotorch.Optical_Components.Four_F_system import Four_F_system
from holotorch.Optical_Components.Thin_Lens import Thin_Lens
# from holotorch.Optical_Components.SimpleMask import SimpleMask
from holotorch.Optical_Components.Radial_Optical_Aperture import Radial_Optical_Aperture
from holotorch.Optical_Components.Field_Padder_Unpadder import Field_Padder_Unpadder
from holotorch.Optical_Components.Field_Resampler import Field_Resampler
from holotorch.Optical_Setups.Ideal_Imaging_Lens import Ideal_Imaging_Lens
from holotorch.Miscellaneous_Components.Memory_Reclaimer import Memory_Reclaimer

from ScattererModel import Scatterer, ScattererModel, ScattererDrawing
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
	print(prefixStr + "Simulation Size: %.2fx%.2f mm\t|\t(dx, dy): (%.2fum, %.2fum)" % (lx_mm, ly_mm, dx_um, dy_um))

################################################################################################################################


use_cuda = True
gpu_no = 0
device = torch.device("cuda:"+str(gpu_no) if use_cuda else "cpu")

Memory_Utils.initialize(RESERVED_MEM_CLEAR_CACHE_THRESHOLD_INIT=0.5, ALLOC_TO_RESERVED_RATIO_CLEAR_CACHE_THRESHOLD_INIT=0.8)


################################################################################################################################


syntheticWavelength = 0.025*mm #0.05*mm
lambda1 = 1400*nm
lambda2 = lambda1 * syntheticWavelength / (syntheticWavelength - lambda1)

wavelengths = [lambda1, lambda2]
# wavelengths = [lambda1]

inputRes = (512, 512)
inputSpacing = 6.4*um

intermediateRes = (4096, 4096)	# (int(8*inputRes[0]), int(8*inputRes[0]))
intermediateSpacing = 3.2*um # inputSpacing / 2

outputRes = (64, 64)
outputSpacing = 1.85*um


################################################################################################################################


inputBoolMask = TransferMatrixProcessor.getUniformSampleBoolMask(inputRes[0], inputRes[1], 64, 64)
outputBoolMask = TransferMatrixProcessor.getUniformSampleBoolMask(outputRes[0], outputRes[1], 64, 64)

pixelResolution, pixelSize = TransferMatrixProcessor._calculateMacropixelParameters(inputBoolMask)
dx_pixel = pixelSize[0] * inputSpacing
dy_pixel = pixelSize[1] * inputSpacing

centerXInd = int(np.floor((inputRes[0] - 1) / 2))
centerYInd = int(np.floor((inputRes[1] - 1) / 2))

if np.isscalar(wavelengths):
	wavelengthContainer = WavelengthContainer(wavelengths=wavelengths, tensor_dimension=Dimensions.C(n_channel=1))
elif isinstance(wavelengths, list):
	wavelengthContainer = WavelengthContainer(wavelengths=wavelengths, tensor_dimension=Dimensions.C(n_channel=np.size(wavelengths)))
spacingContainer = SpacingContainer(spacing=inputSpacing)


fieldData = torch.zeros(1,1,1,wavelengthContainer.data_tensor.numel(),inputRes[0],inputRes[1],dtype=torch.complex64,device=device)
fieldIn = ElectricField(data=fieldData, wavelengths=wavelengthContainer, spacing=spacingContainer)
fieldIn.wavelengths.to(device=device)
fieldIn.spacing.to(device=device)

macropixelRes, macropixelSize = TransferMatrixProcessor._calculateMacropixelParameters(inputBoolMask)
vecIn = torch.zeros(macropixelRes, dtype=torch.complex64, device=device)

[vecInGridX, vecInGridY] = torch.meshgrid(torch.arange(vecIn.shape[0]), torch.arange(vecIn.shape[1]))
vecInGridX = (vecInGridX / vecIn.shape[0]).to(device=device)
vecInGridY = (vecInGridY / vecIn.shape[1]).to(device=device)

vecIn[...] = 0
# vecIn[...] = 1
# vecIn[2, 2] = 1
# vecIn[-2, -2] = 1
# vecIn[...] = torch.exp(1j*2*np.pi * (31*vecInGridX + 0*vecInGridY))
# vecIn[...] = vecIn[...] + torch.exp(1j*2*np.pi * (0*vecInGridX + 24*vecInGridY))
vecIn[...] = vecIn[...] + torch.exp(1j*2*np.pi * (0*vecInGridX + 17*vecInGridY))
# vecIn[...] = vecIn[...] + torch.exp(1j*2*np.pi * (0*vecInGridX + 8*vecInGridY))


vecIn = vecIn.view(1,1,1,1,macropixelRes[0]*macropixelRes[1])
fieldIn = TransferMatrixProcessor.getModelInputField(macropixelVector=vecIn, samplingBoolMask=inputBoolMask, fieldPrototype=fieldIn)

printSimulationSize(inputRes, inputSpacing, 'Simulation Input\t|\t')
printSimulationSize(intermediateRes, intermediateSpacing, 'Intermediate Calcs\t|\t')
printSimulationSize(outputRes, outputSpacing, 'Simulation Output\t|\t')
print()


################################################################################################################################

do_ffts_inplace = False

# scattererList = [
# 					Scatterer(location_x=-0.4*mm, location_y=-0.4*mm, diameter=0.02*mm, scatteringResponse=0.7),
# 					Scatterer(location_x=0.4*mm, location_y=0.4*mm, diameter=0.03*mm, scatteringResponse=0.8),

# 					# Scatterer(location_x=2*mm, location_y=0*mm, diameter=0.02*mm, scatteringResponse=0.7),
# 					# Scatterer(location_x=np.sqrt(2)*mm, location_y=np.sqrt(2)*mm, diameter=0.02*mm, scatteringResponse=0.7),
# 					# Scatterer(location_x=0*mm, location_y=2*mm, diameter=0.02*mm, scatteringResponse=0.7),
# 					# Scatterer(location_x=-np.sqrt(2)*mm, location_y=np.sqrt(2)*mm, diameter=0.02*mm, scatteringResponse=0.7),
# 					# Scatterer(location_x=-2*mm, location_y=0*mm, diameter=0.02*mm, scatteringResponse=0.7),
# 					# Scatterer(location_x=-np.sqrt(2)*mm, location_y=-np.sqrt(2)*mm, diameter=0.02*mm, scatteringResponse=0.7),
# 					# Scatterer(location_x=0*mm, location_y=-2*mm, diameter=0.02*mm, scatteringResponse=0.7),
# 					# Scatterer(location_x=np.sqrt(2)*mm, location_y=-np.sqrt(2)*mm, diameter=0.02*mm, scatteringResponse=0.7),
# 				]

scattererDrawing = ScattererDrawing()
	# scattererDrawing.drawLine(-0.3*mm, 0, 0.3*mm, 0, 200, 0.05*mm, 0.05*mm, 0.7, 0.8)
scattererDrawing.drawLine(-0.45*mm, -0.3*mm, 0.45*mm, -0.3*mm, 200, 0.05*mm, 0.05*mm, 0.79, 0.8)
scattererDrawing.drawLine(0.45*mm, -0.3*mm, -0.45*mm, 0.3*mm, 200, 0.05*mm, 0.05*mm, 0.79, 0.8)
scattererDrawing.drawLine(-0.45*mm, 0.3*mm, 0.45*mm, 0.3*mm, 200, 0.05*mm, 0.05*mm, 0.79, 0.8)
scattererList = scattererDrawing.getScattererList()

inputResampler = Field_Resampler(outputHeight=intermediateRes[0], outputWidth=intermediateRes[1], outputPixel_dx=intermediateSpacing, outputPixel_dy=intermediateSpacing, device=device)
scattererModel = ScattererModel(scattererList)
memoryReclaimer = Memory_Reclaimer(device=device, clear_cuda_cache=True, collect_garbage=True,
										print_cleaning_actions=False, print_memory_status=False, print_memory_status_printType=2)
outputResampler = Field_Resampler(outputHeight=outputRes[0], outputWidth=outputRes[1], outputPixel_dx=outputSpacing, outputPixel_dy=outputSpacing, device=device)

screenDist = 5*mm #0.5*mm
wavefrontAberratorGen = RandomThicknessScreenGenerator(	surfaceVariationStdDev = 1.3*um,
														correlationLength = 8.8*um,
														maxThickness = 200*um,
														n_screen = 1.52,
														generateBidirectional = True,
														resolution = intermediateRes,
														elementSpacings = [intermediateSpacing, intermediateSpacing],
														device = device
													)
wavefrontAberrator = wavefrontAberratorGen.get_model()
wavefrontAberratorReverse = wavefrontAberratorGen.get_model_reversed()




thinLens1 = Thin_Lens(focal_length=50*mm)
asmProp1 = ASM_Prop(init_distance=50*mm, do_ffts_inplace=do_ffts_inplace)
# asmProp2_no_aberrator = ASM_Prop(init_distance=33*mm, do_ffts_inplace=do_ffts_inplace)
# asmProp2 = ASM_Prop(init_distance=(33*mm - screenDist), do_ffts_inplace=do_ffts_inplace)#36*mm, do_ffts_inplace=do_ffts_inplace)
# asmProp3 = ASM_Prop(init_distance=(screenDist - wavefrontAberratorGen.maxThickness), do_ffts_inplace=do_ffts_inplace)
model = torch.nn.Sequential	(
								inputResampler,
								# Ideal_Imaging_Lens(focal_length=50*mm, object_dist=52.5*mm, interpolationMode='bicubic', rescaleCoords=False, device=device),
								asmProp1,
								Radial_Optical_Aperture(aperture_radius=5*mm),
								thinLens1,
								asmProp1,
								# asmProp2_no_aberrator,
								# asmProp2,
								# wavefrontAberrator,
								# asmProp3,
								scattererModel,
								# asmProp3,
								# wavefrontAberratorReverse,
								# asmProp2,
								# asmProp2_no_aberrator,
								asmProp1,
								thinLens1,
								Radial_Optical_Aperture(aperture_radius=5*mm),
								asmProp1,
								Ideal_Imaging_Lens(focal_length=10*mm, object_dist=250*mm, interpolationMode='bicubic', rescaleCoords=True, device=device),
								outputResampler
							)


################################################################################################################################

modelComponentSequence = getSequentialModelComponentSequence(model=model, recursive=True)
addSequentialModelOutputHooks(model=model, recursive=True)
fieldOut = model(fieldIn)
outputs = getSequentialModelOutputSequence(model=model, recursive=True)

# plotModelOutputSequence(outputs=outputs, inputField=fieldIn, channel_inds_range=0, rescale_factor=1, plot_xlims=(-0.075,0.075), plot_ylims=(-0.075,0.075))
plotModelOutputSequence(outputs=outputs, inputField=fieldIn, componentSequenceList=modelComponentSequence, channel_inds_range=0)
# plotModelOutputSequence(outputs=outputs, inputField=fieldIn, componentSequenceList=modelComponentSequence, channel_inds_range=0, rescale_factor=0.25)




o1 = outputs[3]
synthFieldData = torch.zeros(1,1,1,1,o1.data.shape[-2],o1.data.shape[-1], device=device) + 0j
synthFieldData[..., :, :] = o1.data[0,0,0,0,:,:] * o1.data[0,0,0,1,:,:].conj()
synthField = ElectricField(
							data = synthFieldData,
							wavelengths = float(fieldIn.wavelengths.data_tensor[0]*fieldIn.wavelengths.data_tensor[1]/(fieldIn.wavelengths.data_tensor[1]-fieldIn.wavelengths.data_tensor[0])),
							spacing = float(o1.spacing.data_tensor[...,0].squeeze())
						)
synthField.wavelengths.to(device)
synthField.spacing.to(device)

# fieldOutSynth = asmProp2_no_aberrator(synthField)
fieldOutSynth = asmProp1(synthField)

plt.figure(10)
plt.clf()
fieldOutSynth.visualize(flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE)

pass
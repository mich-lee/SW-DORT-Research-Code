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

from ScattererModel import Scatterer, ScattererModel
from TransferMatrixProcessor import TransferMatrixProcessor
from WavefrontAberrator import WavefrontAberrator

import holotorch.utils.Dimensions as Dimensions
from holotorch.utils.Enumerators import *
from holotorch.utils.units import * # E.g. to get nm, um, mm etc.
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
from holotorch.Miscellaneous_Components.Memory_Reclaimer import Memory_Reclaimer

from holotorch.utils.Helper_Functions import applyFilterSpaceDomain
from holotorch.utils.Field_Utils import get_field_slice
from MiscHelperFunctions import getSequentialModelComponentSequence, addSequentialModelOutputHooks, getSequentialModelOutputSequence, plotModelOutputSequence

import holotorch.utils.Memory_Utils as Memory_Utils


################################################################################################################################


use_cuda = True
gpu_no = 0
device = torch.device("cuda:"+str(gpu_no) if use_cuda else "cpu")

Memory_Utils.initialize(RESERVED_MEM_CLEAR_CACHE_THRESHOLD_INIT=0.5, ALLOC_TO_RESERVED_RATIO_CLEAR_CACHE_THRESHOLD_INIT=0.8)


################################################################################################################################


syntheticWavelength = 0.1*mm
lambda1 = 854*nm
lambda2 = lambda1 * syntheticWavelength / (syntheticWavelength - lambda1)

wavelengths = [lambda1, lambda2]
# wavelengths = [lambda1]

inputRes = (256, 256)
inputSpacing = 6.4*um

intermediateRes = (3036, 3036)	# (int(8*inputRes[0]), int(8*inputRes[0]))
intermediateSpacing = inputSpacing / 2

outputRes = (3036, 4024)
outputSpacing = 1.85*um


################################################################################################################################


centerXInd = int(np.floor((inputRes[0] - 1) / 2))
centerYInd = int(np.floor((inputRes[1] - 1) / 2))

if np.isscalar(wavelengths):
	wavelengthContainer = WavelengthContainer(wavelengths=wavelengths, tensor_dimension=Dimensions.C(n_channel=1))
elif isinstance(wavelengths, list):
	wavelengthContainer = WavelengthContainer(wavelengths=wavelengths, tensor_dimension=Dimensions.C(n_channel=np.size(wavelengths)))
spacingContainer = SpacingContainer(spacing=inputSpacing)


fieldData = torch.zeros(1,1,1,wavelengthContainer.data_tensor.numel(),inputRes[0],inputRes[1],device=device)
# fieldData[...,centerXInd:centerXInd+1,centerYInd:centerYInd+1] = 1
# fieldData[... , 0:4, 0:4] = 1
# fieldData[...,centerXInd-7:centerXInd+8,centerYInd-7:centerYInd+8] = 1
fieldData[...,centerXInd-3:centerXInd+4,centerYInd-3:centerYInd+4] = 1
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


################################################################################################################################


# scattererList = [
# 					# Scatterer(location_x=0, location_y=0.065*mm, diameter=0.015*mm, scatteringResponse=1),
# 					# Scatterer(location_x=0.2*mm, location_y=0.2*mm, diameter=0.015*mm, scatteringResponse=1),
# 					# Scatterer(location_x=0, location_y=0.07*mm, diameter=0.015*mm, scatteringResponse=1),
# 					# Scatterer(location_x=0*mm, location_y=0*mm, diameter=0.3*mm, scatteringResponse=1),
# 					# Scatterer(location_x=0.75*mm, location_y=-1*mm, diameter=0.1*mm, scatteringResponse=1),
# 					# Scatterer(location_x=-1*mm, location_y=-1*mm, diameter=0.1*mm, scatteringResponse=1),
# 					Scatterer(location_x=1.605*mm, location_y=1.605*mm, diameter=0.1*mm, scatteringResponse=1),
# 				]

inputResampler = Field_Resampler(outputHeight=intermediateRes[0], outputWidth=intermediateRes[1], outputPixel_dx=intermediateSpacing, outputPixel_dy=intermediateSpacing, device=device)
# asmProp1 = ASM_Prop(init_distance=12.5*mm)
# asmProp2 = ASM_Prop(init_distance=25*mm)
# asmProp3 = ASM_Prop(init_distance=50*mm)
# thinLens = Thin_Lens(focal_length=25*mm)
# scattererModel = ScattererModel(scattererList)
# memoryReclaimer = Memory_Reclaimer(device=device, clear_cuda_cache=True, collect_garbage=True)
# outputResampler = Field_Resampler(outputHeight=outputRes[0], outputWidth=outputRes[1], outputPixel_dx=outputSpacing, outputPixel_dy=outputSpacing, device=device)
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

inputBoolMask = TransferMatrixProcessor.getUniformSampleBoolMask(inputRes[0], inputRes[1], 64, 64)
outputBoolMask = TransferMatrixProcessor.getUniformSampleBoolMask(outputRes[0], outputRes[1], 60, 80)

################################################################################################################################

if False:
	transferMtxMeasurer = TransferMatrixProcessor(	inputFieldPrototype=fieldIn,
													inputBoolMask=inputBoolMask,
													outputBoolMask=outputBoolMask,
													model=model,
													numParallelColumns=4)
	H_mtx = transferMtxMeasurer.measureTransferMatrix()

	experimentSaveDict =	{
								'Transfer_Matrix'				:	H_mtx,
								'Model'							:	model,
								'Scatterer_List'				:	scattererList,
								'Field_Input_Prototype'			:	fieldIn,
								'Input_Bool_Mask'				:	inputBoolMask,
								'Output_Bool_Mask'				:	outputBoolMask,
								'Transfer_Matrix_Processor'		:	transferMtxMeasurer,
								'NOTE'							:	''
							}
	
	curDateTime = datetime.datetime.today()
	experimentSaveFileStr = 'Experiment_' + str(curDateTime.year) + '-' + str(curDateTime.month) + '-' + str(curDateTime.day) + '_' + \
							str(curDateTime.hour).zfill(2) + 'h' + str(curDateTime.minute).zfill(2) + 'm' + str(curDateTime.second).zfill(2) + 's.pt'
	
	while True:
		resp = input("Save experiment data as " + experimentSaveFileStr + "? (y/n): ")
		if (resp == 'y'):
			print("Saved '" + experimentSaveFileStr + "' to current working directory.")
			torch.save(experimentSaveDict, experimentSaveFileStr)
			break
		elif (resp == 'n'):
			print("Exiting...")
			break
		else:
			print("Invalid input.")


################################################################################################################################


model = WavefrontAberrator(0.1*mm, 2*np.pi*(1/intermediateSpacing)*100, 4, intermediateRes, [intermediateSpacing, intermediateSpacing], device=device).model
model = torch.nn.Sequential(inputResampler, model, ASM_Prop(init_distance=25*mm))


################################################################################################################################


modelComponentSequence = getSequentialModelComponentSequence(model=model, recursive=True)
addSequentialModelOutputHooks(model=model, recursive=True)
fieldOut = model(fieldIn)
outputs = getSequentialModelOutputSequence(model=model, recursive=True)

plotModelOutputSequence(outputs=outputs, inputField=fieldIn, channel_inds_range=0, rescale_factor=1, plot_xlims=(-0.075,0.075), plot_ylims=(-0.075,0.075))
# plotModelOutputSequence(outputs=outputs, inputField=fieldIn, channel_inds_range=0)

pass
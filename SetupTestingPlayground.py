import numpy as np
import torch
import matplotlib.pyplot as plt

import warnings
import sys
import copy
import datetime
# import pathlib

from torch.profiler import profile, record_function, ProfilerActivity

sys.path.append("holotorch-lib/")
sys.path.append("holotorch-lib/holotorch")

from ScattererModel import Scatterer, ScattererModel
from TransferMatrixProcessor import TransferMatrixProcessor

import holotorch.utils.Dimensions as Dimensions
from holotorch.utils.Enumerators import *
from holotorch.utils.units import * # E.g. to get nm, um, mm etc.
# from holotorch.LightSources.CoherentSource import CoherentSource
from holotorch.Spectra.WavelengthContainer import WavelengthContainer
from holotorch.Spectra.SpacingContainer import SpacingContainer
from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.Optical_Propagators.ASM_Prop import ASM_Prop
from holotorch.Optical_Propagators.ASM_Prop_Legacy import ASM_Prop_Legacy
# from holotorch.Sensors.Detector import Detector
from holotorch.Optical_Components.FT_Lens import FT_Lens
from holotorch.Optical_Components.Thin_Lens import Thin_Lens
from holotorch.Optical_Components.SimpleMask import SimpleMask
from holotorch.Optical_Components.Field_Padder_Unpadder import Field_Padder_Unpadder
from holotorch.Optical_Components.Field_Resampler import Field_Resampler

from holotorch.utils.Field_Utils import get_field_slice, applyFilterSpaceDomain
from MiscHelperFunctions import addSequentialModelOutputHooks, getSequentialModelOutputSequence


################################################################################################################################


use_cuda = True
gpu_no = 0
device = torch.device("cuda:"+str(gpu_no) if use_cuda else "cpu")


################################################################################################################################


syntheticWavelength = 0.1*mm
lambda1 = 854*nm
lambda2 = lambda1 * syntheticWavelength / (syntheticWavelength - lambda1)

wavelengths = [lambda1, lambda2]
# wavelengths = [lambda1]
inputSpacing = 6.4*um
inputRes = (512, 512)
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
fieldData[... , 0:16, 0:16] = 1
# fieldData[...,centerXInd-7:centerXInd+8,centerYInd-7:centerYInd+8] = 1
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


scattererList = [
					# Scatterer(location_x=0, location_y=0.065*mm, diameter=0.015*mm, scatteringResponse=1),
					# Scatterer(location_x=0.2*mm, location_y=0.2*mm, diameter=0.015*mm, scatteringResponse=1),
					# Scatterer(location_x=0, location_y=0.07*mm, diameter=0.015*mm, scatteringResponse=1),
					# Scatterer(location_x=0*mm, location_y=0*mm, diameter=0.3*mm, scatteringResponse=1),
					Scatterer(location_x=0.25*mm, location_y=0.25*mm, diameter=0.1*mm, scatteringResponse=1),
					Scatterer(location_x=-0.25*mm, location_y=-0.25*mm, diameter=0.1*mm, scatteringResponse=1),
				]

inputResampler = Field_Resampler(outputHeight=int(4*inputRes[0]), outputWidth=int(4*inputRes[1]), outputPixel_dx=inputSpacing/2, outputPixel_dy=inputSpacing/2, device=device)
asmProp = ASM_Prop(init_distance=50*mm)
thinLens = Thin_Lens(focal_length=25*mm)
scattererModel = ScattererModel(scattererList)
outputResampler = Field_Resampler(outputHeight=outputRes[0], outputWidth=outputRes[1], outputPixel_dx=outputSpacing, outputPixel_dy=outputSpacing, device=device)
model = torch.nn.Sequential	(
								inputResampler,
								asmProp,
								thinLens,
								asmProp,
								scattererModel,
								asmProp,
								thinLens,
								asmProp,
								outputResampler
							)


################################################################################################################################

inputBoolMask = TransferMatrixProcessor.getUniformSampleBoolMask(inputRes[0], inputRes[1], 64, 64)
outputBoolMask = TransferMatrixProcessor.getUniformSampleBoolMask(outputRes[0], outputRes[1], 60, 80)

################################################################################################################################

if True:
	# model2 = testModel()	#torch.nn.Identity()
	transferMtxMeasurer = TransferMatrixProcessor(	inputFieldPrototype=fieldIn,
													inputBoolMask=inputBoolMask,
													outputBoolMask=outputBoolMask,
													model=model,
													numParallelColumns=8)
	H_mtx = transferMtxMeasurer.measureTransferMatrix()

	experimentSaveDict =	{
								'Transfer_Matrix'				:	H_mtx,
								'Model'							:	model,
								'Scatterer_List'				:	scattererList,
								'Field_Input_Prototype'			:	fieldIn,
								'Input_Bool_Mask'				:	inputBoolMask,
								'Output_Bool_Mask'				:	outputBoolMask,
								'Transfer_Matrix_Processor'		:	transferMtxMeasurer,
								'NOTE'							:	'I am not 100% sure if all the information/model/etc are correct here as the transfer matrix was saved about four days ago.'
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


addSequentialModelOutputHooks(model)
fieldOut = model(fieldIn)
outputs = getSequentialModelOutputSequence(model)



# tempInd = 3
# plt.clf()
# get_field_slice(outputs[tempInd], channel_inds_range=0).visualize(flag_axis=True,plot_type=ENUM_PLOT_TYPE.MAGNITUDE)

pass
import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
import os
import shutil
import copy
import datetime
# import pathlib
import gc
import warnings

sys.path.append("holotorch-lib/")
sys.path.append("holotorch-lib/holotorch")

from ScattererModel import Scatterer, ScattererModel
from TransferMatrixProcessor import TransferMatrixProcessor
from WavefrontAberrator import WavefrontAberrator
from holotorch.Optical_Components.Field_Resampler import Field_Resampler

import holotorch.utils.Dimensions as Dimensions
from holotorch.utils.Enumerators import *
from holotorch.utils.units import * # E.g. to get nm, um, mm etc.
# from holotorch.LightSources.CoherentSource import CoherentSource
from holotorch.Spectra.WavelengthContainer import WavelengthContainer
from holotorch.Spectra.SpacingContainer import SpacingContainer
from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.Optical_Propagators.Propagator import Propagator
from holotorch.Optical_Propagators.ASM_Prop import ASM_Prop
# from holotorch.Optical_Components.Resize_Field import Resize_Field
# from holotorch.Sensors.Detector import Detector
from holotorch.Optical_Components.FT_Lens import FT_Lens
from holotorch.Optical_Components.Thin_Lens import Thin_Lens
from holotorch.Optical_Components.SimpleMask import SimpleMask
from holotorch.Optical_Components.Field_Padder_Unpadder import Field_Padder_Unpadder

from holotorch.utils.Field_Utils import get_field_slice, applyFilterSpaceDomain

################################################################################################################################

def generateHadamardBasisVector(n : int, nRows : int, nCols : int, nStep : int):
	def getBinaryRepresentation(nums, nBits):
		bits = torch.div((nums[:,None] % (2 ** torch.tensor(range(nBits,0,-1))))[None,:], (2 ** torch.tensor(range(nBits-1,-1,-1))), rounding_mode='trunc')
		return bits.squeeze(0)

	def getPattern(b):
		inds = torch.ones(b.shape[-2], 2 ** b.shape[-1])
		for i in range(b.shape[-1]):
			inds[:,(2**i):(2**(i+1))] = inds[:,0:(2**i)] * (1 - 2*(b[:,i] != 0))[:,None]
		return inds
	
	errorFlag = False
	if (nRows != nCols):
		errorFlag = True

	M = np.log2(nRows)
	if (M % 1 != 0):
		errorFlag = True
	M = int(M)

	if (np.isscalar(n)):
		nVec = torch.tensor([n])
	else:
		nVec = torch.tensor(n)

	if (nVec >= (2 ** M)**2).sum() != 0:
		errorFlag = True
	
	if errorFlag:
		raise Exception("Invalid input arguments.")
	
	colPattern = getPattern(getBinaryRepresentation((nVec / nCols).floor(), M))[:,None,:]
	rowPattern = getPattern(getBinaryRepresentation(nVec % nRows, M))[:,:,None]

	return ((colPattern * rowPattern) / np.sqrt(nRows * nCols)).repeat_interleave(nStep, dim=-2).repeat_interleave(nStep, dim=-1) + 0j


# Feed input fields into the returned 'inputModel' to get the fields that one uses to get the synthetic wavelength field that is backpropagated.
def getInputAndBackpropagationModels(model : torch.nn.Sequential):
	def get_aberratorless_propagator(comps : torch.nn.Sequential):
		if len(comps) == 0:
			warnings.warn("Empty backpropagation model")
			return comps

		totalDist = 0
		asmStateDict : dict = None
		for i in range(len(comps)):
			cur = comps[i]
			if issubclass(type(cur), Propagator):
				totalDist = totalDist + cur.z
				if (type(cur) is ASM_Prop) and (asmStateDict is None):
					asmStateDict = cur.__dict__
			elif issubclass(type(cur), WavefrontAberrator):
				totalDist = totalDist + cur.get_thickness()
			else:
				raise Exception("Unknown component")

		prop = ASM_Prop(init_distance=totalDist)
		if asmStateDict is not None:
			# Loading settings from an existing ASM_Prop object (if there is one)
			prop.__setstate__(asmStateDict)	# Will override the `init_distance=totalDist` a few lines back
											#	`prop.z` will be set to whatever was saved (i.e. NOT totalDist)
			prop.z = totalDist	# Setting `prop.z` again after it was overwritten
			prop.prop_kernel = None	# Will force the propagation kernel to be rebuilt the next time a field is input (different `prop.z` means different propagation kernel needed)
									#	Probably not necessary as `prop_kernel` was probably saved as `None` anyways.

		return prop
		

	if type(model) is not torch.nn.Sequential:
		raise Exception("Model must be of type 'torch.nn.Sequential'.")
	
	preScattererModel = torch.nn.Sequential()
	scattererModel = None
	for i in range(len(model)):
		# if type(model[i]) is not ScattererModel:
		if not issubclass(type(model[i]), ScattererModel):
			preScattererModel.append(model[i])
		else:
			scattererModel = model[i]
			break
	if scattererModel is None:
		raise Exception("No scatterers found!")
	
	tempBackpropModelStartInd = len(preScattererModel)
	for i in range(len(preScattererModel) - 1, -1, -1):
		curType = type(preScattererModel[i])
		if issubclass(curType, Propagator) or issubclass(curType, WavefrontAberrator):
			tempBackpropModelStartInd = i
		else:
			break
	inputModel = preScattererModel[0:tempBackpropModelStartInd]
	tempBackpropModel = preScattererModel[tempBackpropModelStartInd:]

	# Making a model for backpropagation (of fields) that does not include wavefront abberators
	backpropModel = get_aberratorless_propagator(tempBackpropModel)

	return preScattererModel, scattererModel, inputModel, backpropModel

################################################################################################################################

# Random code for Hadamard basis vectors:
#	aaa = generateHadamardBasisVector(list(range(64*64)),64,64,1)
#	bbb = torch.matmul(aaa.reshape(4096,4096), aaa.reshape(4096,4096).transpose(-2,-1))

################################################################################################################################








################################################################################################################################

use_cuda = True
gpu_no = 0
device = torch.device("cuda:"+str(gpu_no) if use_cuda else "cpu")

################################################################################################################################

# dataFilePath = 'DATA/Experiment_202210201534.pt'
# dataFilePath = 'DATA/Experiment_202210192329.pt'
# dataFilePath = 'DATA/Experiment_2022-10-27_19h14m38s.pt'
# dataFilePath = 'DATA/Experiment_2022-11-6_22h18m58s.pt'
# dataFilePath = 'DATA/Experiment_2022-11-16_20h46m21s.pt'
# dataFilePath = 'DATA/Experiment_2022-11-22_13h38m51s.pt'
# dataFilePath = 'DATA/Experiment_2022-11-22_20h00m26s.pt'	# Good data?
# dataFilePath = 'DATA/Blah/Experiment_2023-1-3_17h03m58s.pt'
# dataFilePath = 'DATA/GOOD DATA/Single Scatterer Demo/Experiment_2023-1-7_21h35m04s.pt'
# dataFilePath = 'DATA/Experiment_2023-1-7_23h04m14s.pt'

# dataFilePath = 'DATA/GOOD DATA/Experiment_2023-1-8_04h34m16s.pt'
# dataFilePath = 'DATA/Experiment_2023-1-25_14h13m51s.pt'
# dataFilePath = 'DATA/Experiment_2023-1-25_16h01m48s.pt'
# dataFilePath = 'DATA/Experiment_2023-1-25_17h25m58s.pt'
# dataFilePath = 'DATA/Experiment_2023-1-26_15h42m57s.pt'

# dataFilePath = 'DATA/GOOD DATA/GREAT/Experiment_2023-1-26_20h01m29s.pt'

# dataFilePath = 'DATA/Experiment_2023-2-13_13h56m39s.pt'
# dataFilePath = 'DATA/Experiment_2023-3-20_16h33m08s.pt'		# "N" with aberrating layers
# dataFilePath = 'DATA/Experiment_2023-3-20_18h13m30s.pt'		# "N" without aberrating layers

# dataFilePath = 'DATA/Experiment_2023-4-2_14h21m09s.pt'		# Two pointlike scatterers, no aberrating layer
# dataFilePath = 'DATA/Experiment_2023-4-2_15h42m31s.pt'		# Two pointlike scatterers, aberrating layer
# dataFilePath = 'DATA/Experiment_2023-4-2_17h18m18s.pt'		# Two pointlike scatterers, aberrating layer

# dataFilePath = 'DATA/GOOD DATA/Experiment_2023-4-2_20h49m48s.pt'		# Two pointlike scatterers, aberrating layer

# dataFilePath = 'DATA/GOOD DATA/GREAT/Experiment_2023-4-8_18h10m34s.pt'		# Two pointlike scatterers, aberrating layer


# dataFilePath = 'DATA/Temp/Experiment_2023-4-21_14h52m56s.pt'		# Two pointlike scatterers, aberrating layer
# dataFilePath = 'DATA/Temp/Experiment_2023-4-21_16h46m19s.pt'		# Two pointlike scatterers, aberrating layer
# dataFilePath = 'DATA/Temp/Experiment_2023-5-8_16h28m22s.pt'		# Three pointlike scatterers, aberrating layer


# dataFilePath = 'DATA/Temp2/Experiment_2023-4-28_20h59m03s.pt'		# Line, no aberrating layer
# dataFilePath = 'DATA/Temp2/Experiment_2023-4-30_19h03m11s.pt'		# Line, aberrating layer

# dataFilePath = 'DATA/Temp3/Experiment_2023-5-9_12h50m22s.pt'		# Three pointlike scatterers, no aberrating layer
dataFilePath = 'DATA/Temp3/Experiment_2023-5-9_14h51m01s.pt'		# Three pointlike scatterers, aberrating layer

################################################################################################################################

doSyntheticWavelengths = True

nonSyntheticWavelengthSelectionIndex = 0	# Only relevant if doSyntheticWavelengths = False

# IMPORTANT NOTE:
#	See the source code notes above the eigenstructure demixing method ('demixEigenstructure' in the TransferMatrixProcessor class) for more information. 
doEigenstructureDemixing = True
singValMagnitudeSimilarityThreshold = 0.15

doPlotting = True
plotScatterers = True

################################################################################################################################




################################################################################################################################

dataFilePathHeadTail = os.path.split(dataFilePath)
dataFileBackupPath = dataFilePathHeadTail[0] + '/_' + dataFilePathHeadTail[1] + '.bak'
if not os.path.isfile(dataFileBackupPath):
	os.rename(dataFilePath, dataFileBackupPath)
	shutil.copy2(dataFileBackupPath, dataFilePath)

loadedData = torch.load(dataFilePath, map_location=device)

model = loadedData['Model']
inputBoolMask = loadedData['Input_Bool_Mask']
outputBoolMask = loadedData['Output_Bool_Mask']

H = loadedData['Transfer_Matrix']

if not 'U' in loadedData:
	print("Taking singular value decomposition of the transfer matrix...", end='')
	U, S, Vh = torch.linalg.svd(H)
	V = Vh.conj().transpose(-2, -1)
	print("Done!")
	print("Saving the computed singular value decomposition of the transfer matrix...", end='')
	loadedData['U'] = U
	loadedData['S'] = S
	loadedData['V'] = V
	torch.save(loadedData, dataFilePath)
	print("Done!")
else:
	print("Loading previously computed singular value decomposition of the transfer matrix...", end='')
	U = loadedData['U']
	S = loadedData['S']
	V = loadedData['V']
	Vh = V.conj().transpose(-2, -1)
	print("Done!")

################################################################################################################################

if doEigenstructureDemixing:
	print("Demixing eigenstructure...", end='')
	U, S, V = TransferMatrixProcessor.demixEigenstructure(U, S, V, 'V', singValMagnitudeSimilarityThreshold)
	Vh = V.conj().transpose(-2, -1)
	print("Done!")
else:
	print("Skipped eigenstructure demixing step.")

q = torch.matmul(Vh[0,0,0,1,:,:], V[0,0,0,0,:,:])
# w = (S[0,0,0,1,:][:,None] + S[0,0,0,0,:][None,:]) / 2		# I don't know why I did it this way.
w = (S[0,0,0,1,:] + S[0,0,0,0,:]) / 2

# tempResampler = Field_Resampler(outputHeight=8192, outputWidth=8192, outputPixel_dx=6.4*um, outputPixel_dy=6.4*um, device=device)
preScattereModel, scattererModel, inputModel, backpropModel = getInputAndBackpropagationModels(model)



# Plot settings
nCols1 = 3
nRows1 = 2
xLims1 = [-1, 1]
yLims1 = [-1, 1]
coordsMultiplier1 = 1e3	# Scale for millimeters

# Deriving some quantities
nSubplots1 = nCols1 * nRows1
scattererLocsX = []
scattererLocsY = []
for i in range(len(scattererModel.scatterers)):
	sTemp = scattererModel.scatterers[i]
	scattererLocsX = scattererLocsX + [sTemp.location_x * coordsMultiplier1]
	scattererLocsY = scattererLocsY + [sTemp.location_y * coordsMultiplier1]

# fields = []
# synthFields = []
imgField0 = None
temp0 = None


if doSyntheticWavelengths:
	print("NOTE: Using synthetic wavelengths.")
else:
	print("NOTE: Not using synthetic wavelengths.")


for singVecNum in range(500):
	print("Processing singular vector #" + str(singVecNum) + "...", end='')

	vecIn = V[... , :, singVecNum]
	fieldIn = TransferMatrixProcessor.getModelInputField(macropixelVector=vecIn, samplingBoolMask=inputBoolMask, fieldPrototype=loadedData['Field_Input_Prototype'])

	o1 = inputModel(fieldIn)
	fieldOut_o1 = backpropModel(o1)
	
	# Resample to force spacing to be the same for all dimensions (B, T, P, and C)
	# o1 = model[0](o1)

	if doSyntheticWavelengths:
		synthFieldData = torch.zeros(1,1,1,1,o1.data.shape[-2],o1.data.shape[-1], device=device) + 0j
		synthFieldData[..., :, :] = o1.data[0,0,0,0,:,:] * o1.data[0,0,0,1,:,:].conj()
		synthField = ElectricField(
									data = synthFieldData,
									wavelengths = float(fieldIn.wavelengths.data_tensor[0]*fieldIn.wavelengths.data_tensor[1]/(fieldIn.wavelengths.data_tensor[1]-fieldIn.wavelengths.data_tensor[0])),
									spacing = float(o1.spacing.data_tensor[...,0].squeeze())
								)
		synthField.wavelengths.to(device)
		synthField.spacing.to(device)

		fieldOut = backpropModel(synthField)
	else:
		# fieldOut = fieldOut_o1[:,:,:,0,:,:]
		fieldOut = get_field_slice(fieldOut_o1, channel_inds_range=nonSyntheticWavelengthSelectionIndex)


	if imgField0 is None:
		imgField0 = torch.zeros(fieldOut.data.shape, device=fieldOut.data.device)
		imgField1 = torch.zeros(fieldOut.data.shape, device=fieldOut.data.device)
		imgField2 = torch.zeros(fieldOut.data.shape, device=fieldOut.data.device)
		imgField3 = torch.zeros(fieldOut.data.shape, device=fieldOut.data.device)
		imgField4 = torch.zeros(fieldOut.data.shape, device=fieldOut.data.device)
		imgField5 = torch.zeros(fieldOut.data.shape, device=fieldOut.data.device)
		temp0 = torch.zeros(fieldOut.data.shape, device=fieldOut.data.device)
	temp0[...] = fieldOut.data
	temp0Max = temp0.abs().max()
	
	# temp0 = temp0 / temp0Max
	# temp0[temp0.abs() < 0.9 * temp0Max] = 0
	temp0[temp0.abs() < 0.99 * temp0Max] = 0

	imgField0 = torch.maximum(imgField0, w[singVecNum] * temp0.abs())
	imgField1 = imgField1 + (w[singVecNum] * temp0.abs())
	imgField2 = torch.maximum(imgField2, temp0.abs())
	imgField3 = imgField3 + temp0.abs()

	temp0[...] = fieldOut.data
	imgField4 = imgField4 + (w[singVecNum] * temp0)
	imgField5 = imgField5 + temp0.abs()


	gc.collect()
	torch.cuda.empty_cache()
	gc.collect()
	torch.cuda.empty_cache()


	# tempField = fieldOut_o1.detach().cpu()
	# tempSynthField = fieldOut.detach().cpu()
	# fields.append(tempField)
	# synthFields.append(tempSynthField)

	print("Done!")

	if not doPlotting:
		continue


	# For debugging
	# plt.clf()
	# a = get_field_slice(fieldOut_o1,channel_inds_range=0)
	# a.visualize(flag_axis=True,cmap='turbo',plot_type=ENUM_PLOT_TYPE.MAGNITUDE)

	subplotNum = (singVecNum % nSubplots1) + 1
	plt.figure(int(np.floor(singVecNum / nSubplots1)) + 1)
	if subplotNum == 1:
		plt.clf()

	plt.subplot(nRows1, nCols1, subplotNum)
	fieldOut.visualize(flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE, cmap='turbo')
	if plotScatterers:
		plt.scatter(scattererLocsY, scattererLocsX, s=96, marker='o', alpha=0.5, color='red', edgecolor='none', label='Scatterer')		# X and Y are switched because HoloTorch has the horizontal and vertical dimensions switched (relative to what the plots consider as horizontal and vertical)
	plt.xlim(xLims1)
	plt.ylim(yLims1)
	plt.title("$\Lambda$ - Singular Value #" + str(singVecNum + 1))
	plt.legend()

	# plt.subplot(nRows1*2, nCols1, subplotNum)
	# fieldOut.visualize(flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE, cmap='turbo')
	# plt.scatter(scattererLocsY, scattererLocsX, s=96, marker='+', color='red', edgecolor='none', label='Scatterer')		# X and Y are switched because HoloTorch has the horizontal and vertical dimensions switched (relative to what the plots consider as horizontal and vertical)
	# plt.xlim(xLims1)
	# plt.ylim(yLims1)
	# plt.title("$\Lambda$ - Singular Value #" + str(singVecNum + 1))
	# plt.legend()

	# plt.subplot(nRows1*2, nCols1, subplotNum + nCols1)
	# fieldOut.visualize(flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE, cmap='Paired')
	# plt.scatter(scattererLocsY, scattererLocsX, s=96, marker='+', color='black', edgecolor='none', label='Scatterer')		# X and Y are switched because HoloTorch has the horizontal and vertical dimensions switched (relative to what the plots consider as horizontal and vertical)
	# plt.xlim(xLims1)
	# plt.ylim(yLims1)
	# plt.title("Singular Value #" + str(singVecNum + 1))
	# plt.legend()

	# plt.subplot(nRows1*2, nCols1, subplotNum + nCols1)
	# tempField = get_field_slice(fieldOut_o1, channel_inds_range=0)
	# tempField.visualize(flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE, cmap='turbo')
	# plt.scatter(scattererLocsY, scattererLocsX, s=96, marker='+', color='red', edgecolor='none', label='Scatterer')		# X and Y are switched because HoloTorch has the horizontal and vertical dimensions switched (relative to what the plots consider as horizontal and vertical)
	# plt.xlim(xLims1)
	# plt.ylim(yLims1)
	# plt.title("$\lambda$ - Singular Value #" + str(singVecNum + 1))
	# plt.legend()

	plt.show()

pass



# a1 = fieldOut.detach().cpu()
# a1.data = imgField
# plt.clf()
# a1.visualize(flag_axis=True,cmap='turbo')
# plt.xlim(-1, 1)
# plt.ylim(-1, 1)









# a1 = fieldOut.detach().cpu()
# a1.data = imgField4
# plt.clf()
# a1.visualize(flag_axis=True,cmap='turbo')
# #plt.scatter(scattererLocsY, scattererLocsX, s=96, marker='o', alpha=0.5, color='red', edgecolor='none', label='Scatterer')		# X and Y are switched because HoloTorch has the horizontal and vertical dimensions switched (relative to what the plots consider as horizontal and vertical)
# plt.xlim(-1, 1)
# plt.ylim(-1, 1)




# plt.clf()
# aaa = get_field_slice(model[0:7](fieldIn), channel_inds_range=0)
# aaa.visualize(flag_axis=True)
# plt.scatter(scattererLocsY, scattererLocsX, s=96, marker='o', alpha=0.5, color='red', edgecolor='none', label='Scatterer')







# Experimental backpropagation.
# Attempting to do the process described in Figure 1a of the paper "Fast non-line-of-sight imaging with high-resolution and wide field of view using synthetic wavelength holography".
# Run this in the debug console right after o1 and fieldOut_o1 are computed.

# This combination of backpropModel.z signs and conjugation of data seems to work the best for some reason.
#		Feel free to experiment though.

	# syntheticWavelength = 0.05*mm # 0.05*mm
	# lambda1 = 1400*nm
	# lambda2 = lambda1 * syntheticWavelength / (syntheticWavelength - lambda1)
	# fieldIn.data[...] = 1
	# fieldIn.wavelengths.data_tensor = torch.tensor([lambda1, lambda2], device=fieldIn.wavelengths.data_tensor.device)

	# inputModel = model[0:11]
	# backpropModel.z = -backpropModel.z.abs()
	# backpropModel.prop_kernel = None

	# o1 = inputModel(fieldIn)

	# o1.data = o1.data.conj()

	# fieldOut_o1 = backpropModel(o1)






# a = torch.zeros(synthFields[0].data.shape).cuda()
# for i in range(len(fields)):
# 	print(str(i))
# 	temp0 = synthFields[i].data.cuda()
# 	temp0Max = temp0.abs().max()
# 	temp0[temp0.abs() < 0.99 * temp0Max] = 0
# 	# a = torch.maximum(a, w[i].cpu() * temp0.abs())
# 	a = a + abs(w[i].cpu() * temp0.abs())
# 	synthFields[i].data.cpu()
# a1 = fieldOut.detach().cpu()
# a1.data = a



# fin0 = get_field_slice(o1,channel_inds_range=0)
# fin1 = get_field_slice(o1,channel_inds_range=1)
# fout0 = get_field_slice(fieldOut_o1,channel_inds_range=0)
# fout1 = get_field_slice(fieldOut_o1,channel_inds_range=1)

# torch.cuda.empty_cache()

# plt.figure(1)
# plt.clf()
# plt.subplot(2,3,1)
# fin0.visualize(flag_axis=True,cmap='turbo',plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
# plt.title('$u_{out}(x,y;\lambda = ' + str(round(float(fieldIn.wavelengths.data_tensor[0]*1e9),3)) + 'nm)$', fontsize=16)
# plt.xlabel('Position (mm)', fontsize=14)
# plt.ylabel('Position (mm)', fontsize=14)
# plt.subplot(2,3,2)
# fin1.visualize(flag_axis=True,cmap='turbo',plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
# plt.title('$u_{out}(x,y;\lambda = ' + str(round(float(fieldIn.wavelengths.data_tensor[1]*1e9),3)) + 'nm)$', fontsize=16)
# plt.xlabel('Position (mm)', fontsize=14)
# plt.ylabel('Position (mm)', fontsize=14)
# plt.subplot(2,3,3)
# synthField.visualize(flag_axis=True,cmap='turbo',plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
# plt.title('$u_{out,synth}(x,y;\Lambda = 0.05mm)$', fontsize=16)
# plt.xlabel('Position (mm)', fontsize=14)
# plt.ylabel('Position (mm)', fontsize=14)

# tempLim1 = 1
# tempColor1 = 'red'
# tempMarker1 = 'x'
# tempSize1 = 160
# fout0 = get_field_slice(fieldOut_o1,channel_inds_range=0)
# fout1 = get_field_slice(fieldOut_o1,channel_inds_range=1)
# plt.figure(1123)
# plt.clf()
# plt.subplot(2,3,1)
# fout0.visualize(flag_axis=True,cmap='turbo',plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
# plt.title('$u_{backprop}(x,y;\lambda = ' + str(round(float(fieldIn.wavelengths.data_tensor[0]*1e9),3)) + 'nm)$', fontsize=16)
# plt.scatter(scattererLocsY, scattererLocsX, s=tempSize1, marker=tempMarker1, color=tempColor1, edgecolor='none', label='Scatterer')
# plt.legend()
# plt.xlabel('Position (mm)', fontsize=14)
# plt.ylabel('Position (mm)', fontsize=14)
# plt.xlim(-tempLim1,tempLim1)
# plt.ylim(-tempLim1,tempLim1)
# plt.subplot(2,3,2)
# fout1.visualize(flag_axis=True,cmap='turbo',plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
# plt.title('$u_{backprop}(x,y;\lambda = ' + str(round(float(fieldIn.wavelengths.data_tensor[1]*1e9),3)) + 'nm)$', fontsize=16)
# plt.scatter(scattererLocsY, scattererLocsX, s=tempSize1, marker=tempMarker1, color=tempColor1, edgecolor='none', label='Scatterer')
# plt.legend()
# plt.xlabel('Position (mm)', fontsize=14)
# plt.ylabel('Position (mm)', fontsize=14)
# plt.xlim(-tempLim1,tempLim1)
# plt.ylim(-tempLim1,tempLim1)
# plt.subplot(2,3,3)
# fieldOut.visualize(flag_axis=True,cmap='turbo',plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
# plt.title('$u_{backprop,synth}(x,y;\Lambda = 0.05mm)$', fontsize=16)
# plt.scatter(scattererLocsY, scattererLocsX, s=tempSize1, marker=tempMarker1, color=tempColor1, edgecolor='none', label='Scatterer')
# plt.legend()
# plt.xlabel('Position (mm)', fontsize=14)
# plt.ylabel('Position (mm)', fontsize=14)
# plt.xlim(-tempLim1,tempLim1)
# plt.ylim(-tempLim1,tempLim1)

# tempLim2a = 0.2
# tempLim2b = 0.5
# tempColor2 = 'red'
# tempMarker2 = 'x'
# tempSize2 = 160
# fout0 = get_field_slice(fieldOut_o1,channel_inds_range=0)
# fout1 = get_field_slice(fieldOut_o1,channel_inds_range=1)
# plt.figure(1124)
# plt.clf()
# plt.subplot(2,3,1)
# fout0.visualize(flag_axis=True,cmap='turbo',plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
# plt.clim(0.22,0.245)
# plt.title('$u_{backprop}(x,y;\lambda = ' + str(round(float(fieldIn.wavelengths.data_tensor[0]*1e9),3)) + 'nm)$', fontsize=16)
# plt.scatter(scattererLocsY, scattererLocsX, s=tempSize2, marker=tempMarker2, color=tempColor2, edgecolor='none', label='Scatterer')
# plt.legend()
# plt.xlabel('Position (mm)', fontsize=14)
# plt.ylabel('Position (mm)', fontsize=14)
# plt.xlim(tempLim2a,tempLim2b)
# plt.ylim(tempLim2a,tempLim2b)
# plt.subplot(2,3,2)
# fout1.visualize(flag_axis=True,cmap='turbo',plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
# plt.clim(0.22,0.245)
# plt.title('$u_{backprop}(x,y;\lambda = ' + str(round(float(fieldIn.wavelengths.data_tensor[1]*1e9),3)) + 'nm)$', fontsize=16)
# plt.scatter(scattererLocsY, scattererLocsX, s=tempSize2, marker=tempMarker2, color=tempColor2, edgecolor='none', label='Scatterer')
# plt.legend()
# plt.xlabel('Position (mm)', fontsize=14)
# plt.ylabel('Position (mm)', fontsize=14)
# plt.xlim(tempLim2a,tempLim2b)
# plt.ylim(tempLim2a,tempLim2b)
# plt.subplot(2,3,3)
# fieldOut.visualize(flag_axis=True,cmap='turbo',plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
# plt.clim(0.000263,0.00026319269090890884)
# plt.title('$u_{backprop,synth}(x,y;\Lambda = 0.05mm)$', fontsize=16)
# plt.scatter(scattererLocsY, scattererLocsX, s=tempSize2, marker=tempMarker2, color=tempColor2, edgecolor='none', label='Scatterer')
# plt.legend()
# plt.xlabel('Position (mm)', fontsize=14)
# plt.ylabel('Position (mm)', fontsize=14)
# plt.xlim(tempLim2a,tempLim2b)
# plt.ylim(tempLim2a,tempLim2b)



# tempLim1 = 1
# tempColor1 = 'red'
# tempMarker1 = 'x'
# tempSize1 = 160
# fout0 = get_field_slice(fieldOut_o1,channel_inds_range=0)
# fout1 = get_field_slice(fieldOut_o1,channel_inds_range=1)
# plt.figure(1124)
# plt.clf()
# plt.subplot(2,3,1)
# fout0.visualize(flag_axis=True,cmap='turbo',plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
# plt.title('$u_{backprop}(x,y;\lambda = ' + str(round(float(fieldIn.wavelengths.data_tensor[0]*1e9),3)) + 'nm)$', fontsize=16)
# plt.scatter(scattererLocsY, scattererLocsX, s=tempSize1, marker=tempMarker1, color=tempColor1, edgecolor='none', label='Scatterer Location')
# plt.legend()
# plt.xlabel('Position (mm)', fontsize=14)
# plt.ylabel('Position (mm)', fontsize=14)
# plt.xlim(-tempLim1,tempLim1)
# plt.ylim(-tempLim1,tempLim1)
# plt.subplot(2,3,2)
# fieldOut.visualize(flag_axis=True,cmap='turbo',plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
# plt.title('$u_{backprop,synth}(x,y;\Lambda = 0.05mm)$', fontsize=16)
# plt.scatter(scattererLocsY, scattererLocsX, s=tempSize1, marker=tempMarker1, color=tempColor1, edgecolor='none', label='Scatterer Location')
# plt.legend()
# plt.xlabel('Position (mm)', fontsize=14)
# plt.ylabel('Position (mm)', fontsize=14)
# plt.xlim(-tempLim1,tempLim1)
# plt.ylim(-tempLim1,tempLim1)
# plt.subplot(2,3,3)
# fieldOut.visualize(flag_axis=True,cmap='turbo',plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
# plt.clim(100, 200)
# plt.title('Scatterer Location Predictions', fontsize=16)
# plt.scatter(scattererLocsY, scattererLocsX, s=tempSize1, marker=tempMarker1, color=tempColor1, edgecolor='none', label='Scatterer Location')
# plt.scatter(0.385, 0.45, s=tempSize1, marker='s', linewidth=2, color='none', edgecolor='white', label='Predicted Location ($\lambda = ' + str(round(float(fieldIn.wavelengths.data_tensor[0]*1e9),3)) + 'nm$)')
# plt.scatter(0.3855, 0.3953, s=tempSize1, marker='s', linewidth=2, color='none', edgecolor='orange', label='Predicted Location ($\Lambda = 0.05mm$)')
# plt.legend(loc='lower left')
# plt.xlabel('Position (mm)', fontsize=14)
# plt.ylabel('Position (mm)', fontsize=14)
# plt.xlim(0.3, 0.5)
# plt.ylim(0.3, 0.5)
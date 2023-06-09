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

sys.path.append(os.path.dirname(__file__) + "\\..\\")
sys.path.append(os.path.dirname(__file__) + "\\..\\holotorch-lib\\")
sys.path.append(os.path.dirname(__file__) + "\\..\\holotorch-lib\\holotorch")

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

# Random code for Hadamard basis vectors:
#	aaa = generateHadamardBasisVector(list(range(64*64)),64,64,1)
#	bbb = torch.matmul(aaa.reshape(4096,4096), aaa.reshape(4096,4096).transpose(-2,-1))

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
					asmStateDict = copy.deepcopy(cur).__dict__	# copy.deepcopy(cur) was done because when "prop.__setstate__(asmStateDict)"" is called, prop shares its fields
																# with the copy (i.e. the fields have the same pointer).  If copy.deepcopy(cur) was not called, modifying prop's
																# fields later (e.g. the "prop.z = totalDist" and "prop.prop_kernel = None" lines) would modify the fields of a
																# component in the original model, which is undesirable (mainly, overwriting the 'z' field is undesirable).
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


def getFieldsAtScattererPlane(
								vecIn : torch.Tensor,
								samplingBoolMask,
								fieldPrototype,
								inputModel,
								backpropModel,
								doSyntheticWavelengths : bool = False,
							):
	fieldIn = TransferMatrixProcessor.getModelInputField(macropixelVector=vecIn, samplingBoolMask=samplingBoolMask, fieldPrototype=fieldPrototype)

	o1 = inputModel(fieldIn)

	if backpropModel is not None:
		if doSyntheticWavelengths:
			synthFieldData = torch.zeros(1,1,1,1,o1.data.shape[-2],o1.data.shape[-1], device=device) + 0j
			synthFieldData[..., :, :] = o1.data[0,0,0,0,:,:] * o1.data[0,0,0,1,:,:].conj()
			synthField = ElectricField(
										data = synthFieldData,
										wavelengths = float(fieldIn.wavelengths.data_tensor[0]*fieldIn.wavelengths.data_tensor[1]/((fieldIn.wavelengths.data_tensor[1]-fieldIn.wavelengths.data_tensor[0]).abs())),
										spacing = float(o1.spacing.data_tensor[...,0].squeeze())
									)
			synthField.wavelengths.to(device)
			synthField.spacing.to(device)

			backpropModelInputField = synthField
			fieldOut = backpropModel(synthField)
		else:
			backpropModelInputField = o1
			fieldOut = backpropModel(o1)
	else:
		fieldOut = o1
		backpropModelInputField = None

	return fieldIn, backpropModelInputField, fieldOut


def visualizeScattererPlaneField(
									field,
									xLims : None,
									yLims : None,
									titleStr : None,
									titleFontSize : int = 16,
									plotScatterers : bool = True,
									scattererLocsX : list = [],
									scattererLocsY : list = [],
									plot_type : ENUM_PLOT_TYPE = ENUM_PLOT_TYPE.MAGNITUDE,
									plot_cmap : str = 'turbo'
								):
	field.visualize(flag_axis=True, plot_type=plot_type, cmap=plot_cmap)
	if plotScatterers:
		plt.scatter(scattererLocsY, scattererLocsX, s=192, marker='x', alpha=0.75, linewidths=4, color='red', edgecolor='red', label='Scatterer')		# X and Y are switched because HoloTorch has the horizontal and vertical dimensions switched (relative to what the plots consider as horizontal and vertical)
	if xLims is not None:
		plt.xlim(xLims)
	if yLims is not None:
		plt.ylim(yLims)
	if titleStr is not None:
		plt.title(titleStr, fontsize=titleFontSize)
	if plotScatterers:
		plt.legend()


################################################################################################################################








################################################################################################################################
################################################################################################################################
#	SETTINGS
################################################################################################################################
use_cuda = True
gpu_no = 0
device = torch.device("cuda:"+str(gpu_no) if use_cuda else "cpu")
################################################################################################################################
# dataFilePath = 'DATA/Temp3/Experiment_2023-5-9_14h51m01s.pt'
# dataFilePath = 'DATA/THESIS DATA (OLD)/Experiment_2023-5-29_16h06m12s.pt'
# dataFilePath = 'DATA/THESIS DATA (OLD)/Experiment_2023-5-31_16h36m58s.pt'
dataFilePath = 'DATA/THESIS DATA/Experiment_2023-6-3_09h26m16s.pt'
################################################################################################################################
# IMPORTANT NOTE:
#	See the source code notes above the eigenstructure demixing method ('demixEigenstructure' in the TransferMatrixProcessor class) for more information. 
doEigenstructureDemixing = False
singValMagnitudeSimilarityThreshold = 0.15
################################################################################################################################
# Plot settings
xLims1 = [-1, 1]
yLims1 = [-1, 1]
coordsMultiplier1 = 1e3	# Scale for millimeters
numToPlot = 3
subplotTitleFontSize = 16
################################################################################################################################
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


################################################################################################################################

# Deriving some quantities
scattererLocsX = []
scattererLocsY = []
for i in range(len(scattererModel.scatterers)):
	sTemp = scattererModel.scatterers[i]
	scattererLocsX = scattererLocsX + [sTemp.location_x * coordsMultiplier1]
	scattererLocsY = scattererLocsY + [sTemp.location_y * coordsMultiplier1]

syntheticWavelength = float((loadedData['Field_Input_Prototype'].wavelengths.data_tensor[0] * loadedData['Field_Input_Prototype'].wavelengths.data_tensor[1]) / (loadedData['Field_Input_Prototype'].wavelengths.data_tensor[1] - loadedData['Field_Input_Prototype'].wavelengths.data_tensor[0]))

################################################################################################################################

# Plotting


# For making a specific figure
plt.figure(20)
vecIn = V[... , :, 0]
fieldIn, backpropModelInputField, fieldOut = getFieldsAtScattererPlane(	vecIn=vecIn,
											samplingBoolMask=inputBoolMask,
											fieldPrototype=loadedData['Field_Input_Prototype'],
											inputModel=inputModel, backpropModel=backpropModel,
											doSyntheticWavelengths=False
										)
plt.subplot(2, 4, 1)
get_field_slice(fieldIn, channel_inds_range=0).visualize(cmap='turbo', flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
plt.title(r"$u_{slm}(x, y; \lambda_1)$", fontsize=36)
plt.subplot(2, 4, 5)
get_field_slice(fieldIn, channel_inds_range=1).visualize(cmap='turbo', flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
plt.title(r"$u_{slm}(x, y; \lambda_2)$", fontsize=36)
plt.subplot(2, 4, 2)
get_field_slice(backpropModelInputField, channel_inds_range=0).visualize(cmap='turbo', flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
plt.title(r"$u_{out}(x, y; \lambda_1)$", fontsize=36)
plt.subplot(2, 4, 6)
get_field_slice(backpropModelInputField, channel_inds_range=1).visualize(cmap='turbo', flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
plt.title(r"$u_{out}(x, y; \lambda_2)$", fontsize=36)
plt.subplot(2, 4, 3)
visualizeScattererPlaneField(	field=get_field_slice(fieldOut, channel_inds_range=0),
								xLims=xLims1, yLims=yLims1,
								titleStr="",
								titleFontSize=subplotTitleFontSize,
								plotScatterers=False, scattererLocsX=scattererLocsX, scattererLocsY=scattererLocsY,
								plot_type=ENUM_PLOT_TYPE.MAGNITUDE,
								plot_cmap='turbo'
							)
plt.subplot(2, 4, 7)
visualizeScattererPlaneField(	field=get_field_slice(fieldOut, channel_inds_range=1),
								xLims=xLims1, yLims=yLims1,
								titleStr="",
								titleFontSize=subplotTitleFontSize,
								plotScatterers=False, scattererLocsX=scattererLocsX, scattererLocsY=scattererLocsY,
								plot_type=ENUM_PLOT_TYPE.MAGNITUDE,
								plot_cmap='turbo'
							)
fieldIn, backpropModelInputField, fieldOut = getFieldsAtScattererPlane(	vecIn=vecIn,
											samplingBoolMask=inputBoolMask,
											fieldPrototype=loadedData['Field_Input_Prototype'],
											inputModel=inputModel, backpropModel=backpropModel,
											doSyntheticWavelengths=True
										)
plt.subplot(2, 4, 4)
backpropModelInputField.visualize(cmap='turbo', flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
plt.subplot(2, 4, 8)
visualizeScattererPlaneField(	field=fieldOut,
								xLims=xLims1, yLims=yLims1,
								titleStr="",
								titleFontSize=subplotTitleFontSize,
								plotScatterers=False, scattererLocsX=scattererLocsX, scattererLocsY=scattererLocsY,
								plot_type=ENUM_PLOT_TYPE.MAGNITUDE,
								plot_cmap='turbo'
							)



for i in range(numToPlot):
	vecIn = V[... , :, i]
	fieldIn, _, _ = getFieldsAtScattererPlane(	vecIn=vecIn,
												samplingBoolMask=inputBoolMask,
												fieldPrototype=loadedData['Field_Input_Prototype'],
												inputModel=inputModel, backpropModel=backpropModel,
												doSyntheticWavelengths=False
											)
	fieldTemp = preScattereModel(fieldIn)
	plt.figure(0)
	plt.subplot(2, numToPlot, i + 1)
	visualizeScattererPlaneField(	field=get_field_slice(fieldTemp, channel_inds_range=0),
									xLims=xLims1, yLims=yLims1,
									titleStr=r"From right singular vector #%d (Magnitude)" "\n" r"$\lambda$ = %.3f nm" % (i+1, float(loadedData['Field_Input_Prototype'].wavelengths.data_tensor[0]) * 1e9),
									titleFontSize=subplotTitleFontSize,
									plotScatterers=True, scattererLocsX=scattererLocsX, scattererLocsY=scattererLocsY,
									plot_type=ENUM_PLOT_TYPE.MAGNITUDE,
									plot_cmap='turbo'
								)
	plt.subplot(2, numToPlot, numToPlot + i + 1)
	visualizeScattererPlaneField(	field=get_field_slice(fieldTemp, channel_inds_range=1),
									xLims=xLims1, yLims=yLims1,
									titleStr=r"From right singular vector #%d (Magnitude)" "\n" r"$\lambda$ = %.3f nm" % (i+1, float(loadedData['Field_Input_Prototype'].wavelengths.data_tensor[1]) * 1e9),
									titleFontSize=subplotTitleFontSize,
									plotScatterers=True, scattererLocsX=scattererLocsX, scattererLocsY=scattererLocsY,
									plot_type=ENUM_PLOT_TYPE.MAGNITUDE,
									plot_cmap='turbo'
								)

plt.suptitle("Backpropagated fields (with aberrating layer model)", fontsize=24, fontweight="bold")








for i in range(numToPlot):
	vecIn = V[... , :, i]
	fieldIn, backpropModelInputField, fieldOut = getFieldsAtScattererPlane(	vecIn=vecIn,
												samplingBoolMask=inputBoolMask,
												fieldPrototype=loadedData['Field_Input_Prototype'],
												inputModel=inputModel, backpropModel=backpropModel,
												doSyntheticWavelengths=False
											)
	plt.figure(1)
	plt.subplot(2, numToPlot, i + 1)
	get_field_slice(fieldIn, channel_inds_range=0).visualize(cmap='turbo', flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
	plt.title(r"From right singular vector #%d (Magnitude)" "\n" r"$\lambda$ = %.3f nm" % (i+1, float(loadedData['Field_Input_Prototype'].wavelengths.data_tensor[0]) * 1e9), fontsize=subplotTitleFontSize)
	plt.subplot(2, numToPlot, numToPlot + i + 1)
	get_field_slice(fieldIn, channel_inds_range=0).visualize(cmap='twilight', flag_axis=True, plot_type=ENUM_PLOT_TYPE.PHASE)
	plt.title(r"From right singular vector #%d (Phase)" "\n" r"$\lambda$ = %.3f nm" % (i+1, float(loadedData['Field_Input_Prototype'].wavelengths.data_tensor[0]) * 1e9), fontsize=subplotTitleFontSize)
	
	plt.figure(2)
	plt.subplot(2, numToPlot, i + 1)
	get_field_slice(fieldIn, channel_inds_range=1).visualize(cmap='turbo', flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
	plt.title(r"From right singular vector #%d (Magnitude)" "\n" r"$\lambda \approx$ %.3f nm" % (i+1, float(loadedData['Field_Input_Prototype'].wavelengths.data_tensor[1]) * 1e9), fontsize=subplotTitleFontSize)
	plt.subplot(2, numToPlot, numToPlot + i + 1)
	get_field_slice(fieldIn, channel_inds_range=1).visualize(cmap='twilight', flag_axis=True, plot_type=ENUM_PLOT_TYPE.PHASE)
	plt.title(r"From right singular vector #%d (Phase)" "\n" r"$\lambda \approx$ %.3f nm" % (i+1, float(loadedData['Field_Input_Prototype'].wavelengths.data_tensor[1]) * 1e9), fontsize=subplotTitleFontSize)

	plt.figure(3)
	plt.subplot(2, numToPlot, i + 1)
	visualizeScattererPlaneField(	field=get_field_slice(fieldOut, channel_inds_range=0),
									xLims=xLims1, yLims=yLims1,
									titleStr=r"From right singular vector #%d (Magnitude)" "\n" r"$\lambda$ = %.3f nm" % (i+1, float(loadedData['Field_Input_Prototype'].wavelengths.data_tensor[0]) * 1e9),
									titleFontSize=subplotTitleFontSize,
									plotScatterers=True, scattererLocsX=scattererLocsX, scattererLocsY=scattererLocsY,
									plot_type=ENUM_PLOT_TYPE.MAGNITUDE,
									plot_cmap='turbo'
								)
	plt.subplot(2, numToPlot, numToPlot + i + 1)
	visualizeScattererPlaneField(	field=get_field_slice(fieldOut, channel_inds_range=1),
									xLims=xLims1, yLims=yLims1,
									titleStr=r"From right singular vector #%d (Magnitude)" "\n" r"$\lambda$ = %.3f nm" % (i+1, float(loadedData['Field_Input_Prototype'].wavelengths.data_tensor[1]) * 1e9),
									titleFontSize=subplotTitleFontSize,
									plotScatterers=True, scattererLocsX=scattererLocsX, scattererLocsY=scattererLocsY,
									plot_type=ENUM_PLOT_TYPE.MAGNITUDE,
									plot_cmap='turbo'
								)



plt.figure(1)
plt.suptitle("Input fields from right singular vectors", fontsize=24, fontweight="bold")
plt.figure(2)
plt.suptitle("Input fields from right singular vectors", fontsize=24, fontweight="bold")
plt.figure(3)
plt.suptitle("Scatterer plane fields from backpropagated right singular vectors", fontsize=24, fontweight="bold")


for i in range(numToPlot):
	if (((i + 1) % 10) == 1):
		numberingSuffix = "st"
	elif (((i + 1) % 10) == 2):
		numberingSuffix = "nd"
	elif (((i + 1) % 10) == 3):
		numberingSuffix = "rd"
	else:
		numberingSuffix = "th"
	numberingStr = str(i + 1) + numberingSuffix
	
	vecIn = V[... , :, i]
	_, backpropModelInputField, fieldOut = getFieldsAtScattererPlane(	vecIn=vecIn,
												samplingBoolMask=inputBoolMask,
												fieldPrototype=loadedData['Field_Input_Prototype'],
												inputModel=inputModel, backpropModel=backpropModel,
												doSyntheticWavelengths=True
											)
	plt.figure(4)
	plt.subplot(2, numToPlot, i + 1)
	backpropModelInputField.visualize(cmap='turbo', flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
	plt.title(r"From the %s right singular vectors (Magnitude)" "\n" r"$\Lambda$ = %.3f nm" % (numberingStr, syntheticWavelength*1e3), fontsize=subplotTitleFontSize)
	plt.subplot(2, numToPlot, numToPlot + i + 1)
	backpropModelInputField.visualize(cmap='twilight', flag_axis=True, plot_type=ENUM_PLOT_TYPE.PHASE)
	plt.title(r"From the %s right singular vectors (Phase)" "\n" r"$\Lambda$ = %.3f nm" % (numberingStr, syntheticWavelength*1e3), fontsize=subplotTitleFontSize)

	plt.figure(5)
	plt.subplot(2, numToPlot, i + 1)
	visualizeScattererPlaneField(	field=fieldOut,
									xLims=xLims1, yLims=yLims1, 
									titleStr="From the %s right singular vectors (Magnitude)" "\n" r"$\Lambda$ = %.3f nm" % (numberingStr, syntheticWavelength*1e3),
									titleFontSize=subplotTitleFontSize,
									plotScatterers=True, scattererLocsX=scattererLocsX, scattererLocsY=scattererLocsY,
									plot_type=ENUM_PLOT_TYPE.MAGNITUDE,
									plot_cmap='turbo'
								)
	plt.title(r"From the %s right singular vectors (Magnitude)" "\n" r"$\Lambda$ = %.3f nm" % (numberingStr, syntheticWavelength*1e3), fontsize=subplotTitleFontSize)


plt.figure(4)
plt.suptitle("Synthetic fields to be backpropagated", fontsize=24, fontweight="bold")
plt.figure(5)
plt.suptitle("Scatterer plane fields from backpropagated synthetic fields", fontsize=24, fontweight="bold")

plt.show()

pass
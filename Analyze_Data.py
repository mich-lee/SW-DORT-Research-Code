import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
import copy
import datetime
# import pathlib
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
			raise Exception("Error")

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
		if type(model[i]) is not ScattererModel:
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

# aaa = generateHadamardBasisVector(list(range(64*64)),64,64,1)
# bbb = torch.matmul(aaa.reshape(4096,4096), aaa.reshape(4096,4096).transpose(-2,-1))

################################################################################################################################

use_cuda = True
gpu_no = 0
device = torch.device("cuda:"+str(gpu_no) if use_cuda else "cpu")

################################################################################################################################

# loadedData = torch.load('DATA/Experiment_202210201534.pt', map_location=device)
# loadedData = torch.load('DATA/Experiment_202210192329.pt', map_location=device)
# loadedData = torch.load('DATA/Experiment_2022-10-27_19h14m38s.pt', map_location=device)
# loadedData = torch.load('DATA/Experiment_2022-11-6_22h18m58s.pt', map_location=device)
# loadedData = torch.load('DATA/Experiment_2022-11-16_20h46m21s.pt', map_location=device)
# loadedData = torch.load('DATA/Experiment_2022-11-22_13h38m51s.pt', map_location=device)
# loadedData = torch.load('DATA/Experiment_2022-11-22_20h00m26s.pt', map_location=device)		# Good data?
# loadedData = torch.load('DATA/Blah/Experiment_2023-1-3_17h03m58s.pt', map_location=device)
# loadedData = torch.load('DATA/GOOD DATA/Single Scatterer Demo/Experiment_2023-1-7_21h35m04s.pt', map_location=device)
# loadedData = torch.load('DATA/Experiment_2023-1-7_23h04m14s.pt', map_location=device)
loadedData = torch.load('DATA/Experiment_2023-1-8_04h34m16s.pt', map_location=device)

################################################################################################################################

doEigenstructureDemixing = True

################################################################################################################################

model = loadedData['Model']
inputBoolMask = loadedData['Input_Bool_Mask']
outputBoolMask = loadedData['Output_Bool_Mask']

H = loadedData['Transfer_Matrix']

print("Taking singular value decomposition of the transfer matrix...", end='')
U, S, Vh = torch.linalg.svd(H)
V = Vh.conj().transpose(-2, -1)
print("Done!")

if doEigenstructureDemixing:
	print("Demixing eigenstructure...", end='')
	U, S, V = TransferMatrixProcessor.demixEigenstructure(U, S, V, 'V')
	Vh = V.conj().transpose(-2, -1)
	print("Done!")
else:
	print("Skipped eigenstructure demixing step.")

q = torch.matmul(Vh[0,0,0,1,:,:], V[0,0,0,0,:,:])
w = (S[0,0,0,1,:][:,None] + S[0,0,0,0,:][None,:]) / 2

# tempResampler = Field_Resampler(outputHeight=8192, outputWidth=8192, outputPixel_dx=6.4*um, outputPixel_dy=6.4*um, device=device)
preScattereModel, scattererModel, inputModel, backpropModel = getInputAndBackpropagationModels(model)



# Plot settings
nCols1 = 3
nRows1 = 1
xLims1 = [-3, 3]
yLims1 = [-3, 3]
coordsMultiplier1 = 1e3	# Scale for millimeters

# Deriving some quantities
nSubplots1 = nCols1 * nRows1
scattererLocsX = []
scattererLocsY = []
for i in range(len(scattererModel.scatterers)):
	sTemp = scattererModel.scatterers[i]
	scattererLocsX = scattererLocsX + [sTemp.location_x * coordsMultiplier1]
	scattererLocsY = scattererLocsY + [sTemp.location_y * coordsMultiplier1]


for singVecNum in range(nSubplots1 * 8):
	vecIn = V[... , :, singVecNum]
	fieldIn = TransferMatrixProcessor.getModelInputField(macropixelVector=vecIn, samplingBoolMask=inputBoolMask, fieldPrototype=loadedData['Field_Input_Prototype'])

	o1 = inputModel(fieldIn)
	fieldOut_o1 = backpropModel(o1)
	
	# Resample to force spacing to be the same for all dimensions (B, T, P, and C)
	# o1 = model[0](o1)

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


	subplotNum = (singVecNum % nSubplots1) + 1
	plt.figure(int(np.floor(singVecNum / nSubplots1)) + 1)
	if subplotNum == 1:
		plt.clf()

	plt.subplot(nRows1*2, nCols1, subplotNum)
	fieldOut.visualize(flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE, cmap='turbo')
	plt.scatter(scattererLocsY, scattererLocsX, s=96, marker='+', color='red', edgecolor='none', label='Scatterer')		# X and Y are switched because HoloTorch has the horizontal and vertical dimensions switched (relative to what the plots consider as horizontal and vertical)
	plt.xlim(xLims1)
	plt.ylim(yLims1)
	plt.title("[REDACTED] #" + str(singVecNum + 1))
	plt.legend()

	plt.subplot(nRows1*2, nCols1, subplotNum + nCols1)
	fieldOut.visualize(flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE, cmap='Paired')
	plt.scatter(scattererLocsY, scattererLocsX, s=96, marker='+', color='black', edgecolor='none', label='Scatterer')		# X and Y are switched because HoloTorch has the horizontal and vertical dimensions switched (relative to what the plots consider as horizontal and vertical)
	plt.xlim(xLims1)
	plt.ylim(yLims1)
	plt.title("[REDACTED] #" + str(singVecNum + 1))
	plt.legend()

	plt.show()

pass
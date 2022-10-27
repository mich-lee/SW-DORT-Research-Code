import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
import copy
import datetime
# import pathlib

import warnings
from ScattererModel import Scatterer, ScattererModel
from TransferMatrixProcessor import TransferMatrixProcessor
from holotorch.Optical_Components.Field_Resampler import Field_Resampler

sys.path.append("holotorch-lib/")
sys.path.append("holotorch-lib/holotorch")

import holotorch.utils.Dimensions as Dimensions
from holotorch.utils.Enumerators import *
from holotorch.utils.units import * # E.g. to get nm, um, mm etc.
# from holotorch.LightSources.CoherentSource import CoherentSource
from holotorch.Spectra.WavelengthContainer import WavelengthContainer
from holotorch.Spectra.SpacingContainer import SpacingContainer
from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.Optical_Propagators.ASM_Prop import ASM_Prop
# from holotorch.Optical_Components.Resize_Field import Resize_Field
# from holotorch.Sensors.Detector import Detector
from holotorch.Optical_Components.FT_Lens import FT_Lens
from holotorch.Optical_Components.Thin_Lens import Thin_Lens
from holotorch.Optical_Components.SimpleMask import SimpleMask
from holotorch.Optical_Components.Field_Padder_Unpadder import Field_Padder_Unpadder

from holotorch.utils.Field_Utils import get_field_slice, applyFilterSpaceDomain

warnings.filterwarnings('always',category=UserWarning)

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

################################################################################################################################

use_cuda = True
gpu_no = 0
device = torch.device("cuda:"+str(gpu_no) if use_cuda else "cpu")

################################################################################################################################

# aaa = generateHadamardBasisVector(list(range(64*64)),64,64,1)
# bbb = torch.matmul(aaa.reshape(4096,4096), aaa.reshape(4096,4096).transpose(-2,-1))

################################################################################################################################

loadedData = torch.load('Experiment_202210201534.pt', map_location=device)
# loadedData = torch.load('Experiment_202210192329.pt', map_location=device)
H = loadedData['Transfer_Matrix']
U, S, Vh = torch.linalg.svd(H)
V = Vh.conj().transpose(-2, -1)

model = loadedData['Model']
fieldInputPadder = model[0]
thinLens1 = model[1][1]
# asmProp1a = model[1][0]
asmProp1a = ASM_Prop(init_distance=model[1][0]._z)
scattererModel = model[2]

fieldIn = copy.deepcopy(loadedData['Field_Input_Prototype'])
fieldIn.data[...] = 0

inputBoolMask = loadedData['Input_Bool_Mask']
outputBoolMask = loadedData['Output_Bool_Mask']

q = torch.matmul(Vh[0,0,0,1,:,:], V[0,0,0,0,:,:])
w = (S[0,0,0,1,:][:,None] + S[0,0,0,0,:][None,:]) / 2

v1 = V[0,0,0,0,:,288]
v2 = V[0,0,0,1,:,287]

m1 = torch.nn.Sequential(fieldInputPadder, asmProp1a, thinLens1)

fieldIn.data[0,0,0,0,inputBoolMask] = v1
fieldIn.data[0,0,0,1,inputBoolMask] = v2

o1 = m1(fieldIn)

synthFieldData = torch.zeros(1,1,1,1,o1.data.shape[-2],o1.data.shape[-1], device=device) + 0j
synthFieldData[..., :, :] = o1.data[0,0,0,0,:,:] * o1.data[0,0,0,1,:,:].conj()
synthField = ElectricField(
							data = synthFieldData,
							wavelengths = float(fieldIn.wavelengths.data_tensor[0]*fieldIn.wavelengths.data_tensor[1]/(fieldIn.wavelengths.data_tensor[1]-fieldIn.wavelengths.data_tensor[0])),
							spacing = float(o1.spacing.data_tensor[...,0].squeeze())
						)
synthField.wavelengths.to(device)
synthField.spacing.to(device)

pass




# tempSingVecInd = 100
# fieldIn.data[...] = torch.zeros(1,1,1,wavelengthContainer.data_tensor.numel(),inputRes[0],inputRes[1],device=device) + 0j
# fieldIn.data[...,inputBoolMask] = V[0,0,0,0,:,tempSingVecInd]
# aaa = torch.zeros(fieldIn.data.shape[-2:], device=device)
# # aaa[270,480] = 1
# # aaa[263:278,473:488]
# aaa[263:278,473:488] = (-torch.tensor(range(-7,8)).abs() - torch.tensor([list(range(-7,8))]).abs().transpose(-2,-1)) + 14
# fieldIn.data[...,:,:] = applyFilterSpaceDomain(aaa, fieldIn.data[...,:,:])
# # fieldIn.data = fieldIn.data * S[... , tempSingVecInd]
# temp2 = lensSys1(fieldIn)
# plt.clf()
# plt.subplot(1,2,1)
# fieldIn.visualize(flag_axis=True)
# plt.subplot(1,2,2)
# temp2.visualize(flag_axis=True)
# # scattererModel(temp2).visualize(flag_axis=True)
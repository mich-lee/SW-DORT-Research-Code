import numpy as np
import sys
import torch
import matplotlib.pyplot as plt

# from turtle import clone

# import pathlib
import copy

from numpy import asarray
# import gc	# For garbage collection/freeing up memory

# Image wranglers
# import imageio
# from PIL import Image

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

from holotorch.utils.Field_Utils import get_field_slice

warnings.filterwarnings('always',category=UserWarning)

################################################################################################################################

from holotorch.Optical_Components.CGH_Component import CGH_Component
class testModel(CGH_Component):
	def __init__(self):
		super().__init__()
		self.initializedFlag = False

	def initializeOutput(self, field):
		data = torch.zeros(list(field.data.shape[0:-2]) + [1, field.data.shape[-1]*field.data.shape[-2]], device=field.data.device)
		data[..., :] = torch.tensor(range(data.shape[-1]))
		data = data.view(list(data.shape[0:-2]) + [field.data.shape[-2], field.data.shape[-1]])
		self.output = data

	def forward(self, field : ElectricField) -> ElectricField:
		if not self.initializedFlag:
			self.initializeOutput(field)
			self.initializedFlag = True
		elif (field.data.shape != self.output.shape):
			self.initializeOutput(field)

		data = self.output * field.data
		out = ElectricField(data=data, wavelengths=field.wavelengths, spacing=field.spacing)
		return out

################################################################################################################################


use_cuda = True
gpu_no = 0
device = torch.device("cuda:"+str(gpu_no) if use_cuda else "cpu")


syntheticWavelength = 0.1*mm
lambda1 = 854*nm
lambda2 = lambda1 * syntheticWavelength / (syntheticWavelength - lambda1)

# wavelengths = [lambda1, lambda2]
wavelengths = [lambda1]
inputSpacing = 6.4*um
inputRes = (540, 960)
outputRes = (380, 500)
outputSpacing = 8*1.85*um


centerXInd = int(np.floor((inputRes[0] - 1) / 2))
centerYInd = int(np.floor((inputRes[1] - 1) / 2))

if np.isscalar(wavelengths):
	wavelengthContainer = WavelengthContainer(wavelengths=wavelengths, tensor_dimension=Dimensions.C(n_channel=1))
elif isinstance(wavelengths, list):
	wavelengthContainer = WavelengthContainer(wavelengths=wavelengths, tensor_dimension=Dimensions.C(n_channel=np.size(wavelengths)))
spacingContainer = SpacingContainer(spacing=inputSpacing)


fieldData = torch.zeros(1,1,1,wavelengthContainer.data_tensor.numel(),inputRes[0],inputRes[1],device=device) + 0j
# fieldData[...,centerXInd:centerXInd+1,centerYInd:centerYInd+1] = 1
fieldData[...,0,0] = 1
# fieldData[...,-1,0] = 1
# fieldData[...,0,-1] = 1
# fieldData[...,-1,-1] = 1
# fieldData[...,:,:] = torch.exp(1j*10*(2*np.pi/res[0])*torch.tensor(range(res[0])).repeat([res[1],1]))

fieldIn = ElectricField(data=fieldData, wavelengths=wavelengthContainer, spacing=spacingContainer)
fieldIn.wavelengths.to(device=device)
fieldIn.spacing.to(device=device)




################################################################################################################################

fieldInputPadder = Field_Padder_Unpadder(pad_x = int(1.5*inputRes[0]), pad_y = int(1.5*inputRes[1]))

asmProp1a = ASM_Prop(init_distance=140*mm, do_padding=True)
thinLens1 = Thin_Lens(focal_length=75*mm)
asmProp1b = ASM_Prop(init_distance=150*mm, do_padding=True)
lensSys1 = torch.nn.Sequential(asmProp1a, thinLens1, asmProp1b)

scattererList = [
					# Scatterer(location_x=0, location_y=0.065*mm, diameter=0.015*mm, scatteringResponse=1),
					# Scatterer(location_x=0.2*mm, location_y=0.2*mm, diameter=0.015*mm, scatteringResponse=1),
					# Scatterer(location_x=0, location_y=0.07*mm, diameter=0.015*mm, scatteringResponse=1),
					# Scatterer(location_x=0*mm, location_y=0*mm, diameter=0.3*mm, scatteringResponse=1),
					Scatterer(location_x=0.25*mm, location_y=0.25*mm, diameter=0.2*mm, scatteringResponse=1),
					Scatterer(location_x=-0.25*mm, location_y=-0.25*mm, diameter=0.2*mm, scatteringResponse=1),
				]
scattererModel = ScattererModel(scattererList)

# focalLength2 = 75*mm
# asmProp2a = ASM_Prop(init_distance=focalLength2)
# thinLens2 = Thin_Lens(focal_length=focalLength2)
# asmProp2b = ASM_Prop(init_distance=focalLength2)
# lensSys2 = torch.nn.Sequential(asmProp2a, thinLens2, asmProp2b)

fieldOutputResampler = Field_Resampler(outputHeight=outputRes[0], outputWidth=outputRes[1], outputPixel_dx=outputSpacing, outputPixel_dy=outputSpacing, device=device)

model = torch.nn.Sequential(fieldInputPadder, lensSys1, scattererModel, lensSys1, fieldOutputResampler)

################################################################################################################################


if True:
	# model2 = testModel()	#torch.nn.Identity()
	transferMtxMeasurer = TransferMatrixProcessor(	inputFieldPrototype=fieldIn,
													inputBoolMask=TransferMatrixProcessor.getUniformSampleBoolMask(inputRes[0], inputRes[1], 36, 64),
													outputBoolMask=TransferMatrixProcessor.getUniformSampleBoolMask(outputRes[0], outputRes[1], 60, 80),
													# inputBoolMask=TransferMatrixProcessor.getUniformSampleBoolMask(inputRes[0], inputRes[1], 8, 8),
													# outputBoolMask=TransferMatrixProcessor.getUniformSampleBoolMask(inputRes[0], inputRes[1], 8, 8),
													model=model,
													numParallelColumns=8)
	asdf = transferMtxMeasurer.measureTransferMatrix()
elif False:
	H = torch.load('H_mtx_202210192329.pt', map_location=device)
	U, S, Vh = torch.linalg.svd(H)
	V = Vh.conj().transpose(-2, -1)






# fieldA = asmProp2(thinLens(asmProp1(fieldInputPadder(fieldIn))))
# fieldB = ASM_Prop(init_distance=75*mm, do_padding=False)(thinLens(asmProp1(fieldInputPadder(fieldIn))))

fieldInputPadder.add_output_hook()
thinLens1.add_output_hook()
asmProp1a.add_output_hook()
asmProp1b.add_output_hook()
# thinLens2.add_output_hook()
# asmProp2a.add_output_hook()
# asmProp2b.add_output_hook()
scattererModel.add_output_hook()
fieldBlah = model(fieldIn)
plt.clf()
plt.subplot(2,4,1)
get_field_slice(fieldInputPadder.outputs[-1], channel_inds_range=0, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
plt.title("Padded Input")
plt.subplot(2,4,2)
get_field_slice(asmProp1a.outputs[-2], channel_inds_range=0, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
plt.title("After Propagation")
plt.subplot(2,4,3)
get_field_slice(thinLens1.outputs[-2], channel_inds_range=0, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
plt.title("Lens Output")
plt.subplot(2,4,4)
get_field_slice(asmProp1b.outputs[-2], channel_inds_range=0, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
plt.title("After Propagation")
plt.subplot(2,4,5)
get_field_slice(scattererModel.outputs[-1], channel_inds_range=0, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
plt.title("After Scattering")
plt.subplot(2,4,6)
get_field_slice(asmProp1a.outputs[-1], channel_inds_range=0, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
plt.title("After Propagation")
plt.subplot(2,4,7)
get_field_slice(thinLens1.outputs[-1], channel_inds_range=0, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
plt.title("Lens Output")
plt.subplot(2,4,8)
get_field_slice(asmProp1b.outputs[-1], channel_inds_range=0, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
plt.title("After Propagation")


# plt.imshow(asdf[0,0,0,0,:,:].abs())


pass
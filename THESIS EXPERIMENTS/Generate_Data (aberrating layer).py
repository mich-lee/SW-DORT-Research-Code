import numpy as np
import torch
import matplotlib.pyplot as plt

import warnings
import sys
import copy
import datetime
import os

sys.path.append(os.path.dirname(__file__) + "\\..\\")
sys.path.append(os.path.dirname(__file__) + "\\..\\holotorch-lib\\")
sys.path.append(os.path.dirname(__file__) + "\\..\\holotorch-lib\\holotorch")

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
from holotorch.Optical_Components.Field_Mirrorer import Field_Mirrorer
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


syntheticWavelength = 0.025*mm # 0.05*mm
lambda1 = 1350*nm
lambda2 = lambda1 * syntheticWavelength / (syntheticWavelength - lambda1)

wavelengths = [lambda1, lambda2]

inputRes = (512, 512)
inputSpacing = 6.4*um

intermediateRes = (4096, 4096)
intermediateSpacing = 3.2*um

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

printSimulationSize(inputRes, inputSpacing, 'Simulation Input\t|\t')
printSimulationSize(intermediateRes, intermediateSpacing, 'Intermediate Calcs\t|\t')
printSimulationSize(outputRes, outputSpacing, 'Simulation Output\t|\t')
print()


################################################################################################################################

do_ffts_inplace = True

scattererList = [
					Scatterer(location_x=-0.4*mm, location_y=-0.3*mm, diameter=0.01*mm, scatteringResponse=0.73),
					Scatterer(location_x=-0.3*mm, location_y=0.4*mm, diameter=0.015*mm, scatteringResponse=0.82),
					Scatterer(location_x=0.35*mm, location_y=0*mm, diameter=0.015*mm, scatteringResponse=0.78),
				]

inputResampler = Field_Resampler(outputHeight=intermediateRes[0], outputWidth=intermediateRes[1], outputPixel_dx=intermediateSpacing, outputPixel_dy=intermediateSpacing, device=device)
scattererModel = ScattererModel(scattererList)
memoryReclaimer = Memory_Reclaimer(device=device, clear_cuda_cache=True, collect_garbage=True,
										print_cleaning_actions=False, print_memory_status=False, print_memory_status_printType=2)
outputResampler = Field_Resampler(outputHeight=outputRes[0], outputWidth=outputRes[1], outputPixel_dx=outputSpacing, outputPixel_dy=outputSpacing, device=device)

screenDist = 5*mm #5*mm #3*mm #0.5*mm

# The "surfaceVariationStdDev" and "correlationLength" values came from the 400SA diffuser in Table 4 of
#	"Optical quality of the eye lens surfaces from roughness and diffusion measurements" by Navarro et al.
wavefrontAberratorGen = RandomThicknessScreenGenerator(	surfaceVariationStdDev = 1.3*um,
														correlationLength = 8.8*um,
														maxThickness = 200*um,
														n_screen = 1.52,
														generateBidirectional = True,
														doubleSidedRoughness = True,
														resolution = intermediateRes,
														elementSpacings = [intermediateSpacing, intermediateSpacing],
														device = device
													)
wavefrontAberrator = wavefrontAberratorGen.get_model()
wavefrontAberratorReverse = wavefrontAberratorGen.get_model_reversed()

f1 = 50*mm
d1a = 50*mm
d1b = 50*mm




thinLens1 = Thin_Lens(focal_length=f1)
asmProp1 = ASM_Prop(init_distance=d1a, do_ffts_inplace=do_ffts_inplace)
asmProp2_no_aberrator = ASM_Prop(init_distance=d1b, do_ffts_inplace=do_ffts_inplace)
asmProp2 = ASM_Prop(init_distance=(d1b - screenDist - wavefrontAberratorGen.maxThickness), do_ffts_inplace=do_ffts_inplace)
asmProp3 = ASM_Prop(init_distance=screenDist, do_ffts_inplace=do_ffts_inplace)
model = torch.nn.Sequential	(
								inputResampler,
								asmProp1,
								Radial_Optical_Aperture(aperture_radius=5*mm),
								thinLens1,
								# asmProp2_no_aberrator,
								asmProp2,
								wavefrontAberrator,
								asmProp3,
								scattererModel,
								asmProp3,
								wavefrontAberratorReverse,
								asmProp2,
								# asmProp2_no_aberrator,
								thinLens1,
								Radial_Optical_Aperture(aperture_radius=5*mm),
								asmProp1,
								Field_Mirrorer(mirror_horizontal=True, mirror_vertical=False),	# For modeling the mirroring applied by a beamsplitter
								Ideal_Imaging_Lens(focal_length=10*mm, object_dist=250*mm, interpolationMode='bicubic', rescaleCoords=True, device=device),
								outputResampler
							)

################################################################################################################################

transferMtxMeasurer = TransferMatrixProcessor(	inputFieldPrototype=fieldIn,
												inputBoolMask=inputBoolMask,
												outputBoolMask=outputBoolMask,
												model=model,
												numParallelColumns=3)
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
		torch.save(experimentSaveDict, experimentSaveFileStr)
		print("Saved '" + experimentSaveFileStr + "' to current working directory.")
		break
	elif (resp == 'n'):
		print("Exiting...")
		break
	else:
		print("Invalid input.")


################################################################################################################################

pass
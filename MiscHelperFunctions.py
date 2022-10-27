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


################################################################################################################################


def addSequentialModelOutputHooks(model : torch.nn.Sequential):
	for comp in model:
		comp.add_output_hook()


# Since the models are not egregiously long, writing this method inefficiently.
def getSequentialModelOutputSequence(model : torch.nn.Sequential):
	def getObjectIndInList(l, obj):
		for i in range(len(l)):
			if (l[i] is obj):
				return i
		return -1

	components = []
	for comp in model:
		if (getObjectIndInList(components, comp) == -1):
			components.append(comp)
	
	outputs = []
	curOutputInds = [-1] * len(components)
	for i in range(len(model)-1, -1, -1):
		comp = model[i]
		compInd = getObjectIndInList(components, comp)
		outputs.append(comp.outputs[curOutputInds[compInd]])
		curOutputInds[compInd] -= 1
	outputs.reverse()

	return outputs
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


def plotModelOutputSequence(outputs : list,
							inputField : ElectricField = None,
							maxNumColsPerFigure = 4,
							batch_inds_range : int = None,
							time_inds_range : int = None,
							pupil_inds_range : int = None,
							channel_inds_range : int = None,
							height_inds_range : int = None,
							width_inds_range : int = None,
							figureStartNumber : int = 1,
							flag_colorbar: bool = True,
							flag_axis: bool = True,
							rescale_factor : float = 1,
							plot_xlims : tuple = None,
							plot_ylims : tuple = None,
						):
	if (inputField is not None):
		fields = [1] * (len(outputs) + 1)
		fields[0] = inputField
		fields[1:] = outputs
	else:
		fields = outputs

	numCols = np.minimum(len(fields), maxNumColsPerFigure)
	for i in range(len(fields)):
		curSubplotColInd = (i % numCols) + 1
		if curSubplotColInd == 1:
			plt.figure((i // numCols) + figureStartNumber)
			plt.clf()
		tempOutput = get_field_slice(fields[i], batch_inds_range=batch_inds_range, time_inds_range=time_inds_range, pupil_inds_range=pupil_inds_range,
										channel_inds_range=channel_inds_range, height_inds_range=height_inds_range, width_inds_range=width_inds_range)
		if ((i == 0) and (inputField is not None)):
			titleStr = "Input Field"
		else:
			if (inputField is not None):
				outputNumStr = str(i)
			else:
				outputNumStr = str(i + 1)
			titleStr = "Output " + outputNumStr
		plt.subplot(2, numCols, curSubplotColInd)
		tempOutput.visualize(plot_type=ENUM_PLOT_TYPE.MAGNITUDE, title=titleStr+" (Magnitude)", rescale_factor=rescale_factor, flag_colorbar=flag_colorbar, flag_axis=flag_axis)
		if plot_xlims is not None:
			plt.xlim(plot_xlims)
		if plot_ylims is not None:
			plt.ylim(plot_ylims)
		plt.subplot(2, numCols, curSubplotColInd + numCols)
		tempOutput.visualize(plot_type=ENUM_PLOT_TYPE.PHASE, title=titleStr+" (Phase)", rescale_factor=rescale_factor, flag_colorbar=flag_colorbar, flag_axis=flag_axis)
		if plot_xlims is not None:
			plt.xlim(plot_xlims)
		if plot_ylims is not None:
			plt.ylim(plot_ylims)
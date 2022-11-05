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
from holotorch.Miscellaneous_Components.Memory_Reclaimer import Memory_Reclaimer
from WavefrontAberrator import WavefrontAberrator

from holotorch.utils.Field_Utils import get_field_slice, applyFilterSpaceDomain


################################################################################################################################


def getSequentialModelComponentSequence(model : torch.nn.Sequential, recursive : bool = False):
	compList = []
	for comp in model:
		if (type(comp) is Memory_Reclaimer):
			continue
		if (recursive) and (type(comp) is torch.nn.Sequential):
			compList = compList + getSequentialModelComponentSequence(comp, recursive)
		elif (recursive) and (type(comp) is WavefrontAberrator):
			compList = compList + getSequentialModelComponentSequence(comp.model, recursive)
		else:
			compList = compList + [comp]
	return compList


def addSequentialModelOutputHooks(model : torch.nn.Sequential, recursive : bool = False):
	compList = getSequentialModelComponentSequence(model=model, recursive=recursive)
	for comp in compList:
		comp.add_output_hook()


# Since the models are not egregiously long, writing this method inefficiently.
def getSequentialModelOutputSequence(model : torch.nn.Sequential, recursive : bool = False):
	def getObjectIndInList(l, obj):
		for i in range(len(l)):
			if (l[i] is obj):
				return i
		return -1

	compList = getSequentialModelComponentSequence(model=model, recursive=recursive)
	
	uniqueComponents = []
	for comp in compList:
		if (getObjectIndInList(uniqueComponents, comp) == -1):
			uniqueComponents.append(comp)
	
	outputs = []
	curOutputInds = [-1] * len(uniqueComponents)
	for i in range(len(compList)-1, -1, -1):
		comp = compList[i]
		compInd = getObjectIndInList(uniqueComponents, comp)
		outputs.append(comp.outputs[curOutputInds[compInd]])
		curOutputInds[compInd] -= 1
	outputs.reverse()

	return outputs


def plotModelOutputSequence(outputs : list,
							inputField : ElectricField = None,
							maxNumColsPerFigure = 4,
							componentSequenceList : list = None,
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
		fields = [None] * (len(outputs) + 1)
		fields[0] = inputField
		fields[1:] = outputs
	else:
		fields = outputs

	if (componentSequenceList is not None):
		if (len(componentSequenceList) != len(outputs)):
			raise Exception("'componentSequenceList' has a different number of elements than 'output'.")
		components = [None] * len(componentSequenceList)
		components[:] = componentSequenceList
		if (inputField is not None):
			components = [None] + components
	else:
		components = [None] * len(fields)

	figList = []
	numCols = np.minimum(len(fields), maxNumColsPerFigure)
	for i in range(len(fields)):
		curSubplotColInd = (i % numCols) + 1
		if curSubplotColInd == 1:
			curFig = plt.figure((i // numCols) + figureStartNumber)
			figList = figList + [curFig]
			plt.clf()
		tempOutput = get_field_slice(fields[i], batch_inds_range=batch_inds_range, time_inds_range=time_inds_range, pupil_inds_range=pupil_inds_range,
										channel_inds_range=channel_inds_range, height_inds_range=height_inds_range, width_inds_range=width_inds_range)
		if ((i == 0) and (inputField is not None)):
			titleStr = "[ Input Field"
			titleStrMag = titleStr + " | Magnitude ]"
			titleStrPhase = titleStr + " | Phase ]"
		else:
			if (inputField is not None):
				outputNumStr = str(i)
			else:
				outputNumStr = str(i + 1)
			titleStrL1A = "[ Output " + outputNumStr
			if (components[i] is not None):
				titleStrL2A = "\n" + str(type(components[i]))[8:-2].split('.')[-1]
			else:
				titleStrL2A = ""
			titleStrMag = titleStrL1A + " | Magnitude ]" + titleStrL2A
			titleStrPhase = titleStrL1A + " | Phase ]" + titleStrL2A
		plt.subplot(2, numCols, curSubplotColInd)
		tempOutput.visualize(plot_type=ENUM_PLOT_TYPE.MAGNITUDE, title=titleStrMag, rescale_factor=rescale_factor, flag_colorbar=flag_colorbar, flag_axis=flag_axis)
		if plot_xlims is not None:
			plt.xlim(plot_xlims)
		if plot_ylims is not None:
			plt.ylim(plot_ylims)
		plt.subplot(2, numCols, curSubplotColInd + numCols)
		tempOutput.visualize(plot_type=ENUM_PLOT_TYPE.PHASE, title=titleStrPhase, rescale_factor=rescale_factor, flag_colorbar=flag_colorbar, flag_axis=flag_axis)
		if plot_xlims is not None:
			plt.xlim(plot_xlims)
		if plot_ylims is not None:
			plt.ylim(plot_ylims)

	for curFig in figList:
		plt.figure(curFig.number)
		plt.subplots_adjust(
								left=0.1,
								bottom=0.1,
								right=0.9,
								top=0.9,
								wspace=0.4,
								hspace=0.4
							)
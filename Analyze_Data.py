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

use_cuda = True
gpu_no = 0
device = torch.device("cuda:"+str(gpu_no) if use_cuda else "cpu")

################################################################################################################################


# data = torch.load('Experiment_202210201534.pt', map_location=device)
loadedData = torch.load('Experiment_202210192329.pt', map_location=device)
H = loadedData['Transfer_Matrix']
U, S, Vh = torch.linalg.svd(H)
V = Vh.conj().transpose(-2, -1)

# model = data['Model']
# fieldInputPadder = model[0]
# thinLens1 = model[1][1]
# asmProp1a = model[1][0]
# scattererModel = model[2]
# fieldIn = copy.deepcopy(data['Field_Input_Prototype'])
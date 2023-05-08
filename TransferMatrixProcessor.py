import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt

import copy
import time

import holotorch.utils.Dimensions as Dimensions
from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.Spectra.WavelengthContainer import WavelengthContainer
import holotorch.utils.Memory_Utils as Memory_Utils
from holotorch.utils.Field_Utils import applyFilterSpaceDomain


class TransferMatrixProcessor:
	def __init__(	self,
					inputFieldPrototype : ElectricField,
					inputBoolMask : torch.Tensor,
					outputBoolMask : torch.Tensor,
					model : torch.nn.Module,
					numParallelColumns: int = 8,
					useMacropixels : bool = True
				):
		self.inputBoolMask = inputBoolMask
		self.outputBoolMask = outputBoolMask
		self.model = model
		self.numParallelColumns = numParallelColumns
		self.H_mtx = torch.tensor([])
		self._inputFieldShape = list(inputFieldPrototype.shape)
		self.initializeTempField(inputFieldPrototype)

		self.useMacropixels = useMacropixels

		# This has been replaced by the _calculateMacropixelParameters(...) method:
			# self.inputMacropixelResolution = [int((torch.sum(inputBoolMask, 1) > 0).sum()), int((torch.sum(inputBoolMask, 0) > 0).sum())]
			# self.outputMacropixelResolution = [int((torch.sum(outputBoolMask, 1) > 0).sum()), int((torch.sum(outputBoolMask, 0) > 0).sum())]
			# self.inputMacropixelSize = [inputBoolMask.shape[-2] // self.inputMacropixelResolution[0], inputBoolMask.shape[-1] // self.inputMacropixelResolution[1]]
			# self.outputMacropixelSize = [outputBoolMask.shape[-2] // self.outputMacropixelResolution[0], outputBoolMask.shape[-1] // self.outputMacropixelResolution[1]]

		self.inputMacropixelResolution, self.inputMacropixelSize = TransferMatrixProcessor._calculateMacropixelParameters(inputBoolMask)
		self.outputMacropixelResolution, self.outputMacropixelSize = TransferMatrixProcessor._calculateMacropixelParameters(outputBoolMask)


	def initializeTempField(self, inputFieldPrototype : ElectricField):
		numParallelColumns = self.numParallelColumns

		fieldTensorShape = []
		parallelDim = -1
		for i in range(len(inputFieldPrototype.data.shape) - 2):
			if inputFieldPrototype.shape[i] == 1:
				parallelDim = i
				fieldTensorShape = fieldTensorShape + [numParallelColumns] + list(inputFieldPrototype.data.shape[(parallelDim+1):])
				break
			else:
				fieldTensorShape = fieldTensorShape + [inputFieldPrototype.shape[i]]
				if (i == (len(inputFieldPrototype.data.shape) - 2 - 1)):
					fieldTensorShape = fieldTensorShape + list(self._tempField.data.shape[-2:])

		if (parallelDim != -1):
			columnStep = numParallelColumns
		else:
			# columnStep = 1
			raise Exception("Could not find any available singleton dimensions to calculate responses in parallel.")

		# if (numParallelColumns > 1) and (parallelDim == -1):
		# 	warnings.warn("Specified more than one column to be computed in parallel but was unable to find an available dimension on the tensor.  Computing one column at a time.")
		
		self._fieldTensorShape = fieldTensorShape
		self.parallelDim = parallelDim
		self._columnStep = columnStep

		tempFieldData = torch.zeros(fieldTensorShape, device=inputFieldPrototype.data.device) + 0j
		tempWavelength = copy.deepcopy(inputFieldPrototype.wavelengths)
		tempSpacing = copy.deepcopy(inputFieldPrototype.spacing)
		self._tempField = ElectricField(data = tempFieldData, wavelengths=tempWavelength, spacing=tempSpacing)


	def measureTransferMatrix(self):
		numParallelColumns = self.numParallelColumns
		fieldTensorShape = self._fieldTensorShape
		parallelDim = self.parallelDim
		columnStep = self._columnStep

		heightIndGrid, widthIndGrid = torch.meshgrid(torch.tensor(range(self._tempField.data.shape[-2])), torch.tensor(range(self._tempField.data.shape[-1])))
		heightInds = heightIndGrid[... , self.inputBoolMask]
		widthInds = widthIndGrid[... , self.inputBoolMask]

		matrixInputLen = len(heightInds)
		matrixOutputLen = int((self.outputBoolMask == True).sum())

		H_mtx = torch.zeros(list(self._inputFieldShape[0:-2]) + [matrixOutputLen, matrixInputLen], device=self._tempField.data.device) + 0j

		t0 = time.time()
		t_last = t0
		times = []
		nTimeAvg = 8

		for i in range(0, matrixInputLen, columnStep):
			t1 = time.time()
			times = times + [t1 - t_last]
			t_last = t1
			elapsedTime = t1 - t0

			if (len(times) != 1):
				t_n = times[max(1, len(times) - nTimeAvg):]
				rate = (len(t_n) * columnStep) / sum(t_n)
				est_time_completion = (matrixInputLen - i) / rate

			printStr = f"|  Progress: {(i/len(heightInds))*100:>6.2f}% ({str(i):>4}/{str(len(heightInds)):<4} columns)"
			printStr += f"  |  ETC: "
			if (len(times) == 1):
				printStr += f"{'N/A':8}"
			else:
				printStr += f"{int(est_time_completion // 3600):02}:{int((est_time_completion % 3600) // 60):02}:{int(est_time_completion % 60):02}"
			printStr += f"  |  Runtime: "
			printStr += f"{int(elapsedTime // 3600):02}:{int((elapsedTime % 3600) // 60):02}:{int(elapsedTime % 60):02}"
			printStr += f"  |  Rate: "
			if (len(times) == 1):
				printStr += f"{'N/A':29}"
			else:
				printStr += f"{rate:>5.2f} col/sec ({1/rate:>5.2f} sec/col)"
			printStr += f"  |"
			if (self._tempField.data.device.type == 'cuda'):
				memStats = Memory_Utils.get_cuda_memory_stats(self._tempField.data.device)
				allocCur = memStats['allocated_gb']
				allocMax = memStats['allocated_max_gb']
				reservedCur = memStats['reserved_gb']
				reservedMax = memStats['reserved_max_gb']
				totalMem = memStats['total_gb']
				printStr += f"  GPU Mem: {{Cur: {allocCur:.2f}/{reservedCur:.2f}/{totalMem:.2f} GiB, Max: {allocMax:.2f}/{reservedMax:.2f}/{totalMem:.2f} GiB}}  |"
			print(printStr)

			# print("|    Progress: %.3f%%  " % ((i/len(heightInds))*100), end="")
			# print("(" + str(i) + "/" + str(len(heightInds)) + " columns)\t|    ", end="")
			# print("ETC: ", end="")
			# if (len(times) == 1):
			# 	print("N/A\t\t|    ", end="")
			# else:
			# 	print("%02d:%02d:%02d\t|    " % (int(est_time_completion // 3600), int((est_time_completion % 3600) // 60), int(est_time_completion % 60)), end="")
			# print("Elapsed time: %02d:%02d:%02d\t|    " % (int(elapsedTime // 3600), int((elapsedTime % 3600) // 60), int(elapsedTime % 60)), end="")
			# print("Rate: ", end="")
			# if (len(times) == 1):
			# 	print("N/A\t\t\t\t\t|", end="")
			# else:
			# 	print("%.3f col/sec  (%.3f sec/col)\t|" % (rate, 1/rate), end="")

			self._tempField.data[... , :, :] = 0

			tempViewPermuteInds = list(range(len(fieldTensorShape) - 2))
			del tempViewPermuteInds[parallelDim]
			tempViewPermuteInds = tempViewPermuteInds + [parallelDim] + list(range((len(fieldTensorShape) - 2), len(fieldTensorShape)))
			tempDataTensor = torch.permute(self._tempField.data, tempViewPermuteInds)

			for j in range(min(columnStep, matrixInputLen - i)):
				# 'tempDataTensor' is a view of self._tempField.data so modifying 'tempDataTensor' will modify self._tempField.data
				if (self.useMacropixels):
					# This has been replaced by the TransferMatrixProcessor.getMacropixelInds(...) method:
						# startIndHeight = int(heightInds[i+j] - np.ceil(self.inputMacropixelSize[0] / 2) + 1)
						# endIndHeight = int(heightInds[i+j] + np.floor(self.inputMacropixelSize[0] / 2))
						# startIndWidth = int(widthInds[i+j] - np.ceil(self.inputMacropixelSize[1] / 2) + 1)
						# endIndWidth = int(widthInds[i+j] + np.floor(self.inputMacropixelSize[1] / 2))

					startIndHeight, endIndHeight, startIndWidth, endIndWidth = TransferMatrixProcessor._getMacropixelInds(self.inputMacropixelSize, heightInds[i+j], widthInds[i+j])
					tempDataTensor[... , j, startIndHeight:(endIndHeight+1), startIndWidth:(endIndWidth+1)] = 1
				else:
					tempDataTensor[... , j, int(heightInds[i+j]), int(widthInds[i+j])] = 1

			fieldOut = self.model(self._tempField)
			tempPts = fieldOut.data[... , self.outputBoolMask]

			tempPermuteInds = list(range(len(tempPts.shape)))
			del tempPermuteInds[parallelDim]
			tempPermuteInds = tempPermuteInds + [parallelDim]
			columnsTemp = torch.unsqueeze(torch.permute(tempPts, tempPermuteInds), parallelDim)
			numColsTemp = min(columnStep, H_mtx.shape[-1] - i)
			H_mtx[..., :, i:(i+numColsTemp)] = columnsTemp[..., 0:numColsTemp]

		self.H_mtx = H_mtx
		return H_mtx


	@classmethod
	def getUniformSampleBoolMask(cls, height, width, nx, ny, samplingType : str = 'block'):
		# Notes on 'samplingType' argument:
		# 	-	If samplingType is 'point', then sample to get the tightest fit for the collection of points.
		# 	-	If samplingType is 'block', then sample to get the tightest fit for a collection of blocks.
		if (samplingType != 'point') and (samplingType != 'block'):
			raise Exception("Invalid arguments.  'samplingType' must be either 'point' or 'block'.")
		if (nx > height) or (ny > width):
			raise Exception("Invalid arguments.  'nx' must be less than 'height' and 'ny' must be less than 'width'.")
		
		boolMask = torch.zeros(height, width) != 0	# Initializes everything to False
		
		if (height == 0) or (width == 0) or (nx == 0) or (ny == 0):
			return boolMask
		
		if (nx == 1):
			xStep = 1
			xUpper = 1
		else:
			if (samplingType == 'point'):
				xStep = np.maximum((height-1)//(nx-1), 1)
			else:	# <--- samplingType == 'block'
				xStep = np.maximum(height//nx, 1)
			xUpper = ((nx - 1) * xStep) + 1		# Adding 1 because the upper index in Python indexing is non-inclusive
		
		if (ny == 1):
			yStep = 1
			yUpper = 1
		else:
			if (samplingType == 'point'):
				yStep = np.maximum((width-1)//(ny-1), 1)
			else:	# <--- samplingType == 'block'
				yStep = np.maximum(width//ny, 1)
			yUpper = ((ny - 1) * yStep) + 1
			
		boolMask[0:xUpper:xStep, 0:yUpper:yStep] = True
		rollShiftX = (height - xUpper) // 2
		rollShiftY = (width - yUpper) // 2
		boolMask = torch.roll(boolMask, shifts=(rollShiftX,rollShiftY), dims=(0,1))
		return boolMask


	# This helps facilitate adding up pixel field values over each macropixel.  One can use the returned 'inds' to index the height and width 
	# dimensions of a field tensor.
	# Specifically, one can call something akin to:
	#		field_data[heightInds2,widthInds2].view({dimensions of field_data minus the last one), macropixelSize[0]*macropixelSize[1]).sum(-1)
	# to get the sum of pixel field values over each macropixel.
	@classmethod
	def _getMacropixelIndsFromBoolMask(cls, boolMask : torch.Tensor, macropixelSize : list or tuple, device : torch.device = None):
		heightIndGrid, widthIndGrid = torch.meshgrid(torch.tensor(range(boolMask.shape[-2]), device=device), torch.tensor(range(boolMask.shape[-1]), device=device))
		heightInds = heightIndGrid[... , boolMask]
		widthInds = widthIndGrid[... , boolMask]

		numPixelsPerInd = macropixelSize[0] * macropixelSize[1]

		# Intentionally coupling this to the _getMacroPixelInds method so that stuff has to stay consistent
		hStartOffset, hEndOffset, wStartOffset, wEndOffset = TransferMatrixProcessor._getMacropixelInds(macropixelSize, 0, 0)

		hOffsetRange = torch.arange(hStartOffset, hEndOffset+1, device=device).view(-1, 1).repeat(1, macropixelSize[1]).view(1, numPixelsPerInd)
		wOffsetRange = torch.arange(wStartOffset, wEndOffset+1, device=device).repeat(macropixelSize[0])
		
		heightInds2 = (heightInds.view(-1, 1) + hOffsetRange).view(len(heightInds) * numPixelsPerInd).to(dtype=torch.long)
		widthInds2 = (widthInds.view(-1, 1) + wOffsetRange).view(len(widthInds) * numPixelsPerInd).to(dtype=torch.long)

		return heightInds2, widthInds2

	
	# NOTE: Use endIndHeight+1 and endIndWidth+1 as the end indices when slicing an array or tensor
	@classmethod
	def _getMacropixelInds(cls, macropixelSize : list or tuple, heightCenterInd : int, widthCenterInd : int):
		startIndHeight = int(heightCenterInd - np.ceil(macropixelSize[0] / 2) + 1)
		endIndHeight = int(heightCenterInd + np.floor(macropixelSize[0] / 2))
		startIndWidth = int(widthCenterInd - np.ceil(macropixelSize[1] / 2) + 1)
		endIndWidth = int(widthCenterInd + np.floor(macropixelSize[1] / 2))
		
		return startIndHeight, endIndHeight, startIndWidth, endIndWidth


	@classmethod
	def _getMacropixelConvolutionMask(cls, macropixelSize : list or tuple, height : int, width : int, device : torch.device = None):
		heightCenterInd = np.ceil(height / 2)
		widthCenterInd = np.ceil(width / 2)
		startIndHeight, endIndHeight, startIndWidth, endIndWidth = TransferMatrixProcessor._getMacropixelInds(macropixelSize, heightCenterInd, widthCenterInd)

		if (device is None):
			macropixelMask = torch.zeros(height, width)
		else:
			macropixelMask = torch.zeros(height, width, device=device)
		macropixelMask[startIndHeight:(endIndHeight+1), startIndWidth:(endIndWidth+1)] = 1

		return macropixelMask


	@classmethod
	def _calculateMacropixelParameters(cls, samplingBoolMask : torch.Tensor):
		macropixelResolution = [int((torch.sum(samplingBoolMask, 1) > 0).sum()), int((torch.sum(samplingBoolMask, 0) > 0).sum())]
		macropixelSize = [samplingBoolMask.shape[-2] // macropixelResolution[0], samplingBoolMask.shape[-1] // macropixelResolution[1]]
		return macropixelResolution, macropixelSize


	@classmethod
	def _getModelInput_arg_checker(cls, macropixelVector : torch.Tensor, samplingBoolMask : torch.Tensor):
		if (len(samplingBoolMask.shape) < 2):
			raise Exception("Invalid shape for 'samplingBoolMask'.  Must have at least two dimensions.")
		if (len(samplingBoolMask.shape) > 2):
			raise Exception("Non-2D tensors are not supported for 'samplingBoolMask'.")


	@classmethod
	def getModelInputField(cls, macropixelVector : torch.Tensor, samplingBoolMask : torch.Tensor, fieldPrototype : ElectricField):
		if (samplingBoolMask.shape[-2:] != fieldPrototype.data.shape[-2:]):
			raise Exception("Size mismatch between 'samplingBoolMask' and 'fieldPrototype.data'.")

		fieldIn = copy.deepcopy(fieldPrototype)
		fieldIn.data[...] = 0

		if ((len(fieldIn.data.shape) - 1) != len(macropixelVector.shape)):
			numExtraDims = len(fieldIn.data.shape) - (len(macropixelVector.shape) + 1)
			# Not checking for cases where the non-extra dimensions are mismatched.
			if (numExtraDims < 0):
				raise Exception("Too many dimensions in 'macropixelVector'.")
			if (fieldIn.data.shape[0:numExtraDims] != torch.Size([1] * numExtraDims)):
				raise Exception("Could not expand 'macropixelVector'.")
			newShape = ([1] * numExtraDims) + ([-1] * len(macropixelVector.shape))
			macropixelVector = macropixelVector.expand(newShape)

		# _, macropixelSize = TransferMatrixProcessor._calculateMacropixelParameters(samplingBoolMask)
		# macropixelConvolutionMask = TransferMatrixProcessor._getMacropixelConvolutionMask(macropixelSize=macropixelSize, height=samplingBoolMask.shape[-2], width=samplingBoolMask.shape[-1], device=fieldIn.data.device)
		# fieldIn.data[..., samplingBoolMask] = macropixelVector
		# fieldIn.data[...] = applyFilterSpaceDomain(macropixelConvolutionMask, fieldIn.data)

		fieldIn.data[...] = TransferMatrixProcessor.getModelInputTensor(macropixelVector, samplingBoolMask)
		return fieldIn


	@classmethod
	def getModelInputTensor(cls, macropixelVector : torch.Tensor, samplingBoolMask : torch.Tensor):
		TransferMatrixProcessor._getModelInput_arg_checker(macropixelVector, samplingBoolMask)

		_, macropixelSize = TransferMatrixProcessor._calculateMacropixelParameters(samplingBoolMask)
		macropixelConvolutionMask = TransferMatrixProcessor._getMacropixelConvolutionMask(macropixelSize=macropixelSize, height=samplingBoolMask.shape[-2], width=samplingBoolMask.shape[-1], device=macropixelVector.device)
		
		# samplingBoolMask should be 2D because _getModelInput_arg_checker enforces that.
		inputTensor = torch.zeros(macropixelVector.shape[:-1] + samplingBoolMask.shape[-2:], device=macropixelVector.device, dtype=macropixelVector.dtype)
		inputTensor[..., samplingBoolMask] = macropixelVector
		inputTensor[...] = applyFilterSpaceDomain(macropixelConvolutionMask, inputTensor)
		
		return inputTensor


	# Purpose:
	#	Demixes the eigenstructure.
	#
	# IMPORTANT NOTES:
	#	-	Eigenstructure mixing is a potential problem that one can encounter when doing ultra-wideband DORT where one's targets
	#		are resonant.
	#	-	Eigenstructure mixing can also be an issue for the narrowband, closely spaced wavelengths we are probably working with here.
	#	=	THIS CODE IS FLAWED.  See the comment in this method's code that says "IMPORTANT NOTE" for more information.
	#
	# Explanation:
	#	When doing DORT (i.e. taking the SVD of the transfer matrix) over multiple wavelengths/frequencies, it is not guaranteed
	#	that the i-th singular vector for each wavelength/frequency will correspond to the same scatterer.  This is problematic when trying
	#	to use multiple wavelengths/frequencies.  However, for nearby wavelengths/frequencies, singular vectors belonging to the same scatterer
	#	will be highly correlated (inner product between such vectors will be close to 1).  Thus, by looking at correlations/inner products,
	#	one can ascertain which singular vectors belong to which scatterers.
	#
	# FOR MORE INFORMATION:
	#	For more information on eigenstructure demixing, one can consult Dr. Sun K. Hong's work on the subject.
	#	The relevant paper is titled "Effects of target resonances on ultrawideband-DORT".
	#
	# 'U', 'S', and 'V' are the U, S, and V matrices from SVD respectively.
	# The columns of 'U' and 'V' are the left- and right-singular vectors respectively
	# 'matrixToDemix' decides whether the left- or right-singular vectors will be correlated/inner-producted together
	#	matrixToDemix = 'U' --> demix with the left-singular vectors
	#	matrixToDemix = 'V' --> demix with the right-singular vectors
	@classmethod
	def demixEigenstructure(cls, U : torch.Tensor, S : torch.Tensor, V : torch.Tensor, matrixToDemix : str = 'V', singValMagnitudeSimilarityThreshold : float = None):
		print("")
		print("")
		print("IMPORTANT WARNING/NOTE:")
		print("Eigenstructure demixing is being performed.  However, the current eigenstructure demixing code is flawed.  See the source code notes above the eigenstructure demixing method ('demixEigenstructure' in the TransferMatrixProcessor class) for more information.")
		print("")

		# Input checking
		if (U.shape[0:4] != V.shape[0:4]):
			raise Exception("'U' and 'V' matrices have incompatible dimensions.  They should both be the same size in the B, T, P, and C dimensions.")
		if (U.shape[-2] != U.shape[-1]):
			raise Exception("'U' should be a square matrix.")
		if (V.shape[-2] != V.shape[-1]):
			raise Exception("'V' should be a square matrix.")
		if (U.shape[-1] != S.shape[-1]) and (V.shape[-2] != S.shape[-1]):
			raise Exception("Number of singular values does not match the dimensions of 'U' and/or 'V'.")
		if (matrixToDemix == 'U'):
			singVecs = U
		elif (matrixToDemix == 'V'):
			singVecs = V
		else:
			raise Exception("Invalid option for 'matrixToDemix'.  Should either be 'U' or 'V'.")
		
		# Getting wavelength dimension index
		wavelengthDimIndex = torch.arange(4)[torch.tensor(singVecs.shape[0:4]) != 1]
		if (len(wavelengthDimIndex) == 0):
			raise Exception("Error: there only appears to be one wavelength.  No need to demix a single wavelength.")
		elif (len(wavelengthDimIndex) != 1):
			# This will catch any case where there is more than one non-singleton dimension among the first four dimensions of singVecs.
			#	That is good.
			raise Exception("The case where wavelengths vary along more than one dimension is not implemented.")	# In light of the above comment, this error message could be more descriptive.
		if (singVecs.shape[wavelengthDimIndex] != 2):
			raise Exception("Demixing more than two wavelengths has not been implemented.")
		
		# 'singVecs' contains the singular vector matrices that will be used for demixing
		singVecs = singVecs.squeeze()	# Should be an N x height x width tensor (where N is the number of wavelengths)
		tempS = S.squeeze()

		numSingVals = S.shape[-1]

		# Cross-correlations/inner products
		corrs = torch.matmul(singVecs.conj().transpose(-2,-1)[1,:,:], singVecs[0,:,:]).abs()

		if (singValMagnitudeSimilarityThreshold is not None):
			# Prevents the demixed singular values from differing too much in magnitude
			for i in range(numSingVals):
				curSingVal = tempS[0,i]
				corrs[:,i] = corrs[:,i] * (tempS[1,:] >= ((1 - singValMagnitudeSimilarityThreshold) * curSingVal)) * (tempS[1,:] <= ((1+singValMagnitudeSimilarityThreshold) * curSingVal))

		# IMPORTANT WARNING: This can allow for multiple of the same singular vector to be assigned to the same singular value
		_, secondWavelengthInds = torch.max(corrs, 0)

		# Demixing
		newU = U.clone()
		newS = S.clone()
		newV = V.clone()
		for i in range(numSingVals):
			newU[... , 1, :, i] = U[..., 1, :, secondWavelengthInds[i]]
			newS[... , 1, i] = S[..., 1, secondWavelengthInds[i]]
			newV[... , 1, :, i] = V[..., 1, :, secondWavelengthInds[i]]

		return newU, newS, newV
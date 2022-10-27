import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt

import copy
import time

from holotorch.CGH_Datatypes.ElectricField import ElectricField


class TransferMatrixProcessor:
	def __init__(	self,
					inputFieldPrototype : ElectricField,
					inputBoolMask : torch.tensor,
					outputBoolMask : torch.tensor,
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
		self.inputMacropixelResolution = [int((torch.sum(inputBoolMask, 1) > 0).sum()), int((torch.sum(inputBoolMask, 0) > 0).sum())]
		self.outputMacropixelResolution = [int((torch.sum(outputBoolMask, 1) > 0).sum()), int((torch.sum(outputBoolMask, 0) > 0).sum())]
		self.inputMacropixelSize = [inputBoolMask.shape[0] // self.inputMacropixelResolution[0], inputBoolMask.shape[1] // self.inputMacropixelResolution[1]]
		self.outputMacropixelSize = [outputBoolMask.shape[0] // self.outputMacropixelResolution[0], outputBoolMask.shape[1] // self.outputMacropixelResolution[1]]

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
			raise Exception("Could not find any available singleton dimensions to calculate responses in paralle.")

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

		for i in range(0, matrixInputLen, columnStep):
			t1 = time.time()
			times = times + [t1 - t_last]
			t_last = t1
			print("Columns processed: " + str(i) + "/" + str(len(heightInds)) + "\t|\tElapsed time: %.3fs\t|\tETC: " % (t1 - t0), end="")
			if (len(times) == 1):
				print("N/A")
			else:
				t_n = times[max(1, len(times) - 8):]
				rate = (len(t_n) * columnStep) / sum(t_n)
				est_time_completion = (matrixInputLen - i) / rate
				print("%02d:%02d:%02d" % (int(est_time_completion // 3600), int((est_time_completion % 3600) // 60), int(est_time_completion % 60)))

			self._tempField.data[... , :, :] = 0

			tempViewPermuteInds = list(range(len(fieldTensorShape) - 2))
			del tempViewPermuteInds[parallelDim]
			tempViewPermuteInds = tempViewPermuteInds + [parallelDim] + list(range((len(fieldTensorShape) - 2), len(fieldTensorShape)))
			tempDataTensor = torch.permute(self._tempField.data, tempViewPermuteInds)

			for j in range(min(columnStep, matrixInputLen - i)):
				# 'tempDataTensor' is a view of self._tempField.data so modifying 'tempDataTensor' will modify self._tempField.data
				if (self.useMacropixels):
					startIndHeight = int(heightInds[i+j] - np.ceil(self.inputMacropixelSize[0] / 2) + 1)
					endIndHeight = int(heightInds[i+j] + np.floor(self.inputMacropixelSize[0] / 2))
					startIndWidth = int(widthInds[i+j] - np.ceil(self.inputMacropixelSize[1] / 2) + 1)
					endIndWidth = int(widthInds[i+j] + np.floor(self.inputMacropixelSize[1] / 2))
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
	def getUniformSampleBoolMask(cls, height, width, nx, ny):
		if (nx > height) or (ny > width):
			raise Exception("Invalid arguments.  'nx' must be less than 'height' and 'ny' must be less than 'width'.")
		boolMask = torch.zeros(height, width) != 0	# Initializes everything to False
		if (height == 0) or (width == 0) or (nx == 0) or (ny == 0):
			return boolMask
		if (nx == 1):
			xStep = 1
			xUpper = 1
		else:
			xStep = np.maximum((height-1)//(nx-1), 1)
			xUpper = ((nx - 1) * xStep) + 1
		if (ny == 1):
			yStep = 1
			yUpper = 1
		else:
			yStep = np.maximum((width-1)//(ny-1), 1)
			yUpper = ((ny - 1) * yStep) + 1
		boolMask[0:xUpper:xStep, 0:yUpper:yStep] = True
		rollShiftX = (height - xUpper) // 2
		rollShiftY = (width - yUpper) // 2
		boolMask = torch.roll(boolMask, shifts=(rollShiftX,rollShiftY), dims=(0,1))
		return boolMask
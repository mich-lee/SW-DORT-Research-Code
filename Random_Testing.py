import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
import time
# import warnings
# import copy
# import datetime
# import os

sys.path.append("holotorch-lib/")
sys.path.append("holotorch-lib/holotorch")

import holotorch.utils.Dimensions as Dimensions
# from holotorch.utils.Enumerators import *
# from holotorch.utils.units import * # E.g. to get nm, um, mm etc.
# from holotorch.Spectra.WavelengthContainer import WavelengthContainer
from holotorch.Spectra.SpacingContainer import SpacingContainer
# from holotorch.CGH_Datatypes.ElectricField import ElectricField
# # from holotorch.Optical_Propagators.ASM_Prop import ASM_Prop
# # from holotorch.Optical_Components.Thin_Lens import Thin_Lens

from holotorch.utils.Helper_Functions import ft2, ift2, fft2_inplace, ifft2_inplace
# from holotorch.utils.Field_Utils import get_field_slice
# from MiscHelperFunctions import getSequentialModelComponentSequence, addSequentialModelOutputHooks, getSequentialModelOutputSequence, plotModelOutputSequence

import holotorch.utils.Memory_Utils as Memory_Utils

################################################################################################################################

use_cuda = True
gpu_no = 0
device = torch.device("cuda:"+str(gpu_no) if use_cuda else "cpu")

################################################################################################################################

# Testing some methods
	# d0 = Dimensions.BTPCHW(n_batch=1, n_time=1, n_pupil=1, n_channel=1, height=1, width=1)
	# d1 = Dimensions.BTPCHW(n_batch=1, n_time=4, n_pupil=3, n_channel=1, height=5, width=1)
	# d2 = Dimensions.BTPCHW(n_batch=5, n_time=4, n_pupil=1, n_channel=2, height=5, width=1)
	# d3 = Dimensions.BTPCHW(n_batch=5, n_time=3, n_pupil=1, n_channel=2, height=5, width=1)
	# d4 = Dimensions.TCD(n_time=4, n_channel=2, height=2)
	# d5 = Dimensions.C(n_channel=2)
	# s0 = SpacingContainer(spacing=53)
	# s1 = SpacingContainer(spacing=53)
	# s2 = SpacingContainer(spacing=123)
	# print(d0.has_compatible_shape(d0))
	# print(d0.has_compatible_shape(d1))
	# print(d0.has_compatible_shape(d2))
	# print(d0.has_compatible_shape(d3))
	# print(d0.has_compatible_shape(d4))
	# print(d0.has_compatible_shape(d5))
	# print(s0.is_equivalent(s0))
	# print(s0.is_equivalent(s1))
	# print(s0.is_equivalent(s2))
	# print(s1.is_equivalent(s1))
	# print(s1.is_equivalent(s2))
	# print(s2.is_equivalent(s2))

# Testing FFT shift and row-column 2D DFT algorithm stuff
	# a = torch.rand(5, 17) + 0j
	# # a = torch.ones(7, 17) + 0j

	# a2 = torch.zeros_like(a)
	# for r in range(a.shape[-2]):
	# 	a2[r,:] = torch.fft.fftshift(torch.fft.fft(torch.fft.fftshift(a[r,:], dim=-1), norm='backward'), dim=-1)
	# for c in range(a.shape[-1]):
	# 	a2[:,c] = torch.fft.fftshift(torch.fft.fft(torch.fft.fftshift(a2[:,c], dim=-1), norm='backward'), dim=-1)

	# # for r in range(a.shape[-2]):
	# # 	a2[r,:] = torch.fft.fft(a[r,:], norm='backward')
	# # for c in range(a.shape[-1]):
	# # 	a2[:,c] = torch.fft.fft(a2[:,c])

	# a3 = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(a, dim=(-2,-1)), norm='backward'), dim=(-2,-1))

# Testing FFT memory usage
	# Memory_Utils.print_cuda_memory_usage(device=device)
	# a = torch.rand(3036, 4024, dtype=torch.complex64, device=device)
	# Memory_Utils.print_cuda_memory_usage(device=device)
	# a = torch.fft.fft2(a)
	# Memory_Utils.print_cuda_memory_usage(device=device)

# Testing in-place 2D FFT
	# # x = torch.arange(1, 4*8 + 1).view(4,8) + 0j
	# x = torch.rand(3, 1, 2, 5, 4, 8, dtype=torch.complex64, device=device)
	# x_orig = x.clone()
	# fft2_inplace(x, centerOrigins=False, norm='ortho')
	# x2 = torch.fft.fft2(x_orig, norm='ortho')
	# print(x)
	# print(x2)
	# print(torch.equal(x, x2))

# Testing memory stuff
	# Memory_Utils.print_cuda_memory_usage(device=device)
	# a = torch.rand(4, 1, 1, 2, 5*3036, 5*3036, dtype=torch.complex64, device=device)
	# Memory_Utils.print_cuda_memory_usage(device=device)
	# # a = torch.fft.fft2(a)
	# fft2_inplace(a)
	# Memory_Utils.print_cuda_memory_usage(device=device)




# Miscellaneous
	# t0 = time.process_time()
	# testInplaceFFTCorrectness(device=device, inverse_fft_flag=False, print_mem_usage=False)
	# elapsed_time = time.process_time() - t0
	# print(elapsed_time)

	# Memory_Utils.print_cuda_memory_usage(device=device)
	# a = torch.rand(4, 1, 1, 2, 2*3036, 2*3036, dtype=torch.complex64, device=device)
	# Memory_Utils.print_cuda_memory_usage(device=device)
	# # a = torch.fft.fft2(a)
	# fft2_inplace(a)
	# Memory_Utils.print_cuda_memory_usage(device=device)

	# print("Start:")
	# Memory_Utils.print_cuda_memory_usage(device=device, printType=2)
	# print()
	# x = torch.rand(2*3036*2*3036, dtype=torch.complex64, device=device)
	# print("After initializing x:")
	# Memory_Utils.print_cuda_memory_usage(device=device, printType=2)
	# print()
	# x = torch.fft.fftshift(x, dim=-1)
	# print("After first fftshift:")
	# Memory_Utils.print_cuda_memory_usage(device=device, printType=2)
	# print()
	# x = torch.fft.fft(x, dim=-1, norm='ortho')
	# print("After first fft:")
	# Memory_Utils.print_cuda_memory_usage(device=device, printType=2)
	# print()
	# x = torch.fft.fftshift(x, dim=-1)
	# print("After second fftshift")
	# Memory_Utils.print_cuda_memory_usage(device=device, printType=2)
	# print()

################################################################################################################################

# Testing to see if in-place FFT results match the results from other fft2 implementations
def testInplaceFFTCorrectness(device : torch.device, norm = 'backward', inverse_fft_flag = False, holotorch_fft_flag = False, do_print = True, short_print = False, print_mem_usage = True):
	centerOrigins = holotorch_fft_flag
	for i in range(10):
		a = torch.rand(np.random.randint(4)+1, np.random.randint(2)+1, np.random.randint(3)+1, np.random.randint(3)+1, np.random.randint(3036)+1, np.random.randint(3036)+1, dtype=torch.complex64, device=device)
		a_orig = a.clone()

		if not inverse_fft_flag:
			fft2_inplace(a, centerOrigins=centerOrigins, norm=norm)
			if holotorch_fft_flag:
				a2 = ft2(a_orig, norm=norm)
			else:
				a2 = torch.fft.fft2(a_orig, norm=norm)
		else:
			ifft2_inplace(a, centerOrigins=centerOrigins, norm=norm)
			if holotorch_fft_flag:
				a2 = ift2(a_orig, norm=norm)
			else:
				a2 = torch.fft.ifft2(a_orig, norm=norm)
		
		err = a - a2
		
		if do_print:
			printTuple = (	float(a.abs().mean()), float(a.abs().max()), float((a.abs()**2).mean().sqrt()),
							float(err.abs().mean()), float(err.abs().max()), float((err.abs()**2).mean().sqrt()),
							float((a.abs()**2).mean().sqrt() / (err.abs()**2).mean().sqrt())
						)
			if not short_print:
				printStr = "\tMean Abs\tMax Abs\t\tRMS\na\t%f\t%f\t%f\tsize: " + str(a.shape) + "\nerr\t%f\t%f\t%f\trms ratio: %f\n"
			else:
				printStr = "%f\t%f\t%f"
				printTuple = (printTuple[2], printTuple[5], printTuple[6])
			print(printStr % printTuple)
			if not short_print:
				print()
		
		randInds = [np.random.randint(a.shape[0]), np.random.randint(a.shape[1]), np.random.randint(a.shape[2]), np.random.randint(a.shape[3])]
		a = a[randInds[0], randInds[1], randInds[2], randInds[3], :, :]
		if not inverse_fft_flag:
			if holotorch_fft_flag:
				a3 = ft2(a_orig[randInds[0], randInds[1], randInds[2], randInds[3], :, :], norm=norm)
			else:
				a3 = torch.fft.fft2(a_orig[randInds[0], randInds[1], randInds[2], randInds[3], :, :], norm=norm)
		else:
			if holotorch_fft_flag:
				a3 = ift2(a_orig[randInds[0], randInds[1], randInds[2], randInds[3], :, :], norm=norm)
			else:
				a3 = torch.fft.ifft2(a_orig[randInds[0], randInds[1], randInds[2], randInds[3], :, :], norm=norm)

		err = a - a3

		if do_print:
			printStr = "\tMean Abs\tMax Abs\t\tRMS\na\t%f\t%f\t%f\tsize: " + str(a.shape) + "\nerr\t%f\t%f\t%f\trms ratio: %f\n"
			printTuple = (	float(a.abs().mean()), float(a.abs().max()), float((a.abs()**2).mean().sqrt()),
							float(err.abs().mean()), float(err.abs().max()), float((err.abs()**2).mean().sqrt()),
							float((a.abs()**2).mean().sqrt() / (err.abs()**2).mean().sqrt())
						)
			if not short_print:
				printStr = "\tMean Abs\tMax Abs\t\tRMS\na\t%f\t%f\t%f\tsize: " + str(a.shape) + "\nerr\t%f\t%f\t%f\trms ratio: %f\n"
			else:
				printStr = "%f\t%f\t%f"
				printTuple = (printTuple[2], printTuple[5], printTuple[6])
			print(printStr % printTuple)
			if not short_print:
				print()

		if print_mem_usage:
			Memory_Utils.print_cuda_memory_usage(device=device, printType=2)

		del a
		del a_orig
		del a2

################################################################################################################################

# a = torch.arange(H*W).view(H,W).expand(3,1,1,2,-1,-1) * ((10**torch.arange(3)).view(-1,1,1,1,1,1))
# a.view(B,T,P,C,H,int(W/wSize),wSize).permute(0,1,2,3,-2,-3,-1).view(3,1,1,2,int(W/wSize),int(H/hSize),hSize,wSize).sum(-2).sum(-1).permute(0,1,2,3,5,4)

hSize, wSize = 6, 7
B,T,P,C,H,W = torch.Size([3,1,1,2,14*hSize,19*wSize])
a = torch.zeros(1,T,P,C,H,W)
for r in range(int(H/hSize)):
	for c in range(int(W/wSize)):
		a[... , r*hSize:(r+1)*hSize, c*wSize:(c+1)*wSize] = (r*int(W/wSize) + c) / (hSize*wSize)
a = (a * (10 ** torch.arange(B).view(B,1,1,1,1,1))) * (10 ** torch.arange(C).view(1,1,1,C,1,1))
aTempView = a.view(B,T,P,C,int(H/hSize),hSize,int(W/wSize),wSize)
aSummed = aTempView.sum(-3).sum(-1)

################################################################################################################################

pass
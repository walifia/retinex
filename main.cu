#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <chrono>
#include "cufft.h"

#include <helper_functions.h>
#include <helper_cuda.h>


#define LOAD_FLOAT(i) tex1Dfetch<float>(texFloat, i)
#define SET_FLOAT_BASE

#define LOG_IN_PLACE 1
#define LOG_OUT_OF_PLACE 0

#ifdef __CUDACC__
typedef float2 fComplex;
#else
typedef struct {
	float x;
	float y;
} fComplex;
#endif

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
// Round a / b to nearest higher integer value
inline int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

// Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b) { return (a % b != 0) ? (a - a % b + b) : a; }

int snapTransformSize(int dataSize) 
{
	int hiBit;
	unsigned int lowPOT, hiPOT;

	dataSize = iAlignUp(dataSize, 16);

	for (hiBit = 31; hiBit >= 0; hiBit--)
		if (dataSize & (1U << hiBit)) 
		{
			break;
		}

	lowPOT = 1U << hiBit;

	if (lowPOT == (unsigned int)dataSize) 
	{
		return dataSize;
	}

	hiPOT = 1U << (hiBit + 1);

	if (hiPOT <= 1024) 
	{
		return hiPOT;
	}
	else 
	{
		return iAlignUp(dataSize, 512);
	}
}

////////////////////////////////////////////////////////////////////////////////
/// Position convolution kernel center at (0, 0) in the image
////////////////////////////////////////////////////////////////////////////////
__global__ void padKernel_kernel(float* d_Dst, float* d_Src, int fftH, int fftW, int kernelH, int kernelW, int kernelY, int kernelX, cudaTextureObject_t texFloat) 
{
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;

	if (y < kernelH && x < kernelW) {
		int ky = y - kernelY;

		if (ky < 0) {
			ky += fftH;
		}

		int kx = x - kernelX;

		if (kx < 0) {
			kx += fftW;
		}

		d_Dst[ky * fftW + kx] = LOAD_FLOAT(y * kernelW + x);
	}
}

void padKernel(float* d_Dst, float* d_Src, int fftH, int fftW, int kernelH, int kernelW, int kernelY, int kernelX)
{
	assert(d_Src != d_Dst);
	dim3 threads(32, 8);
	dim3 grid(iDivUp(kernelW, threads.x), iDivUp(kernelH, threads.y));

	SET_FLOAT_BASE;

	cudaTextureObject_t texFloat;
	cudaResourceDesc texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));

	texRes.resType = cudaResourceTypeLinear;
	texRes.res.linear.devPtr = d_Src;
	texRes.res.linear.sizeInBytes = sizeof(float) * kernelH * kernelW;
	texRes.res.linear.desc = cudaCreateChannelDesc<float>();

	cudaTextureDesc texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = false;
	texDescr.filterMode = cudaFilterModeLinear;
	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.readMode = cudaReadModeElementType;

	cudaCreateTextureObject(&texFloat, &texRes, &texDescr, NULL);

	padKernel_kernel << <grid, threads >> > (d_Dst, d_Src, fftH, fftW, kernelH, kernelW, kernelY, kernelX, texFloat);

	cudaDestroyTextureObject(texFloat);
}

////////////////////////////////////////////////////////////////////////////////
// Prepare data for "pad to border" addressing mode
////////////////////////////////////////////////////////////////////////////////
__global__ void padDataClampToBorder_kernel(float* d_Dst, float* d_Src, int fftH, int fftW, int dataH, int dataW, int kernelH, int kernelW, int kernelY, int kernelX, cudaTextureObject_t texFloat) 
{
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int borderH = dataH + kernelY;
	const int borderW = dataW + kernelX;

	if (y < fftH && x < fftW) {
		int dy, dx;

		if (y < dataH) {
			dy = y;
		}

		if (x < dataW) {
			dx = x;
		}

		if (y >= dataH && y < borderH) {
			dy = dataH - 1;
		}

		if (x >= dataW && x < borderW) {
			dx = dataW - 1;
		}

		if (y >= borderH) {
			dy = 0;
		}

		if (x >= borderW) {
			dx = 0;
		}

		d_Dst[y * fftW + x] = LOAD_FLOAT(dy * dataW + dx);
	}
}

void padDataClampToBorder(float* d_Dst, float* d_Src, int fftH, int fftW, int dataH, int dataW, int kernelW, int kernelH, int kernelY, int kernelX) 
{
	assert(d_Src != d_Dst);
	dim3 threads(32, 8);
	dim3 grid(iDivUp(fftW, threads.x), iDivUp(fftH, threads.y));

	cudaTextureObject_t texFloat;
	cudaResourceDesc texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));

	texRes.resType = cudaResourceTypeLinear;
	texRes.res.linear.devPtr = d_Src;
	texRes.res.linear.sizeInBytes = sizeof(float) * dataH * dataW;
	texRes.res.linear.desc = cudaCreateChannelDesc<float>();

	cudaTextureDesc texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = false;
	texDescr.filterMode = cudaFilterModeLinear;
	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.readMode = cudaReadModeElementType;

	cudaCreateTextureObject(&texFloat, &texRes, &texDescr, NULL);

	padDataClampToBorder_kernel << <grid, threads >> > (d_Dst, d_Src, fftH, fftW, dataH, dataW, kernelH, kernelW, kernelY, kernelX, texFloat);

	cudaDestroyTextureObject(texFloat);
}


////////////////////////////////////////////////////////////////////////////////
// Cut data back to image resolution
////////////////////////////////////////////////////////////////////////////////

__global__ void dePadData_kernel(float* d_Dst, float* d_Src, int dataH, int dataW, int fftH, int fftW, cudaTextureObject_t texFloat)
{
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;

	if (y < dataH && x < dataW)
	{
		const int idx = y * fftW + x;
		d_Dst[y * dataW + x] = LOAD_FLOAT(idx);
	}
}


void dePadData(float* d_Dst, float* d_Src, int dataH, int dataW, int fftH, int fftW)
{
	assert(d_Src != d_Dst);
	dim3 threads(32, 8);
	dim3 grid(iDivUp(dataW, threads.x), iDivUp(dataH, threads.y));

	cudaTextureObject_t texFloat;
	cudaResourceDesc texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));

	texRes.resType = cudaResourceTypeLinear;
	texRes.res.linear.devPtr = d_Src;
	texRes.res.linear.sizeInBytes = sizeof(float) * fftH * fftW;
	texRes.res.linear.desc = cudaCreateChannelDesc<float>();

	cudaTextureDesc texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = false;
	texDescr.filterMode = cudaFilterModeLinear;
	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.readMode = cudaReadModeElementType;

	cudaCreateTextureObject(&texFloat, &texRes, &texDescr, NULL);

	dePadData_kernel << <grid, threads >> > (d_Dst, d_Src, dataH, dataW, fftH, fftW, texFloat);

	cudaDestroyTextureObject(texFloat);
}


////////////////////////////////////////////////////////////////////////////////
// Modulate Fourier image of padded data by Fourier image of padded kernel
// and normalize by FFT size
////////////////////////////////////////////////////////////////////////////////
inline __device__ void mulAndScale(fComplex& a, const fComplex& b, const float& c) 
{
	fComplex t = { c * (a.x * b.x - a.y * b.y), c * (a.y * b.x + a.x * b.y) };
	a = t;
}

__global__ void modulateAndNormalize_kernel(fComplex* d_Dst, const fComplex* d_Src, int dataSize, float c) 
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= dataSize) {
		return;
	}

	fComplex a = d_Src[i];
	fComplex b = d_Dst[i];

	mulAndScale(a, b, c);

	d_Dst[i] = a;
}


void modulateAndNormalize(fComplex* d_Dst, const fComplex* d_Src, int fftH, int fftW, int padding) 
{
	assert(fftW % 2 == 0);
	const int dataSize = fftH * (fftW / 2 + padding);

	modulateAndNormalize_kernel << <iDivUp(dataSize, 256), 256 >> > (d_Dst, d_Src, dataSize, 1.0f / (float)(fftW * fftH));
}




///////////////////////////////////////////////////////////////////////////////
// Retinex algorithm subroutines
///////////////////////////////////////////////////////////////////////////////
__global__ void toLogDomainGPU_kernel(float* d_Dst, float* d_Src, int dataSize)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < dataSize)
	{
		float tmp = log(d_Src[i] > 0 ? d_Src[i] : (d_Src[i] + 1));
		d_Dst[i] = tmp;
	}
}

void toLogDomainGPU(float* d_Dst, float* d_Src, int dataH, int dataW)
{
	const int dataSize = dataH * dataW;
	toLogDomainGPU_kernel << <iDivUp(dataSize, 256), 256 >> > (d_Dst, d_Src, dataSize);
}

__global__ void fromLogDomainGPU_kernel(float* d_Dst, float* d_Src, int dataSize)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < dataSize)
	{
		float tmp = exp(d_Src[i]);
		d_Dst[i] = tmp;
	}
}

void fromLogDomainGPU(float* d_Dst, float* d_Src, int dataH, int dataW)
{
	const int dataSize = dataH * dataW;
	fromLogDomainGPU_kernel << <iDivUp(dataSize, 256), 256 >> > (d_Dst, d_Src, dataSize);
}

__global__ void pixelwiseDifference_kernel(float* d_Dst, float* d_Src1, float* d_Src2, int dataSize)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < dataSize)
		d_Dst[i] = d_Src1[i] - d_Src2[i];
}

void pixelwiseDifference(float* d_Dst, float* d_Src1, float* d_Src2, int dataH, int dataW)
{
	const int dataSize = dataH * dataW;
	pixelwiseDifference_kernel << <iDivUp(dataSize, 256), 256 >> > (d_Dst, d_Src1, d_Src2, dataSize);
}

__global__ void weightedSum_kernel(float* d_Dst, float* d_Src1, float* d_Src2, float* d_Src3, int dataSize)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < dataSize)
		d_Dst[i] = (d_Src1[i] + d_Src2[i] + d_Src3[i]) / 3;
}

void weightedSum(float* d_Dst, float* d_Src1, float* d_Src2, float* d_Src3, int dataH, int dataW)
{
	const int dataSize = dataH * dataW;
	weightedSum_kernel << <iDivUp(dataSize, 256), 256 >> > (d_Dst, d_Src1, d_Src2, d_Src3, dataSize);
}






typedef struct rgb
{
	float r, g, b;
} RGB;

typedef struct kernelInfo
{
	float sd; //standard deviation
	int w; //kernel width
} kerInfo;




///////////////////////////////////////////////////////////////////////////////////////////
// Loading and preprocessing gaussian kernels for a single runtime execution
///////////////////////////////////////////////////////////////////////////////////////////
std::vector<float> generateKernel(kerInfo info)
{
	int w = info.w;
	float sd = info.sd;
	int r = w / 2;
	std::vector<float> kernel(w * w);
	for (int i = -r; i <= r; i++)
		for (int j = -r; j <= r; j++)
		{
			kernel[(i + r) * (2 * r + 1) + j + r] = exp(-(i * i + j * j) / (2 * sd * sd)) / (2 * 3.14159265f * sd * sd);
			//std::cout << ker[(i + r) * (2 * r + 1) + j + r] << ' ';
		}
	return kernel;
}

void loadKernels(const std::vector<kerInfo>& info, std::vector<float*>& d_Kernels)
{	
	d_Kernels.resize(info.size());
	for (int i = 0; i < info.size(); i++)
	{
		std::vector<float> kernel = generateKernel(info[i]);
		cudaMalloc((void**)&d_Kernels[i], kernel.size() * sizeof(float));
		cudaMemcpy(d_Kernels[i], kernel.data(), kernel.size() * sizeof(float), cudaMemcpyHostToDevice);
		std::cout << "Kernel #" << i + 1 << " is initialized!\n";
	}
}

void preprocessKernels(std::vector<float*>& d_Kernels, std::vector<fComplex*>& d_KernelSpectrums, const int dataH, const int dataW, const std::vector<kerInfo>& info, std::vector<std::pair<int, int>>& fftS, std::vector<std::pair<cufftHandle, cufftHandle>>& fftPlans)
{
	fftS.resize(info.size());
	d_KernelSpectrums.resize(info.size());
		
	for (int i = 0; i < info.size(); i++)
	{
		fftS[i] = { snapTransformSize(dataH + info[i].w - 1), snapTransformSize(dataW + info[i].w - 1) };

		float* d_PaddedKernel;

		cudaMalloc((void**)&d_PaddedKernel, fftS[i].first * fftS[i].second * sizeof(float));
		cudaMemset(d_PaddedKernel, 0, fftS[i].first * fftS[i].second * sizeof(float));
		cudaMalloc((void**)&d_KernelSpectrums[i], fftS[i].first * (fftS[i].second / 2 + 1) * sizeof(fComplex));
		cudaMemset(d_KernelSpectrums[i], 0, fftS[i].first * (fftS[i].second / 2 + 1) * sizeof(fComplex));

		padKernel(d_PaddedKernel, d_Kernels[i], fftS[i].first, fftS[i].second, info[i].w, info[i].w, info[i].w / 2, info[i].w / 2);

		cufftPlan2d(&fftPlans[i].first, fftS[i].first, fftS[i].second, CUFFT_R2C);
		cufftPlan2d(&fftPlans[i].second, fftS[i].first, fftS[i].second, CUFFT_C2R);

		cufftExecR2C(fftPlans[i].first, (cufftReal*)d_PaddedKernel, (cufftComplex*)d_KernelSpectrums[i]);

		std::cout << "Kernel #" << i + 1 << " is preprocessed!\n";

		cudaFree(d_PaddedKernel);
		cudaFree(d_Kernels[i]);
	}
}

void unloadKernels(std::vector<fComplex*>& d_KernelSpectrums)
{
	for (int i = 0; i < d_KernelSpectrums.size(); i++)
		cudaFree(d_KernelSpectrums[i]);
}

////////////////////////////////////////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////////////////////////////////////////
void destroyFFTPlans(std::vector<std::pair<cufftHandle, cufftHandle>>& fftPlans)
{
	for (int i = 0; i < fftPlans.size(); i++)
	{
		cufftDestroy(fftPlans[i].first);
		cufftDestroy(fftPlans[i].second);
	}
}

void loadFrame(const float* h_Data, float* d_Data, int dataSize)
{
	cudaMemcpy(d_Data, h_Data, dataSize * sizeof(float), cudaMemcpyHostToDevice);
}

void unloadFrame(float* d_Data, float* h_Data, int dataSize)
{
	cudaMemcpy(h_Data, d_Data, dataSize * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_Data);
}

void intensity2rgb(const RGB* img, const float* ret, const float* origInt, RGB* result, int w, int h)
{
	for (int i = 0; i < w * h; i++)
	{
		if (origInt[i] == 0)
		{
			result[i] = { 0, 0, 0 };
			continue;
		}
		result[i].r = std::max(0.f, std::min(255.f, img[i].r * ret[i] / origInt[i]));
		result[i].g = std::max(0.f, std::min(255.f, img[i].g * ret[i] / origInt[i]));
		result[i].b = std::max(0.f, std::min(255.f, img[i].b * ret[i] / origInt[i]));
	}
}

void fromLogDomain(const float* logImg, float* img, int w, int h)
{
	float minv = 100, maxv = -1;
	for (int i = 0; i < w * h; i++)
	{
		float tmp = logImg[i];
		tmp = exp(tmp);
		img[i] = tmp;
		minv = std::min(minv, tmp);
		maxv = std::max(maxv, tmp);
	}
	for (int i = 0; i < w * h; i++)
		img[i] = round(255 * (img[i] - minv) / (maxv - minv));
}

////////////////////////////////////////////////////////////////////////////////
// Gaussian blur wrapper
////////////////////////////////////////////////////////////////////////////////
void gaussianBlurFFTGPU(float* d_Data, float* d_Out, const int dataW, const int dataH, const fComplex* d_KernelSpectrum, const std::pair<int, int>& fftS, const kerInfo info, const std::pair<cufftHandle, cufftHandle>& fftPlans)
{
	float *d_PaddedData;
	fComplex *d_DataSpectrum;

	
	cudaMalloc((void**)&d_PaddedData, fftS.first * fftS.second * sizeof(float));
	cudaMemset(d_PaddedData, 0, fftS.first * fftS.second * sizeof(float));
	cudaMalloc((void**)&d_DataSpectrum, fftS.first * (fftS.second / 2 + 1) * sizeof(fComplex));


	padDataClampToBorder(d_PaddedData, d_Data, fftS.first, fftS.second, dataH, dataW, info.w, info.w, info.w / 2, info.w / 2);
	
	

	cufftExecR2C(fftPlans.first, (cufftReal*)d_PaddedData, (cufftComplex*)d_DataSpectrum);

	modulateAndNormalize(d_DataSpectrum, d_KernelSpectrum, fftS.first, fftS.second, 1);

	cufftExecC2R(fftPlans.second, (cufftComplex*)d_DataSpectrum, (cufftReal*)d_PaddedData);

	dePadData(d_Out, d_PaddedData, dataH, dataW, fftS.first, fftS.second);

	cudaDeviceSynchronize();


	cudaFree(d_DataSpectrum);
	cudaFree(d_PaddedData);

}

////////////////////////////////////////////////////////////////////////////////
// Main Retinex functions
////////////////////////////////////////////////////////////////////////////////
void singleScaleRetinexGPU(float* d_Data, float* d_Out, const int dataW, const int dataH, const fComplex* d_KernelSpectrum, const std::pair<int, int>& fftS, const kerInfo info, std::pair<cufftHandle, cufftHandle>& fftPlans)
{
	float* d_Blurred, *d_logData, * d_logBlurred, * d_Result;

	cudaMalloc((void**)&d_Blurred, dataW * dataH * sizeof(float));
	cudaMalloc((void**)&d_logData, dataW * dataH * sizeof(float));
	cudaMalloc((void**)&d_logBlurred, dataW * dataH * sizeof(float));



	gaussianBlurFFTGPU(d_Data, d_Blurred, dataW, dataH, d_KernelSpectrum, fftS, info, fftPlans);
	
	
	cudaDeviceSynchronize();





	toLogDomainGPU(d_logData, d_Data, dataH, dataW);
	toLogDomainGPU(d_logBlurred, d_Blurred, dataH, dataW);

	pixelwiseDifference(d_Out, d_logData, d_logBlurred, dataH, dataW);

	

	cudaFree(d_Blurred);
	cudaFree(d_logBlurred);
	cudaFree(d_logData);
}

void multiScaleRetinexGPU(float* d_Data, float* d_Out, const int dataW, const int dataH, std::vector<fComplex*>& d_KernelSpectrums, std::vector<std::pair<int, int>>& fftS, std::vector<kerInfo> info, std::vector<std::pair<cufftHandle, cufftHandle>>& fftPlans)
{
	float* d_Out1, * d_Out2, * d_Out3;

	cudaMalloc((void**)&d_Out1, dataH * dataW * sizeof(float));
	cudaMalloc((void**)&d_Out2, dataH * dataW * sizeof(float));
	cudaMalloc((void**)&d_Out3, dataH * dataW * sizeof(float));




	singleScaleRetinexGPU(d_Data, d_Out1, dataW, dataH, d_KernelSpectrums[0], fftS[0], info[0], fftPlans[0]);
	singleScaleRetinexGPU(d_Data, d_Out2, dataW, dataH, d_KernelSpectrums[1], fftS[1], info[1], fftPlans[1]);
	singleScaleRetinexGPU(d_Data, d_Out3, dataW, dataH, d_KernelSpectrums[2], fftS[2], info[2], fftPlans[2]);
	
	weightedSum(d_Out, d_Out1, d_Out2, d_Out3, dataH, dataW);

	cudaDeviceSynchronize();

	cudaFree(d_Out1);
	cudaFree(d_Out2);
	cudaFree(d_Out3);
}

void MSRGPU(float* h_Data, float* h_Result, int dataH, int dataW, std::vector<fComplex*>& d_KernelSpectrums, std::vector<std::pair<int, int>>& fftS, std::vector<kerInfo> info, std::vector<std::pair<cufftHandle, cufftHandle>>& fftPlans)
{
	float* d_Data, * d_Result;
	float* h_logResult = new float[dataH * dataW];

	cudaMalloc((void**)&d_Data, dataH * dataW * sizeof(float));
	cudaMalloc((void**)&d_Result, dataH * dataW * sizeof(float));

	loadFrame(h_Data, d_Data, dataH * dataW);
	multiScaleRetinexGPU(d_Data, d_Result, dataW, dataH, d_KernelSpectrums, fftS, info, fftPlans);
	unloadFrame(d_Result, h_logResult, dataH * dataW);

	fromLogDomain(h_logResult, h_Result, dataW, dataH);

	cudaFree(d_Data);
	cudaFree(d_Result);
	delete[] h_logResult;
}



//Main provides an example of a single frame execution

int main()
{
	const char* filename = ".png";

	int w, h, n, i;

	unsigned char* data = stbi_load(filename, &w, &h, &n, 3);

	n = 3;
	int imgBytes = w * h * n;
	int imgPix = w * h;

	RGB* img = new RGB[imgPix]();
	float* imgI = new float[imgPix]();


	i = 0;
	for (unsigned char* p = data; p != data + imgBytes; p += n, i++)
	{
		uint8_t r = (uint8_t)(*p);
		uint8_t g = (uint8_t)(*(p + 1));
		uint8_t b = (uint8_t)(*(p + 2));

		img[i] = { (float)r, (float)g, (float)b };
		imgI[i] = ((float)r + (float)g + (float)b) / 3.;
	}



	std::vector<std::pair<cufftHandle, cufftHandle>> fftPlans;
	std::vector<kerInfo> info = { {15, 91}, {80, 481}, { 230, 1381 } };
	fftPlans.resize(info.size());
	std::vector<float*> d_Kernels;
	std::vector<fComplex*> d_KernelSpectrums;
	std::vector<std::pair<int, int>> fftS;

	loadKernels(info, d_Kernels);
	preprocessKernels(d_Kernels, d_KernelSpectrums, h, w, info, fftS, fftPlans);


	float* ret = new float[w * h]();

	while(true)
		MSRGPU(imgI, ret, h, w, d_KernelSpectrums, fftS, info, fftPlans);
	

	destroyFFTPlans(fftPlans);
	

	RGB* result = new RGB[w * h];
	intensity2rgb(img, ret, imgI, result, w, h);


	uint8_t* out = new uint8_t[imgBytes]();

	i = 0;
	for (uint8_t* p = out; p != out + imgBytes; p += n, i++)
	{
		*p = (uint8_t)result[i].r;
		*(p + 1) = (uint8_t)result[i].g;
		*(p + 2) = (uint8_t)result[i].b;
	}


	stbi_write_png(".png", w, h, n, out, w * n);

	stbi_image_free(data);
	delete[] img;
	delete[] ret;
	delete[] imgI;
	delete[] out;
	delete[] result;

	return 0;
}



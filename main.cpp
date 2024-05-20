#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <ctime>
#include <omp.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


typedef struct rgb 
{
	float r, g, b;
} RGB;



///////////////////////////////////////////////////////////////////////////////
// Retinex algorithm subroutines
///////////////////////////////////////////////////////////////////////////////
void toLogDomain(const float* img, float* logImg, int w, int h)
{
	#pragma omp parallel for
	for (int i = 0; i < w * h; i++)
	{
		float tmp = img[i];
		if (tmp == 0)
			tmp += 1;
		tmp = log(tmp);
		logImg[i] = tmp;
	}
}

void fromLogDomain(const float* logImg, float* img, int w, int h)
{
	float minv = 100, maxv = -1;
	#pragma omp parallel for
	for (int i = 0; i < w * h; i++)
	{
		float tmp = logImg[i];
		tmp = exp(tmp);
		img[i] = tmp;
		minv = std::min(minv, tmp);
		maxv = std::max(maxv, tmp);
	}
	#pragma omp parallel for
	for (int i = 0; i < w * h; i++)
	{
		img[i] = round(255 * (img[i] - minv) / (maxv - minv));
		//std::cout << img[i] << ' ';
	}
		
}

void imgSubtraction(const float* img1, const float* img2, float* rr, int w, int h)
{
	for (int i = 0; i < w * h; i++)
		rr[i] = img1[i] - img2[i];
}

///////////////////////////////////////////////////////////////////////////////
// Fast Gaussian blur implementation
///////////////////////////////////////////////////////////////////////////////
void boxBlurH(float* img, float* blurredImg, int w, int h, int r)
{
	float iarr = 1. / (r + r + 1.);

	#pragma omp parallel for
	for (int i = 0; i < h; i++) 
	{
		int ti = i * w, li = ti, ri = ti + r;

		float fv = img[ti], lv = img[ti + w - 1], val = (r + 1) * fv;

		for (int j = 0; j < r; j++) 
			val += img[ti + j];

		for (int j = 0; j <= r; j++) 
		{ 
			val += img[ri++] - fv;   
			blurredImg[ti++] = round(val * iarr); 
		}
		for (int j = r + 1; j < w - r; j++) 
		{ 
			val += img[ri++] - img[li++];   
			blurredImg[ti++] = round(val * iarr); 
		}
		for (int j = w - r; j < w; j++)
		{ 
			val += lv - img[li++];   
			blurredImg[ti++] = round(val * iarr); 
		}
	}
}

void boxBlurT(float* img, float* blurredImg, int w, int h, int r)
{
	int sz = w * h;
	float iarr = 1.f / (r + r + 1.f);
	#pragma omp parallel for
	for (int i = 0; i < w; i++)
	{
		int ti = i, li = ti, ri = ti + r * w;
		if (ri >= sz)
			break;
		//std::cout << ri << ' ';
		float fv = img[ti], lv = img[ti + w * (h - 1)], val = (r + 1) * fv;

		for (int j = 0; j < r; j++) 
			val += img[ti + j * w];
		for (int j = 0; j <= r && ri < sz; j++)
		{ 
			//std::cout << ri << ' ';
			val += img[ri] - fv;
			blurredImg[ti] = round(val * iarr);
			ri += w; 
			ti += w; 
		}
		for (int j = r + 1; j < h - r; j++)
		{ 
			val += img[ri] - img[li];
			blurredImg[ti] = round(val * iarr);
			li += w;
			ri += w;
			ti += w;
		}
		for (int j = h - r; j < h; j++)
		{ 
			val += lv - img[li];
			blurredImg[ti] = round(val * iarr);
			li += w; 
			ti += w; 
		}
	}
}

void boxBlur(float* img, float* blurredImg, int w, int h, int r) 
{
	for (int i = 0; i < w * h; i++) 
		blurredImg[i] = img[i];
	boxBlurH(blurredImg, img, w, h, r);
	boxBlurT(img, blurredImg, w, h, r);
}

std::vector<int> boxesForGauss(float sigma, int n)
{
	float wIdeal = sqrt((12 * sigma * sigma / n) + 1); 
	int wl = floor(wIdeal);  
	if (wl % 2 == 0) 
		wl--;
	int wu = wl + 2;

	float mIdeal = (12 * sigma * sigma - n * wl * wl - 4 * n * wl - 3 * n) / (-4 * wl - 4);
	int m = round(mIdeal);

	// var sigmaActual = Math.sqrt( (m*wl*wl + (n-m)*wu*wu - n)/12 );

	std::vector<int> sizes;

	for (int i = 0; i < n; i++) 
		sizes.push_back(i < m ? wl : wu);

	return sizes;
}

void gaussianBlur_1D(const float* img, float* blurredImg, int w, int h, int r)
{
	float* scl = new float[w * h]();
	memcpy(scl, img, w * h * sizeof(float));

	std::vector<int> bxs = boxesForGauss(r, 3);
	boxBlur(scl, blurredImg, w, h, (bxs[0] - 1) / 2);
	boxBlur(blurredImg, scl, w, h, (bxs[1] - 1) / 2);
	boxBlur(scl, blurredImg, w, h, (bxs[2] - 1) / 2);

	delete[] scl;
}



///////////////////////////////////////////////////////////////////////////////
// Main Retinex functions
///////////////////////////////////////////////////////////////////////////////
void singleScaleRetinex(const float* img, float* result, int w, int h, int sigma)
{
	float* logImg = new float[w * h]();
	float* blurredImg = new float[w * h]();
	float* logBlurredImg = new float[w * h]();

	toLogDomain(img, logImg, w, h);
	gaussianBlur_1D(img, blurredImg, w, h, sigma);
	toLogDomain(blurredImg, logBlurredImg, w, h);

	delete[] blurredImg;

	imgSubtraction(logImg, logBlurredImg, result, w, h);

	delete[] logImg;
	delete[] logBlurredImg;
}

void SSR(const float* img, float* result, int w, int h, int sigma)
{
	float* logResult = new float[w * h]();
	singleScaleRetinex(img, logResult, w, h, sigma);

	fromLogDomain(logResult, result, w, h);

	delete[] logResult;
}

void weightedSum2D(const float* img1, const float* img2, const float* img3, float* result, int w, int h)
{
	#pragma omp parallel for
	for (int i = 0; i < w * h; i++)
		result[i] = (img1[i] + img2[i] + img3[i]) / 3.;
}


void multiScaleRetinex(const float* img, float* result, int w, int h)
{
	float* img1 = new float[w * h]();
	float* img2 = new float[w * h]();
	float* img3 = new float[w * h]();
	
	
	singleScaleRetinex(img, img1, w, h, 15);
	singleScaleRetinex(img, img2, w, h, 80);
	singleScaleRetinex(img, img3, w, h, 230);

	weightedSum2D(img1, img2, img3, result, w, h);

	delete[] img1;
	delete[] img2;
	delete[] img3;
}

void MSR(const float* img, float* result, int w, int h)
{
	float* logResult = new float[w * h]();
	multiScaleRetinex(img, logResult, w, h);

	fromLogDomain(logResult, result, w, h);

	delete[] logResult;
}

///////////////////////////////////////////////////////////////////////////////
// Utility functions
///////////////////////////////////////////////////////////////////////////////
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



int main()
{
	float LUT[256];

	for (int i = 1; i < 256; i++)
		LUT[i] = std::log(i);

	std::string pathdir = "img/4K/";
	std::string pathdirout = "img/out/4K/";


	for (int j = 1; j <= 485; j++)
	{
		std::string pathimg = pathdir + "i (" + std::to_string(j) + ").png";
		std::string pathsave = pathdirout + "MSRInt" + std::to_string(j) + ".png";

		int x, y, n, i;

		unsigned char* data = stbi_load(pathimg.c_str(), &x, &y, &n, 3);

		n = 3;
		int img_size = x * y * n;

		RGB* img = new RGB[x * y]();
		float* imgI = new float[x * y]();

		i = 0;
		for (unsigned char* p = data; p != data + img_size; p += n, i++)
		{
			uint8_t r = (uint8_t)(*p);
			uint8_t g = (uint8_t)(*(p + 1));
			uint8_t b = (uint8_t)(*(p + 2));

			img[i] = { (float)r, (float)g, (float)b };
			imgI[i] = ((float)r + (float)g + (float)b) / 3.;
		}


		unsigned int dummy;
		unsigned long long start = __rdtscp(&dummy);
		std::clock_t c_start = std::clock();

		float* ret = new float[x * y]();

		MSR(imgI, ret, x, y);


		RGB* result = new RGB[x * y]();
		intensity2rgb(img, ret, imgI, result, x, y);


		unsigned long long end = __rdtscp(&dummy);
		std::clock_t c_end = std::clock();	
		std::cout << (int)((end - start) / (5 * 1e6)) << std::endl;




		uint8_t* out = new uint8_t[img_size]();

		i = 0;
		for (uint8_t* p = out; p != out + img_size; p += n, i++)
		{
			*p = (uint8_t)result[i].r;
			*(p + 1) = (uint8_t)result[i].g;
			*(p + 2) = (uint8_t)result[i].b;
		}


		stbi_write_png(pathsave.c_str(), x, y, n, out, x * n);

		stbi_image_free(data);
		delete[] img;
		delete[] imgI;
		delete[] out;
		delete[] result;
		delete[] ret;
	}

	return 0;
}
#ifndef INCLUDE_GPUSLIC_H
#define INCLUDE_GPUSLIC_H

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <limits.h>
#include "slic.h"
#include "common.h"
#include "bitmap/bitmap_image.hpp"

typedef unsigned int uint;

void gpuSLIC(const char *input, const char *output, int sud);
void outputSlic(int bordi, int centri, pixel *pi, center *ck, int K, const char *output, int width, int height);
float getGpuMill();
float getGpuAdvMill();
__global__ void slic_core(center *c, pixel *p);
__global__ void slic_adv_core(center *c, pixel *p);
__device__ double distance(center *c, pel ri, pel gi, pel bi, int xi, int yi);
__global__ void initCluster(center *c, pixel *p, int sudd);
__device__ void GPUlowestGradient(int cx, int cy, pixel *p, int *nx, int *ny);
__global__ void initPixel(center *c, pixel *p, int sudd);
#endif
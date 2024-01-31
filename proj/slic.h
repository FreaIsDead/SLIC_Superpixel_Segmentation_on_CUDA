#ifndef INCLUDE_SLIC_H
#define INCLUDE_SLIC_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <vector>
#include "bitmap/bitmap_image.hpp"

typedef unsigned char pel;

typedef struct
{
  pel r;
  pel g;
  pel b;
  int x;
  int y;
} center;

typedef struct
{
  float r;
  float g;
  float b;
  float x;
  float y;
} counterCenter;

typedef struct
{
  pel r;
  pel g;
  pel b;
  int x;
  int y;
  int cluster;
  int distance;
} pixel;

/**
 * Compute SLIC on CPU
 *
 * @param input Input image name (or path);
 * @param sud Horizontal subdivisions of the uniform grid;
 */
void cpuSLIC(const char *input, int sud);
/**
 * Output SLIC result on given file name
 * 
 * @param bordi Border drawing if 1, uniform color region drawing if 0;
 * @param centri If 1 draw clusters center, if 0 don't;
 * @param output Output image name (or path);
 */
void outputSlic(int bordi, int centri, const char *output); //bordi [0,1], centri [0,1]
double ds(center *c, pel ri, pel gi, pel bi, int xi, int yi);
void lowestGradient(int cx, int cy, bitmap_image *image, int *nx, int *ny);
float getCpuMil();

#endif

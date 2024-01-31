#include "gpuslic.h"
#define blksize 256

//VARIABILI GLOBALI
__constant__ int _N = 0;
__constant__ int _K = 0;
__constant__ int _S = 0;
__constant__ int _ngbSize = 0;
__constant__ int _m = 10;
__constant__ int _h;
__constant__ int _w;

float gpumil;
float gpuAdvmil;

void gpuSLIC(const char * input,const char * output, int sud) {
  bitmap_image image(input);
  if (!image)
  {
    printf("Error - Failed to open input_Image\n");
    return;
  }
  char out[50];
  strcpy(out, output);
  char temp[50];
  
  int height = image.height();
  int width = image.width();
  int N = width * height;
  int sudd = sud;
  float step = round(((float)width) / sudd);
  float rows = round(((float)height) / step);
  int K = (sudd) * (rows);
  int S = sqrt(((float)N) / K);
  int ngbSize = 2 * S;

  CHECK(cudaMemcpyToSymbol(_h, &height, sizeof(int)));
  CHECK(cudaMemcpyToSymbol(_w, &width, sizeof(int)));
  CHECK(cudaMemcpyToSymbol(_ngbSize, &ngbSize, sizeof(int)));
  CHECK(cudaMemcpyToSymbol(_S, &S, sizeof(int)));
  CHECK(cudaMemcpyToSymbol(_K, &K, sizeof(int)));
  CHECK(cudaMemcpyToSymbol(_N, &N, sizeof(int)));

  //IMPLEMENTAZIONE 1 - SLIC CORE
  //init centri e pixels
  center *ck = (center *)malloc(sizeof(center) * K);
  pixel *pi = (pixel *)malloc(sizeof(pixel) * height * width);
  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      rgb_t colour;
      image.get_pixel(x, y, colour);
      pi[y*width + x].r = colour.red;
      pi[y*width + x].g = colour.green;
      pi[y*width + x].b = colour.blue;
      pi[y*width + x].x = x;
      pi[y*width + x].y = y;
      pi[y*width + x].cluster = -1;
      pi[y*width + x].distance = INT_MAX;
    }
  }
  int nx = 0;
  int ny = 0;
  int xs = S / 2;
  int ys = S / 2;
  rgb_t colour;
  for (int i = 0; i < K; i++)
  {
    lowestGradient(xs, ys, &image, &nx, &ny);
    image.get_pixel(nx, ny, colour);
    ck[i].x = nx;
    ck[i].y = ny;
    ck[i].r = colour.red;
    ck[i].g = colour.green;
    ck[i].b = colour.blue;
    if ((xs + S) > (width))
    {
      ys = (ys + S);
      xs = S / 2;
    }
    else
    {
      xs = (xs + S);
    }
  }
  //centri e pixel gpu
  center *gc;
  pixel *gp;
	CHECK(cudaMalloc((void**) &gc, sizeof(center) * K));
	CHECK(cudaMalloc((void**) &gp, sizeof(pixel) * height * width));
  CHECK(cudaMemcpy(gc, ck, sizeof(center) * K, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(gp, pi, sizeof(pixel) * height * width, cudaMemcpyHostToDevice));
  printf("\nGPUSLIC\nwidth: %d, height: %d\n", width, height);
  printf("N: %d, K: %d, S: %d\n\n", N, K, S);

  /*Creazione di eventi per performance metric*/
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  //lancio kernels 
  for(int i = 0; i < 10; i++) {
    slic_core<<<1,K>>>(gc,gp);
    CHECK(cudaDeviceSynchronize());
  }
  CHECK(cudaMemcpy(ck, gc, sizeof(center) * K, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(pi, gp, sizeof(pixel) * height * width, cudaMemcpyDeviceToHost));
  
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gpumil, start, stop);
  printf("GPUSLIC milliseconds: %f\n", gpumil);

  //OUTPUT result
  strcpy(temp, "0_");
  strcat(temp, out);
  strcpy(out,temp);
  outputSlic(0,1,pi,ck, K, out, width, height);
  

  //IMPLEMENTAZIONE 2 - SLIC CORE ADVANCED
  printf("\nGPUSLIC_ADVANCED\nwidth: %d, height: %d\n", width, height);
  printf("N: %d, K: %d, S: %d\n\n", N, K, S);
  ck = (center *)malloc(sizeof(center) * K);
  pi = (pixel *)malloc(sizeof(pixel) * height * width);
  //init pixel CPU
  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      rgb_t colour;
      image.get_pixel(x, y, colour);
      pi[y*width + x].r = colour.red;
      pi[y*width + x].g = colour.green;
      pi[y*width + x].b = colour.blue;
      pi[y*width + x].x = x;
      pi[y*width + x].y = y;
      pi[y*width + x].cluster = -1;
      pi[y*width + x].distance = INT_MAX;
    }
  }
  /*Creazione di eventi per performance metric*/
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

	CHECK(cudaMalloc((void**) &gc, sizeof(center) * K));
	CHECK(cudaMalloc((void**) &gp, sizeof(pixel) * height * width));
  CHECK(cudaMemcpy(gc, ck, sizeof(center) * K, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(gp, pi, sizeof(pixel) * height * width, cudaMemcpyHostToDevice));

  uint clustGrid  = (K + blksize - 1) / blksize;
  initCluster<<<clustGrid,blksize>>>(gc, gp, sudd);
  CHECK(cudaDeviceSynchronize());

  uint pixGrid = ((width*height + blksize - 1) / blksize);
  initPixel<<<pixGrid,blksize>>>(gc, gp, sudd);
  CHECK(cudaDeviceSynchronize()); 

  for(int i = 0; i < 10; i++) {
    initPixel<<<pixGrid,blksize>>>(gc,gp, sudd);
    CHECK(cudaDeviceSynchronize());
    //update each cluster center
    slic_adv_core<<<clustGrid,blksize>>>(gc,gp);
    CHECK(cudaDeviceSynchronize());
  }

  CHECK(cudaMemcpy(ck, gc, sizeof(center) * K, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(pi, gp, sizeof(pixel) * height * width, cudaMemcpyDeviceToHost));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gpuAdvmil, start, stop);
  printf("GPUSLIC_ADVANCED milliseconds: %f\n", gpuAdvmil);

  //OUTPUT result
  strcpy(out, output);
  strcpy(temp, "1_");
  strcat(temp, out);
  strcpy(out,temp);
  outputSlic(0,1,pi,ck, K, out, width, height);

  //Free resources
  CHECK(cudaFree(gp));
  CHECK(cudaFree(gc));
  free(ck);
  free(pi);
}

__global__ void slic_adv_core(center *c, pixel *p){
  uint i = blockIdx.x *blockDim.x + threadIdx.x;
  if(i < _K) {
    int cx;
    int cy;
    int startx;
    int starty;
    cx = c[i].x;
    cy = c[i].y;
    startx = cx - _S;
    starty = cy - _S;
    long counters;
    counterCenter centerAcc;
    counters = 0;
    centerAcc.r = 0;
    centerAcc.g = 0;
    centerAcc.b = 0;
    centerAcc.x = 0;
    centerAcc.y = 0;
    for (int y = starty; y < starty + _ngbSize; y++)
    {
      for (int x = startx; x < startx + _ngbSize; x++)
      {
        if (x >= 0 && x < _w && y >= 0 && y < _h)
        {
          pixel pix = p[y*_w + x];
          int cluster = pix.cluster;
          if (counters == 0 && cluster == i)
          {
            centerAcc.r += pix.r;
            centerAcc.g += pix.g;
            centerAcc.b += pix.b;
            centerAcc.x += pix.x;
            centerAcc.y += pix.y;
            counters ++;
          }
          else if (cluster == i)
          {
            centerAcc.r += (pix.r - centerAcc.r) * (1.0 / (counters + 1));
            centerAcc.g += (pix.g - centerAcc.g) * (1.0 / (counters + 1));
            centerAcc.b += (pix.b - centerAcc.b) * (1.0 / (counters + 1));
            centerAcc.x += (pix.x - centerAcc.x) * (1.0 / (counters + 1));
            centerAcc.y += (pix.y - centerAcc.y) * (1.0 / (counters + 1));
            counters ++;
          }
        }
      }
    }
    c[i].r = (pel)centerAcc.r;
    c[i].g = (pel)centerAcc.g;
    c[i].b = (pel)centerAcc.b;
    c[i].x = (int)centerAcc.x;
    c[i].y = (int)centerAcc.y;
  }
}


__global__ void slic_core(center *c,pixel *p) {
  uint i = blockIdx.x *blockDim.x + threadIdx.x;
  if(i < _K) {
    int cx;
    int cy;
    int startx;
    int starty;
    double tempD;
    cx = c[i].x;
    cy = c[i].y;
    startx = cx - _S;
    starty = cy - _S;
  
    for (int y = starty; y < starty + _ngbSize; y++)
    {
      for (int x = startx; x < startx + _ngbSize; x++)
      {
        if (x >= 0 && x < _w && y >= 0 && y < _h)
        {
          tempD = distance(&c[i], p[y*_w+ x].r, p[y*_w+ x].g, p[y*_w+ x].b, x, y);
          if (tempD < p[y*_w + x].distance)
          {
            __syncthreads();
            p[y*_w+ x].distance = tempD;
            p[y*_w+ x].cluster = i;
          }
        }
      }
    }
    long counters;
    counterCenter centerAcc;
    counters = 0;
    centerAcc.r = 0;
    centerAcc.g = 0;
    centerAcc.b = 0;
    centerAcc.x = 0;
    centerAcc.y = 0;
    for (int y = starty; y < starty + _ngbSize; y++)
    {
      for (int x = startx; x < startx + _ngbSize; x++)
      {
        if (x >= 0 && x < _w && y >= 0 && y < _h)
        {
          if (counters == 0 && p[y*_w + x].cluster == i)
          {
            centerAcc.r += p[y*_w + x].r;
            centerAcc.g += p[y*_w + x].g;
            centerAcc.b += p[y*_w + x].b;
            centerAcc.x += p[y*_w + x].x;
            centerAcc.y += p[y*_w + x].y;
            counters ++;
          }
          else if (p[y*_w + x].cluster == i)
          {
            centerAcc.r += (p[y*_w + x].r - centerAcc.r) * (1.0 / (counters + 1));
            centerAcc.g += (p[y*_w + x].g - centerAcc.g) * (1.0 / (counters + 1));
            centerAcc.b += (p[y*_w + x].b - centerAcc.b) * (1.0 / (counters + 1));
            centerAcc.x += (p[y*_w + x].x - centerAcc.x) * (1.0 / (counters + 1));
            centerAcc.y += (p[y*_w + x].y - centerAcc.y) * (1.0 / (counters + 1));
            counters ++;
          }
        }
      }
    }
    c[i].r = (pel)centerAcc.r;
    c[i].g = (pel)centerAcc.g;
    c[i].b = (pel)centerAcc.b;
    c[i].x = (int)centerAcc.x;
    c[i].y = (int)centerAcc.y;
  }
}

__global__ void initPixel(center *c, pixel *p, int sudd) {
  uint i = blockIdx.x *blockDim.x + threadIdx.x;
  if(i < _h * _w) {
    pixel pix = p[i];
    int x = pix.x;
    int y = pix.y;
    pel r = pix.r;
    pel g = pix.g;
    pel b = pix.b;
    int column = x / _S;
    int row = y / _S;
    double min_distance = DBL_MAX;
    double temp;
    int Winner_cluster;
    int targetClusters[9] = {
      (row*sudd+ column)%_K,
      ((row +1)*sudd+ column)%_K, 
      ((row -1)*sudd+ column)%_K, 
      (row*sudd+ column+1)%_K,
      (row*sudd+ column-1)%_K,
      ((row +1)*sudd+ column+1)%_K,
      ((row -1)*sudd+ column+1)%_K,
      ((row +1)*sudd+ column-1)%_K,
      ((row -1)*sudd+ column-1)%_K
    };
    for(int j = 0; j < 9; j++) {
      temp = distance(&c[targetClusters[j]],r,g,b,x,y);
      if(min_distance > temp){
        min_distance = temp;
        Winner_cluster = targetClusters[j];
      }
    }
    p[i].cluster = Winner_cluster;
    p[i].distance = min_distance;
  }
}

__global__ void initCluster(center *c, pixel*p, int sudd) {
  uint i = blockIdx.x *blockDim.x + threadIdx.x;
  if(i < _K) {
    int x = _S/2;
    int y = _S/2;
    int column = i % sudd;
    int row = i / sudd;
    x= (x + (column*_S)) % _w;
    y= (y + (row*_S)) % _h;
    int nx;
    int ny;
    GPUlowestGradient(x,y,p,&nx,&ny);
    x = nx;
    y = ny;
    c[i].x = x;
    c[i].y = y;
    c[i].r = p[y*_w+ x].r;
    c[i].g = p[y*_w+ x].g;
    c[i].b = p[y*_w+ x].b;
  }
}

__device__ void GPUlowestGradient(int cx, int cy, pixel *p, int *nx, int *ny)
{
  double min_grad = DBL_MAX;
  double red_g = 0;
  double green_g = 0;
  double blue_g = 0;
  double magn = 0;
  int local_minx = 0;
  int local_miny = 0;
  pixel p1,p2,p3,p4;
  int width = _w;
  int height = _h;
  for (int x = cx - 1; x < cx + 2; x++)
  {
    for (int y = cy - 1; y < cy + 2; y++)
    {
      if (x >= 0 && x < width && y >= 0 && y < height)
      {
        p1 = p[y*_w + x+1];
        p2 = p[y*_w + x-1];
        p3 = p[(y+1)*_w + x];
        p3 = p[(y-1)*_w + x];
        red_g = sqrtf(powf(p1.r - p2.r, 2)) + sqrtf(powf(p3.r - p4.r, 2));
        green_g = sqrtf(powf(p1.g - p2.g, 2)) + sqrtf(powf(p3.g - p4.g, 2));
        blue_g = sqrtf(powf(p1.b - p2.b, 2)) + sqrtf(powf(p3.b - p4.b, 2));
        magn = sqrtf(powf(red_g, 2) + powf(green_g, 2) + powf(blue_g, 2));
        if (min_grad > magn)
        {
          min_grad = magn;
          local_minx = x;
          local_miny = y;
        }
      }
    }
  }
  *nx = local_minx;
  *ny = local_miny;
}

__device__ double distance(center *c, pel ri, pel gi, pel bi, int xi, int yi)
{
  double rs = powf(c->r - ri, 2);
  double gs = powf(c->g - gi, 2);
  double bs = powf(c->b - bi, 2);
  double dRGB = sqrtf(rs + gs + bs);

  double xs = powf(c->x - xi, 2);
  double ys = powf(c->y - yi, 2);
  double dxy = sqrtf(xs + ys);

  return dRGB + (_m / _S) * dxy;
}

void outputSlic(int bordi, int centri, pixel *pi, center *ck, int K,const char *output, int width, int height)
{
  //SCRITTURA BMP usando i PIXEL struct
  bitmap_image result(width, height);
  result.clear();
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      result.set_pixel(x, y, pi[y*width + x].r, pi[y*width + x].g, pi[y*width + x].b);
    }
  }
  //Disegno i bordi dei cluster
  if (bordi)
  {
    int cluster = 0;
    int upp = 0;
    int lowp = 0;
    int rightp = 0;
    int leftp = 0;
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        cluster = pi[y*width + x].cluster;
        upp = y - 1;
        lowp = y + 1;
        rightp = x + 1;
        leftp = x - 1;
        if (upp >= 0 && lowp < height && rightp < width && leftp >= 0)
        {
          if (pi[upp*width + x].cluster != cluster || pi[lowp*width + x].cluster != cluster || pi[y*width + leftp].cluster != cluster || pi[y*width + rightp].cluster != cluster)
          result.set_pixel(x, y, 51, 20, 203);
        }
      }
    }
  }
  else
  {
    //Disegno tutti i pixel con il colore del loro cluster center
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        result.set_pixel(x, y, ck[pi[y*width + x].cluster].r, ck[pi[y*width + x].cluster].g, ck[pi[y*width + x].cluster].b);
      }
    }
  }
  // Disegno i cluster centers
  if (centri)
  {
    for (int i = 0; i < K; i++)
    {
      result.set_pixel(ck[i].x, ck[i].y, 111, 0, 255);
      
    }
  }
  result.save_image(output);
}

float getGpuMill() {
  return gpumil;
}
float getGpuAdvMill() {
  return gpuAdvmil;
}


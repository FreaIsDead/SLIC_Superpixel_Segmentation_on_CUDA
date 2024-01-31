#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "common.h"
#include "slic.h"
#include "gpuslic.h"

int main(int argc, char **argv)
{
  if (argc < 4)
  {
    printf("usage: inputImgName outputImgName lato_orizzontale_griglia\n");
    exit(1);
  }
  
  char *inputImg = argv[1];
  char *outputImg = argv[2];
  int sudd = atoi(argv[3]);
  std::string outcpu (outputImg);
  std::string outgpu (outputImg);
  outcpu.append("_cpu.bmp");
  outgpu.append("_gpu.bmp");


  //GPU SLIC
  gpuSLIC(inputImg, outgpu.c_str(), sudd);

  puts("\n-------------------------------------\n");

  //CPU SLIC
  cpuSLIC(inputImg, sudd);
  outputSlic(0, 1, outcpu.c_str());

  puts("\n-------------------------------------\n");
  printf("speedup CPU/GPU: %f x\n\n", getCpuMil()/getGpuMill());
  printf("speedup CPU/GPU_ADV: %f x\n\n", getCpuMil()/getGpuAdvMill());
  
  return 0;
}

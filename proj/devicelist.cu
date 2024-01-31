#include <stdio.h> 

int main() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("\tDevice name: %s\n", prop.name);
    printf("\tMemory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("\tMemory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("\tPeak Memory Bandwidth (GB/s): %f\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf ("\tCompute capability %d.%d.\n\n",
          prop.major, prop.minor);
  }
}
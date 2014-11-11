 /* 
 * DESCRIPTION      These functions are part of the submission to exercises of 
 *                  the "GPU Computing" lecture of the University of Heidelberg.
 * 
 *                  Exercise 3 - Implementation for the global memory access
 *                               measurements.
 * 
 * AUTHORS          Klaus Naumann
 *                  Alexander Schapp
 *                  Günther Schindler
 *
 * LAST CHANGE      11. Nov 2014
 * 
 ********************************************************************************/
#include <stdio.h>
#include "chTimer.h"

int main( int argc, char *argv[])
{
  int i;
  double dBandwidth;
  void *dmem_a, *dmem_b;     // device data (variable scope annotation)
  long iIteration[10] = {1e3, 1e4, 1e5, 1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9};
  chTimerTimestamp tsStart, tsStop;
  
  /* Run over 10 measurement points */
  for(i = 0; i < 10; i++)
  {
    /* Allocate device memory */
    cudaMalloc(&dmem_a, iIteration[i]*sizeof(char));
    cudaMalloc(&dmem_b, iIteration[i]*sizeof(char));

    /* Start timer */
    chTimerGetTime(&tsStart);
    /* Start memory copy */
    cudaMemcpy(dmem_a ,dmem_b , iIteration[i]*sizeof(char), cudaMemcpyDeviceToDevice);
    /* Stop timer */
    chTimerGetTime(&tsStop);
    
    /* Get bandwidth in Byte/sec and print it */
    dBandwidth=chTimerBandwidth(&tsStart, &tsStop, (double) iIteration[i] * sizeof(char));
    printf(" %11li Byte : %.2e B/s\n",iIteration[i],dBandwidth);
    
    /* Free allocated device memory */
    cudaFree(dmem_a);
    cudaFree(dmem_b);
  }

    return 0;
}

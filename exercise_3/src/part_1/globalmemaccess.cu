/*********************************************************************************
 * FILENAME         main.c
 * 
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
  void * dmem_a, dmem_b;     // device data (variable scope annotation)
  int iIteration[10] = {10e3, 10e4, 10e5, 10e6, 5*10e6, 10e7, 5*10e7, 10e8, 5*10e8, 10e9};
  chTimerTimestamp tsStart, tsStop;
  
  /* Run over 10 measurement points */
  for(i = 0; i < 10; i++)
  {
    /* Allocate device memory */
    dmem_a = cudaMalloc(iIteration[i]*sizeof(char));
    dmem_b = cudaMalloc(iIteration[i]*sizeof(char));

    /* Start timer */
    chTimerGetTime(&tsStart);
    /* Start memory copy */
    cudaMemcpy(dmem_a ,dmem_b , iItaration[i]*sizeof(char), cudaMemcpyDeviceToDevice);
    /* Stop timer */
    chTimerGetTime(&tsStop);
    
    /* Get bandwidth in Byte/sec and print it */
    dBandwidth=(&tsStart, &tsStart, (double)iIteration[i] * sizeof(char));
    printf("\n %d Byte : %lf B/s",iIteration[i],dBandwitdth);
    
    /* Free allocated device memory */
    cudeFree(dmem_a);
    cudaFree(dmem_b);
  }

    return 0;
}

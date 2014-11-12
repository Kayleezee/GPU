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
  char *dmem_a, *dmem_b;     // device data (variable scope annotation)
  char *hmem;
  long iIteration[13] = {1e3, 2e3, 1e4, 2e4, 1e5, 2e5, 1e6, 2e6, 1e7, 2e7, 1e8, 2e8, 5e9};
  chTimerTimestamp tsStart, tsStop;
  
  /* Run over 13 measurement points */
  for(i = 0; i < 13; i++)
  {
    /* Allocate device memory */
    cudaMalloc(&dmem_a, iIteration[i]*sizeof(char));
    cudaMalloc(&dmem_b, iIteration[i]*sizeof(char));
    
    hmem = (char*)malloc(iIteration[i]*sizeof(char));
    /* Fill elements with '1' */
    for(int j=0; j<iIteration[i]; j++)
      hmem[j] = 'a';
    cudaMemcpy(dmem_b, hmem, iIteration[i]*sizeof(char), cudaMemcpyHostToDevice);

    /* Start timer */
    chTimerGetTime(&tsStart);
    /* Start memory copy */
    cudaMemcpy(dmem_a ,dmem_b , iIteration[i]*sizeof(char), cudaMemcpyDeviceToDevice);
    /* Stop timer */
    chTimerGetTime(&tsStop);
    
    /* Get bandwidth in Byte/sec and print it */
    dBandwidth=chTimerBandwidth(&tsStart, &tsStop, (double) iIteration[i]);
    printf("%li %.2e\n",iIteration[i],dBandwidth/1e9);
 
    /* Free allocated device memory */
    cudaFree(dmem_a);
    cudaFree(dmem_b);
  }

    return 0;
}

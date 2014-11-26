/*********************************************************************************
 * FILENAME         main.c
 * 
 * DESCRIPTION      These functions are part of the submission to exercises of 
 *                  the "GPU Computing" lecture of the 
 *                  University of Heidelberg.
 * 
 *                  Exercise 4 - Matrix Multiplication
 * 
 * AUTHORS          Günther Schindler
 * 		    Klaus Naumann
 *                  Alexander Schnapp
 *
 * LAST CHANGE      26. Nov 2014
 * 
 ********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include "matrix_multiply.h"
#include "time_measurement.h"

void print_usage() {
	printf("\nUsage:\n");
	printf("<S matrix size>");
} 

int main(int argc, char* argv[]){
  if(argc != 2) {
  	print_usage();
  }
  
  int iSize 	= atoi(argv[1]);
  
  sMatrix sMa, sMb, sMRes;
  double dStartTime=0.0, dElapsedTime=0.0;
 
  /* Allocate memory for matrices */
  if(vAllocMatrix(&sMa, iSize, iSize))
    exit(1);
  if(vAllocMatrix(&sMb, iSize, iSize))
    exit(1);
  if(vAllocMatrix(&sMRes, iSize, iSize))
    exit(1);
  
  /* Initialize matrizen with random numbers */
  vInitMatrixA(&sMa);
  vInitMatrixB(&sMb);
  
  /* Start time measurement */
  dStartTime = dstartMeasurement();
  /* Start multiplication, non-optimized */
  iMatrixMultiply(&sMa, &sMb, &sMRes);
  /* Stop time-measurement*/
  dElapsedTime = dstopMeasurement(dStartTime);
  
  if(iSize <= 10) {
    vPrintMatrix(&sMa);
    vPrintMatrix(&sMb);
    vPrintMatrix(&sMRes);
  }

  printf("\n Time for calculation:  %lfs", dElapsedTime);
  double dGFLOPS = /*1e-9*/((double) (iSize*iSize*(2*iSize)))/dElapsedTime/1000000000;
  printf("\n GFLOP/s      :  %3.10f\n\n", dGFLOPS);
  
  vFreeMatrix(&sMa);
  vFreeMatrix(&sMb);
  vFreeMatrix(&sMRes);
  
  return EXIT_SUCCESS;
}
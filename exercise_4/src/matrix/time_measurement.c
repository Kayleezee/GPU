/*********************************************************************************
 * FILENAME         time_measurement.c
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
#include <sys/time.h>
#include "time_measurement.h"

/* Start time-measurement */
double dstartMeasurement(void)
{
  struct timeval tim;
  gettimeofday(&tim, NULL);
  return tim.tv_sec+(tim.tv_usec/1000000.0);
}

/* Stop time-measurement */
double dstopMeasurement(double dStartTime)
{
  struct timeval tim;
  gettimeofday(&tim, NULL);
  return (tim.tv_sec+(tim.tv_usec/1000000.0)) - dStartTime;
}
/*************************************************************************************************
*
* Heidelberg University - GPU Exercise 06
*
* Group :       PCA03
* Participant : Klaus Naumann
*				Günther Schindler
*               Alexander Schnapp
*
* File :        time_measurememnt.c
*
* Purpose :     GLOBAL SUM REDUCTION (CPU SEQUENTIAL VERSION)
*
* Last Change : 06. Dec. 2014
*
*************************************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "time_measurement.h"

/* Start time-measurement */
double dStartMeasurement(void)
{
  struct timeval tim;
  gettimeofday(&tim, NULL);
  return tim.tv_sec+(tim.tv_usec/1000000.0);
}

/* Stop time-measurement */
double dStopMeasurement(double dStartTime)
{
  struct timeval tim;
  gettimeofday(&tim, NULL);
  return (tim.tv_sec+(tim.tv_usec/1000000.0)) - dStartTime;
}

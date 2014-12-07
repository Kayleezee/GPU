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
#ifndef TIME_MEASUREMENT_H
#define TIME_MEASUREMENT_H

/*
 * DESCRIPTION - Starts a time-measurement, based on the gettimeofday() functions
 *               It has a resolution up to one microsecond.
 * PARAMETER   - void
 * RETURN      - double: elapsed seconds this day (is the parameter for dstopMesGTOD())
 */
double dStartMeasurement(void);

/*
 * DESCRIPTION - Stops the time-measurement, based on the gettimeofday() functions.
 *               It has a resolution up to one microsecond.
 * PARAMETER   - double: return-value of dstartMesGTOD()
 * RETURN      - double: elapsed seconds since dstartMesGTOD()
 */
double dStopMeasurement(double);

#endif

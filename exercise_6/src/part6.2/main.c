/*************************************************************************************************
*
* Heidelberg University - GPU Exercise 06
*
* Group :       PCA03
* Participant : Klaus Naumann
*				Günther Schindler
*               Alexander Schnapp
*
* File :        main.c
*
* Purpose :     GLOBAL SUM REDUCTION (CPU SEQUENTIAL VERSION)
*
* Last Change : 06. Dec. 2014
*
*************************************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "../../inc/time_measurement.h"

void print_usage() {
    printf("\nUsage:");
    printf("\t<S size of array>");
}

float *fFill(float *fData, int iSize) {
    int i;

    for(i = 0; i < iSize; i++) {
        fData[i] = (float)i;
    }

    return fData;
}

float fSum(float *fData, int iSize) {
    int i;

    float fRes = 0.0;

    for(i = 0; i < iSize; i++) {
        fRes += fData[i];
    }

    return fRes;
}

/*************************************************************************************************
 MAIN FUNCTION
**************************************************************************************************/
int main(int argc, char **argv) {
	int iSize    = atoi(argv[1]);
    int i;

    float fGlobalSum = 0.0;
    float *fNumbers;

    double dStart   = 0.0;
    double dTimeSeq = 0.0;
    double dGFLOPS  = 0.0;


	if(argc != 2) {
        print_usage();
        return EXIT_FAILURE;
	}

	printf("\n Starting GLOBAL SUM REDUCTION (CPU SEQUENTIAL VERSION) ...");

    fNumbers = (float *) malloc(iSize*sizeof(float));

    /* fill numbers-array with elements */
    fNumbers = fFill(fNumbers, iSize);

    /*
    for(i = 0; i < iSize; i++) {
        printf("\nElement No.: %d = %.2f", i, fNumbers[i]);
    }
    */

    dStart = dStartMeasurement();

    fGlobalSum = fSum(fNumbers, iSize);

    dTimeSeq = dStopMeasurement(dStart);

    printf("\n RESULTS:");
    printf("\n =========================================");
    printf("\n\n Global sum:\t%.2f", fGlobalSum);
    printf("\n Time elapsed:\t%lf", dTimeSeq);
    dGFLOPS = ((double) (iSize))/dTimeSeq/1000000000;
    printf("\n GFLOP/s:\t%3.10f\n", dGFLOPS);
    printf("\n =========================================\n\n");

    return EXIT_SUCCESS;
}

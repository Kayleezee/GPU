/*********************************************************************************
 * FILENAME         matrix_multiply.c
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

int iMatrixMultiply(sMatrix *pMa, sMatrix *pMb, sMatrix *pMRes)
{
  int i,j,k;

  if(pMa->iCol == pMb->iRow && pMa->iRow == pMRes->iRow && pMb->iCol == pMRes->iCol)
  {
    for(i=0; i<pMa->iRow; i++)
    {
      for(j=0; j<pMb->iCol; j++)
      {
      pMRes->ppaMat[i][j] = 0;
        for(k=0; k<pMa->iCol; k++)
	  pMRes->ppaMat[i][j] += pMa->ppaMat[i][k] * pMb->ppaMat[k][j];
      }
    }
    return 0;
  }
  else
  {
    printf("iMatrixMultiply: No Multiplication − Matrix size not correct\n");
    return 1;
  }
}


int vAllocMatrix(sMatrix *pM, int iRow, int iCol)
{ 
  int i=0, iRet=0;
  /* Init rows and cols in matrix structure */
  pM->iCol=iCol;
  pM->iRow=iRow;
  /* Alloc Mem for Rows-Pointer */
  pM->ppaMat =(int **) malloc(iRow * sizeof(int *));
  if(NULL == pM->ppaMat)
    iRet=1;
  /* Allocate Memory for Rows */
  for(i=0; i < iRow; i++)
  {
    pM->ppaMat[i] = (int *) malloc(iCol * sizeof(int *));
    if(NULL == pM->ppaMat[i])
      iRet=1;
  }
  return iRet;
}

void vFreeMatrix(sMatrix *pM)
{
  int i;
  /* free the Rows */
  for(i=0; i<pM->iRow; i++)
    free(pM->ppaMat[i]);
  /* free cols */
  free(pM->ppaMat);
}


void vInitMatrixA(sMatrix *pM) 
{
  int i, j;
  
  /* Fill the matrix-elements according to: A[i,j] = i + j */
  for(i = 0; i < pM->iRow; i++)
  {
    for(j = 0; j < pM->iCol; j++)
    {
      pM->ppaMat[i][j] = i+j;
    }
  }
}

void vInitMatrixB(sMatrix *pM) 
{
  int i, j;
  
  /* Fill the matrix-elements according to: B[i,j] = i * j */
  for(i = 0; i < pM->iRow; i++)
  {
    for(j = 0; j < pM->iCol; j++)
    {
      pM->ppaMat[i][j] = i*j;
    }
  }
}

void vPrintMatrix(sMatrix *pM)
{
  int i,j;
  /* matrix[zeile][spalte] */
  for(i=0; i<pM->iRow; i++)
  {
    printf("\n");
    for(j=0; j<pM->iCol; j++)
      printf("%d  ", pM->ppaMat[i][j]);
  }
  printf("\n");
}

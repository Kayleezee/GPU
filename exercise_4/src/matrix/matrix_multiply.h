/*********************************************************************************
 * FILENAME         matrix_multiply.h
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
#ifndef MATRIX_MULTIPLY_H
#define MATRIX_MULTIPLY_H

typedef struct sMatrix
{
  int iRow;          // for rows
  int iCol;          // for columns
  int** ppaMat;   // double[][]
} sMatrix;

/* 
 * DESCRIPTION - Naive matrix-matrix multiplication. Matrix A must have the same 
 *               amount of rows that the B has columns.
 * PARAMETER   - sMatrix *: pointer to matrix A which should be multiplied
 *               sMatrix *: pointer to matrix B which should be multiplied
 *               sMatrix *: pointer to matrix in which the result will be stored
 * RETURN      - 0 if multiplication was ok
 *               1 if not
 */
int iMatrixMultiply(sMatrix *, sMatrix *, sMatrix *);

/* 
 * DESCRIPTION - Tilled (blocking) matrix-matrix multiplication. Matrix A must have 
 *               the same amount of rows that the B has columns.
 * PARAMETER   - sMatrix *: pointer to matrix A which should be multiplied
 *               sMatrix *: pointer to matrix B which should be multiplied
 *               sMatrix *: pointer to matrix in which the result will be stored
 *               int : Block size
 * RETURN      - 0 if multiplication was ok
 *               1 if not
 */
int vAllocMatrix(sMatrix *,int, int);

/*
 * DESCRIPTION - will free the allocated memory.
 * PARAMETER   - pointer to type sMatrix, which should be destroid
 * RETURN      - void
 */
void vFreeMatrix(sMatrix *);

/*
 * DESCRIPTION - prints the matrix-elements (for testing)
 * PARAMETER   - pointer to matrix, which should be printed
 * RETURN      - void
 */
void vPrintMatrix(sMatrix *);

/*
 * DESCRIPTION - Initialize a matrix according to: A[i,j] = i + j
 * PARAMETER   - sMatrix: pointer to matrix
 * RETRUN      - void
 */
void vInitMatrixA(sMatrix *);

/*
 * DESCRIPTION - Initialize a matrix according to: B[i,j] = i * j
 * PARAMETER   - sMatrix: pointer to matrix
 * RETRUN      - void
 */
void vInitMatrixB(sMatrix *);

/*
* DESCRIPTION - Compares two matrices. If every
*               element is equal the function 
*               returns true.
* PARAMETER   - sMatrix*: first Matrix
*             - sMatrix*: second Matrix
* RETURN      - boolean
*/

#endif
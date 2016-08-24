/*
 ============================================================================
 Author      : Roberto Diaz Morales
 ============================================================================
 
 Copyright (c) 2016 Roberto Díaz Morales

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files
 (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge,
 publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
 subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
 FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 
 ============================================================================
 */

/**
 * @file ParallelAlgorithms.h
 * @author Roberto Diaz Morales
 * @date 23 Aug 2016
 * @brief Functions to perform some parallel linear algebra tasks.
 *
 * Parallel procedures to solve linear systems, cholesky factorization,
 * matrix products or triangular matrix inversion.
 */


#ifndef PARALLEL_ALGORITHMS_
#define PARALLEL_ALGORITHMS_


/**
 * @brief Function to allocate auxiliar memory to be used in the algebra operations.
 *
 * This functions has to be called at the beginning and to allocate memory as a function
 * of the number of threads and matrices size.
 */

void initMemory(int Threads, int size);


/**
 * @brief Function to update auxiliar memory to be used in the algebra operations.
 *
 * If the memory allocated with the function initMemory is not enough we can increase it
 * using this function.
 */

void updateMemory(int Threads, int size);


/**
 * @brief Main function to perform a parallel cholesky factorization.
 *
 * This function performs a parallel cholesky factorization. It uses openmp to create
 * different threads and every one of them performs a subtask using the function Chol.
 */

void ParallelChol(double *matrix,int r,int c, int ro, int co, int n,int nCores, int deep);


/**
 * @brief Auxiliar function to perform a parallel cholesky factorization.
 *
 * This function performs a subtask of a Cholesky factorization.
 */

void Chol(double *matrix,int r,int c, int ro, int co, int n,int nCores,int numTh, int deep,int posIni,double *memaux, int blockSize);


/**
 * @brief Main function to solve a linear system in parallel.
 *
 * This function solves a linear system. It uses openmp to create
 * different threads and every one of them performs a subtask using the function LinearSystem.
 */

void ParallelLinearSystem(double *matrix1,int r1,int c1, int ro1, int co1,double *matrix2,int r2,int c2, int ro2, int co2,int n, int m,double *result,int rr,int cr, int ror, int cor, int nCores);


/**
 * @brief Auxiliar function to solve a linear system in parallel.
 *
 * This function performs a subtask to solve a linear system by one of the threads of the processor.
 */

void LinearSystem(double *matrix1,int r1,int c1, int ro1, int co1,double *matrix2,int r2,int c2, int ro2, int co2,int n, int m,double *result,int rr,int cr, int ror, int cor, int nCores, int numTh,int posIni,double *memaux, int blockSize);


/**
 * @brief This function solves a triangular matrix inverse in parallel.
 *
 * This function solves a triangular matrix inverse in parallel.
 */

void DiagInversion(double *matrix,int r, int c, int ro, int co, int n,int nCores,int posIni,int numTh);


/**
 * @brief This function performs a product of a normal matrix and a lower triangular matrix in parallel.
 *
 * This function performs a product of a normal matrix and a lower triangular matrix in parallel.
 */

void InversionNLProducts(double *matrix,int r, int c, int ro, int co, int n,int nCores,int posIni,int deep,int numTh, double *memaux, int blockSize);


/**
 * @brief This function performs a product of a lower triangular matrix and a matrix in parallel.
 *
 * This function performs a product of a lower triangular matrix and a matrix in parallel.
 */

void InversionLNProducts(double *matrix,int r, int c, int ro, int co, int n,int nCores,int posIni,int deep,int numTh, double *memaux, int blockSize);


/**
 * @brief This function performs a lower triangular matrix inverse in parallel.
 *
 * This function performs a lower triangular matrix inverse in parallel.
 */

void TriangleInversion(double *matrix,int r, int c, int ro, int co, int n,int nCores,int posIni, int numTh,double *memaux,int blockSize);


/**
 * @brief This function performs a matrix product in parallel.
 *
 * This function performs a matrix product in parallel.
 */

void NNProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,int n3,double K1,double K2,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh,int orientation);


/**
 * @brief This function moves a piece of matrix from a matrix to another.
 *
 * This function moves a piece of matrix from a matrix to another.
 */

void MoveMatrix(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2, int n1, int n2, int nCores,int posIni,int numTh);


/**
 * @brief This function performs a product of the transpose of a matrix and another matrix in parallel.
 *
 * This function performs a product of the transpose of a matrix and another matrix in parallel.
 */

void TNNProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,int n3,double K1,double K2,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh);


/**
 * @brief This function performs a product of a matrix and the transpose of another matrix in parallel.
 *
 * This function performs a product of a matrix and the transpose of another matrix in parallel.
 */

void NNTProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,int n3,double K1,double K2,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh);


/**
 * @brief This function performs a product of a matrix and its transpose in parallel.
 *
 * This function performs a product of a matrix and its transpose in parallel.
 */

void AATProduct(double *m1,int r1,int ro1,int c1, int co1,int n1,int n2,double K1,double K2,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh);

/**
 * @brief This function performs a product of a lower triangular matrix and another matrix in parallel.
 *
 * This function performs a product of a lower triangular matrix and another matrix in parallel.
 */

void LNProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,double K1,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh);

/**
 * @brief This function performs a product of the transpose of a lower triangular matrix and another matrix in parallel.
 *
 * This function performs a product of the transpose of a lower triangular matrix and another matrix in parallel.
 */

void LTNProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,double K1,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh);

/**
 * @brief This function performs a product of a matrix and a lower triangular matrix in parallel.
 *
 * This function performs a product of a matrix and a lower triangular matrix in parallel.
 */

void NLProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,double K1,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh);


/**
 * @brief This function performs a product of a matrix and the transpose of a lower triangular matrix in parallel.
 *
 * This function performs a product of a matrix and the transpose of a lower triangular matrix in parallel.
 */

void NLTProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,double K1,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh);


/**
 * @brief This function copy a piece of a matrix in parallel.
 *
 * This function copy a piece of a matrix in parallel.
 */

void getSubMatrix(double *matrix,int size1,int size2,int O1,int O2,double *A, int size3,int size4,int nCores);


/**
 * @brief This function replace a piece of a matrix in parallel.
 *
 * This function replace a piece of a matrix in parallel.
 */

void putSubMatrix(double *matrix,int size1,int size2,int O1,int O2,double *A, int size3,int size4,int nCores);

#endif


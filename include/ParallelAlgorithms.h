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
 * @brief Functions to perform parallel linear algebra tasks.
 *
 * Parallel procedures to solve linear systems, cholesky factorization,
 * matrix products or triangular matrix inversion.
 *
 * This library also contains many auxiliar functions that are not detailed here with the goal of obtained a readable documentation. Their are only designed to be called by the main functions described here. However, that functions are detailed described in the source code. 
 *
 * @file ParallelAlgorithms.h
 * @author Roberto Diaz Morales
 * @date 23 Aug 2016
 */


#ifndef PARALLEL_ALGORITHMS_
#define PARALLEL_ALGORITHMS_


/**
 * @brief Function to allocate auxiliar memory to be used in the algebra operations.
 *
 * 
 * The parallel lineal algebra functions of this module require some memory for every thead to
 * allocate temporal results. This function must be called before any other function.
 *
 * @param Threads The number of threads to parallelize the linear algebra functions.
 * @param size The size of the dimensions of the matrices that will be handle. If a matrix has a rows and b columns, then n=max(a,b).
 * @see updateMemory()
 */

void initMemory(int Threads, int size);

/**
 * @brief Function to free auxiliar memory to be used in the algebra operations.
 *
 * 
 * The parallel lineal algebra functions of this module require some memory for every thead to
 * allocate temporal results. This function must be called after any other function.
 *
 * @param Threads The number of threads to parallelize the linear algebra functions.
 * @see updateMemory()
 */

void freeMemory(int Threads);


/**
 * @brief Function to update auxiliar memory to be used in the algebra operations.
 *
 * The parallel lineal algebra functions of this module require some memory for every thead to
 * allocate temporal results. This function must be called if we allocated the memory using the function initMemory and we are going to work with bigger matrices.
 *
 * @param Threads The number of threads to parallelize the linear algebra functions.
 * @param size The size of the dimensions of the matrices that will be handle. If a matrix has a rows and b columns, then n=max(a,b).
 * @see initMemory()
 */

void updateMemory(int Threads, int size);

/**
 * @brief This function saves a piece of a matrix in the auxiliar memory.
 *
 * This is an auxiliry function of the linear algebra funtions. It is used to save a submatrix of  
 * a matrix given as a parameter in the auxiliar memory.
 *
 * @param matrix The original matrix.
 * @param size1 The number of rows fo the original matrix.
 * @param size2 The number of columns of the original matrix.
 * @param O1 The row where the submatrix starts.
 * @param O2 The column where the submatrix starts.
 * @param A The pointer where the submatrix will be storaged.
 * @param size3 Number of rows of the submatrix to storage.
 * @param size4 Number of columns of the submatrix to storage.
 * @param nCores Number of threads to perform this task.
 */

void getSubMatrix(double *matrix,int size1,int size2,int O1,int O2,double *A, int size3,int size4,int nCores);


/**
 * @brief This function loads a piece of a matrix in the auxiliar memory and storage it in a matrix.
 *
 * This is an auxiliry function of the linear algebra funtions. It is used to load a matrix from  
 * the auxiliar memory and storage it as a submatrix of another matrix.
 *
 * @param matrix The original matrix where the submatrix will be allocated.
 * @param size1 The number of rows fo the original matrix.
 * @param size2 The number of columns of the original matrix.
 * @param O1 The row of the original matrix where the submatrix starts.
 * @param O2 The column of the original matrix where the submatrix starts.
 * @param A The pointer where the submatrix is storaged.
 * @param size3 Number of rows of the submatrix that is storaged.
 * @param size4 Number of columns of the submatrix that is storaged.
 * @param nCores Number of threads to perform this task.
 */

void putSubMatrix(double *matrix,int size1,int size2,int O1,int O2,double *A, int size3,int size4,int nCores);

/**
 * @brief Main function that performs a parallel cholesky factorization.
 *
 * This function performs a parallel cholesky factorization on a sqaure submatrix 
 * of a matrix recived as an argument. It uses openmp to create to parallelize the task using
 * different threads and every one of them performs a subtask using the function Chol.
 * @param matrix The matrix to perform the cholesky factorization.
 * @param r The number of rows of matrix
 * @param c The number of columns of matrix.
 * @param ro The row where the submatrix starts.
 * @param co The column where the submatrix starts.
 * @param n The order of the square submatrix
 * @param nCores The number of threads to perform the task.
 * @param deep It is a recursive function, this parameter tell the recursion deep to use.
 * @see Chol()
 */

void ParallelChol(double *matrix,int r,int c, int ro, int co, int n,int nCores, int deep);


/**
 * @brief Main function to solve a linear system in parallel.
 *
 * This function solves a linear system of two submatrices of matrix1 and matrix2.
 * It uses openmp to create different threads and every one of them performs a subtask
 * using the function LinearSystem.
 * If submatrix1 is the submatrix of matrix1 and submatrix2 is the submatrix of matrix2 then
 * it performs (submatrix1)^-1*submatrix2.
 *
 * @param matrix1 The first matrix.
 * @param r1 The number of rows of matrix1.
 * @param c1 The number of columns of matrix1.
 * @param ro1 The row where the submatrix starts.
 * @param co1 The column where the submatrix starts.
 * @param matrix2 The second matrix.
 * @param r2 The number of rows of matrix2.
 * @param c2 The number of columns of matrix2.
 * @param ro2 The row where the second submatrix starts.
 * @param co2 The column where the second submatrix starts.
 * @param n The number of rows of the submatrix1
 * @param m The number of columns of the submatrix2.
 * @param result The matrix where the result should be storaged.
 * @param rr The number of rows of matrix result.
 * @param cr The number of columns of matrix result.
 * @param ror The row where the result submatrix starts.
 * @param cor The column where the result submatrix starts.
 * @param nCores The number of threads to launch to solve the task.
 */

void ParallelLinearSystem(double *matrix1,int r1,int c1, int ro1, int co1,double *matrix2,int r2,int c2, int ro2, int co2,int n, int m,double *result,int rr,int cr, int ror, int cor, int nCores);


/**
 * @brief This function performs a product of a vector and the transpose of a square matrix in parallel.
 *
 * This function performs a product of a vector and the transpose of a matrix in parallel.
 * 
 * @param m1 The vector.
 * @param size The length of the vector and the order of the square matrix.
 * @param m2 The square matrix.
 * @param result The matrix to storage the result.
 * @param numThreads Number of threads to pefrom this task.
 */

void ParallelVectorMatrixT(double *m1,int size,double *m2,double *result, int numThreads);

/**
 * @brief This function performs a product of a vector and a square matrix in parallel.
 *
 * This function performs a product of a vector and a matrix in parallel.
 *
 * This function performs a vector matrix product in parallel.
 * @param m1 The vector.
 * @param size The length of the vector and the order of the square matrix.
 * @param m2 The square matrix.
 * @param result The matrix to storage the result.
 * @param numThreads Number of threads to pefrom this task.
 */

void ParallelVectorMatrix(double *m1,int size,double *m2,double *result, int numThreads);

/**
 * @cond
 */

/**
 * @brief Auxiliar function to perform a parallel cholesky factorization.
 *
 * This function is called by ParallelChol, that creates different threads and call this function
 * on each one using its thread index and the total number of threads as parameters.
 *
 * This is a recursive function where every thread finally find out the part of the matrix that need
 * and the linear algebra operations that it has to do.
 * 
 * @param matrix The matrix to perform the cholesky factorization.
 * @param r The number of rows of matrix
 * @param c The number of columns of matrix.
 * @param ro The row where the submatrix starts.
 * @param co The column where the submatrix starts.
 * @param n The order of the square submatrix
 * @param nCores The number of threads to perform the task.
 * @param numTh Thread identifier.
 * @param posIni Variable to know what task to do by this thread.
 * @param memaux The auxiliar memory.
 * @param blockSize The size of its memory to save temporal results.
 * @param deep It is a recursive function, this parameter tell the recursion deep to use.
 * @see ParallelChol()
 * @see initMemory()
 * @see updateMemory()
 */

void Chol(double *matrix,int r,int c, int ro, int co, int n,int nCores,int numTh, int deep,int posIni,double *memaux, int blockSize);


/**
 * @brief This function is called by ParallelLinearSystems, that creates different threads and call this function
 * on each one using its thread index and the total number of threads as parameters.
 *
 * This is a recursive function where every thread finally find out the part of the matrix that need
 * and the linear algebra operations that it has to do.
 *
 * @param matrix1 The first matrix.
 * @param r1 The number of rows of matrix1.
 * @param c1 The number of columns of matrix1.
 * @param ro1 The row where the submatrix starts.
 * @param co1 The column where the submatrix starts.
 * @param matrix2 The second matrix.
 * @param r2 The number of rows of matrix2.
 * @param c2 The number of columns of matrix2.
 * @param ro2 The row where the second submatrix starts.
 * @param co2 The column where the second submatrix starts.
 * @param n The number of rows of the submatrix1
 * @param m The number of columns of the submatrix2.
 * @param result The matrix where the result should be storaged.
 * @param rr The number of rows of matrix result.
 * @param cr The number of columns of matrix result.
 * @param ror The row where the result submatrix starts.
 * @param cor The column where the result submatrix starts.
 * @param nCores The number of threads to launch to solve the task.
 * @param numTh Thread identifier.
 * @param posIni Variable to know what task to do by this thread.
 * @param memaux The auxiliar memory.
 * @param blockSize The size of its memory to save temporal results.
 * @see ParallelLinearSystem()
 */

void LinearSystem(double *matrix1,int r1,int c1, int ro1, int co1,double *matrix2,int r2,int c2, int ro2, int co2,int n, int m,double *result,int rr,int cr, int ror, int cor, int nCores, int numTh,int posIni,double *memaux, int blockSize);


/**
 * @brief This function performs a lower triangular matrix inverse in parallel of a submatrix.
 *
 * This function performs a lower triangular matrix inverse in parallel. It is used by the
 * ParallelLinearSystem function to solve the linear system using Cholesky Factorization.
 *
 * It makes use of the functions DiagInversion, InversionNLProducts and InversionLNProducts
 * to tell every thread the operations that they must do.
 * 
 *
 * @param matrix The matrix.
 * @param r The number of rows of matrix.
 * @param c The number of columns of matrix.
 * @param ro The row where the submatrix starts.
 * @param co The column where the submatrix starts.
 * @param n The order of the submatrix.
 * @param nCores The number of threads to perform the matrix inversion.
 * @param posIni Variable to know what task to do by this thread.
 * @param numTh Thread identifier.
 * @param memaux The auxiliar memory.
 * @param blockSize The size of its memory to save temporal results.
 * @see DiagInversion()
 * @see InversionNLProducts()
 * @see InversionLNProducts()
 */

void TriangleInversion(double *matrix,int r, int c, int ro, int co, int n,int nCores,int posIni, int numTh,double *memaux,int blockSize);


/**
 * @brief This function is an auxiliar function TriangleInversion.
 *
 * This function is an auxiliar function of TriangleInversion and is in charge of small matrix
 * inversions of the submtrices that has a portion of the diagonal.
 *
 * @param matrix The matrix.
 * @param r The number of rows of matrix.
 * @param c The number of columns of matrix.
 * @param ro The row where the submatrix starts.
 * @param co The column where the submatrix starts.
 * @param n The order of the submatrix.
 * @param nCores The number of threads to perform the matrix inversion.
 * @param posIni Variable to know what task to do by this thread.
 * @param numTh Thread identifier.
 * @see TriangleInversion()
 */

void DiagInversion(double *matrix,int r, int c, int ro, int co, int n,int nCores,int posIni,int numTh);


/**
 * @brief This function is an auxiliar function TriangleInversion.
 *
 * This function is an auxiliar function of TriangleInversion and is in charge of products
 * of matrices and lower triangular matrices (previously storaged in the auxiliar memory) that
 * are portions of the original matrix.
 *
 * @param matrix The matrix.
 * @param r The number of rows of matrix.
 * @param c The number of columns of matrix.
 * @param ro The row where the submatrix starts.
 * @param co The column where the submatrix starts.
 * @param n The order of the submatrix.
 * @param nCores The number of threads to perform the matrix inversion.
 * @param posIni Variable to know what task to do by this thread.
 * @param deep The deep of the recursion.
 * @param numTh Thread identifier.
 * @param memaux The auxiliar memory for this task.
 * @param blockSize The size of its memory to save temporal results.
 * @see TriangleInversion()
 */

void InversionNLProducts(double *matrix,int r, int c, int ro, int co, int n,int nCores,int posIni,int deep,int numTh, double *memaux, int blockSize);


/**
 * @brief This function is an auxiliar function TriangleInversion.
 *
 * This function is an auxiliar function of TriangleInversion and is in charge of products
 * of lower triangular matrices (previously storaged in the auxiliar memory) and matrices that
 * are portions of the original matrix.
 *
 * @param matrix The matrix.
 * @param r The number of rows of matrix.
 * @param c The number of columns of matrix.
 * @param ro The row where the submatrix starts.
 * @param co The column where the submatrix starts.
 * @param n The order of the submatrix.
 * @param nCores The number of threads to perform the matrix inversion.
 * @param posIni Variable to know what task to do by this thread.
 * @param deep The deep of the recursion.
 * @param numTh Thread identifier.
 * @param memaux The auxiliar memory for this task.
 * @param blockSize The size of its memory to save temporal results.
 * @see TriangleInversion()
 */

void InversionLNProducts(double *matrix,int r, int c, int ro, int co, int n,int nCores,int posIni,int deep,int numTh, double *memaux, int blockSize);


/**
 * @brief An auxiliar function of used to perform parallel matrix products.
 * 
 * This is an auxiliar function that takes subm1 (a sub matrix of matrix m1), subm2 (a submatrix of matrix m2) and performs the following operation on subresult (a sub matrix
 * of matrix result): subresult = K1*subm1*subm2+K2*subresult
 *
 * @param m1 The first matrix.
 * @param r1 The number of rows of the first matrix.
 * @param c1 The number of columns of the first matrix.
 * @param ro1 The row where the submatrix of m1 starts.
 * @param co1 The column where the submatrix of m1 starts.
 * @param m2 The second matrix.
 * @param r2 The number of rows of matrix.
 * @param c2 The number of columns of matrix.
 * @param ro2 The row where the submatrix starts.
 * @param co2 The column where the submatrix starts.
 * @param n1 The number of rows of the subm1.
 * @param n2 The number of columns of the subm1.
 * @param n3 The number of columns of the subm2.
 * @param K1 The first constant of the formula.
 * @param K2 The second constant of the formula.
 * @param result The matrix to storage the result.
 * @param rr The number of rows of result.
 * @param cr The number of columns of result.
 * @param ror The row where the submatrix of result starts.
 * @param cor The column where the submatrix of result starts.
 * @param nCores The total number of threads.
 * @param posIni Variable to know what task to do by this thread.
 * @param numTh The thread identifier.
 * @param orientation Auxiliar variable to know the next task to perform.
 */

void NNProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,int n3,double K1,double K2,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh,int orientation);


/**
 * @brief An auxiliar function of used to perform parallel matrix products.
 * 
 * This is an auxiliar function that takes subm1 (a sub matrix of matrix of the transpose matrix of m1), subm2 (a submatrix of matrix m2
 * and performs the following operation on subresult (a sub matrix of matrix result): subresult = K1*subm1*subm2+K2*subresult
 *
 * @param m1 The first matrix.
 * @param r1 The number of rows of the first matrix.
 * @param c1 The number of columns of the first matrix.
 * @param ro1 The row where the submatrix of m1 starts.
 * @param co1 The column where the submatrix of m1 starts.
 * @param m2 The second matrix.
 * @param r2 The number of rows of matrix.
 * @param c2 The number of columns of matrix.
 * @param ro2 The row where the submatrix starts.
 * @param co2 The column where the submatrix starts.
 * @param n1 The number of rows of the subm1.
 * @param n2 The number of columns of the subm1.
 * @param n3 The number of columns of the subm2.
 * @param K1 The first constant of the formula.
 * @param K2 The second constant of the formula.
 * @param result The matrix to storage the result.
 * @param rr The number of rows of result.
 * @param cr The number of columns of result.
 * @param ror The row where the submatrix of result starts.
 * @param cor The column where the submatrix of result starts.
 * @param nCores The total number of threads.
 * @param posIni Variable to know what task to do by this thread.
 * @param numTh The thread identifier.
 */

void TNNProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,int n3,double K1,double K2,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh);

/**
 * @brief An auxiliar function of used to perform parallel matrix products.
 * 
 * This is an auxiliar function that takes subm1 (a sub matrix of matrix of m1), subm2 (a submatrix of the transpose matrix of m2
 * and performs the following operation on subresult (a sub matrix of matrix result): subresult = K1*subm1*subm2+K2*subresult
 *
 * @param m1 The first matrix.
 * @param r1 The number of rows of the first matrix.
 * @param c1 The number of columns of the first matrix.
 * @param ro1 The row where the submatrix of m1 starts.
 * @param co1 The column where the submatrix of m1 starts.
 * @param m2 The second matrix.
 * @param r2 The number of rows of matrix.
 * @param c2 The number of columns of matrix.
 * @param ro2 The row where the submatrix starts.
 * @param co2 The column where the submatrix starts.
 * @param n1 The number of rows of the subm1.
 * @param n2 The number of columns of the subm1.
 * @param n3 The number of columns of the subm2.
 * @param K1 The first constant of the formula.
 * @param K2 The second constant of the formula.
 * @param result The matrix to storage the result.
 * @param rr The number of rows of result.
 * @param cr The number of columns of result.
 * @param ror The row where the submatrix of result starts.
 * @param cor The column where the submatrix of result starts.
 * @param nCores The total number of threads.
 * @param posIni Variable to know what task to do by this thread.
 * @param numTh The thread identifier.
 */

void NNTProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,int n3,double K1,double K2,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh);

/**
 * @brief An auxiliar function of used to perform parallel matrix products.
 * 
 * This is an auxiliar function that takes subm1 (a sub matrix of matrix of m1) and performs the following operation
 * on subresult (a sub matrix of matrix result): subresult = K1*subm1*subm1^T+K2*subresult
 *
 * @param m1 The first matrix.
 * @param r1 The number of rows of the first matrix.
 * @param c1 The number of columns of the first matrix.
 * @param ro1 The row where the submatrix of m1 starts.
 * @param co1 The column where the submatrix of m1 starts.
 * @param n1 The number of rows of the subm1.
 * @param n2 The number of columns of the subm1.
 * @param K1 The first constant of the formula.
 * @param K2 The second constant of the formula.
 * @param result The matrix to storage the result.
 * @param rr The number of rows of result.
 * @param cr The number of columns of result.
 * @param ror The row where the submatrix of result starts.
 * @param cor The column where the submatrix of result starts.
 * @param nCores The total number of threads.
 * @param posIni Variable to know what task to do by this thread.
 * @param numTh The thread identifier.
 */

void AATProduct(double *m1,int r1,int ro1,int c1, int co1,int n1,int n2,double K1,double K2,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh);

/**
 * @brief This function performs a product of a lower triangular submatrix of matrix m1
 * and a submatrix of m2 in parallel.
 *
 * Begin subm1 a lower triangular submatrix of matrix m1 and subm2 a submatrix of m2,
 * this funtion execute K1(subm1*subm2) and saves the result in a different matrix.
 *
 * @param m1 The first matrix.
 * @param r1 The number of rows of the first matrix.
 * @param c1 The number of columns of the first matrix.
 * @param ro1 The row where the submatrix of m1 starts.
 * @param co1 The column where the submatrix of m1 starts.
 * @param m2 The second matrix.
 * @param r2 The number of rows of matrix.
 * @param c2 The number of columns of matrix.
 * @param ro2 The row where the submatrix starts.
 * @param co2 The column where the submatrix starts.
 * @param n1 The number of rows of the submatrix1.
 * @param n2 The number of columns of the submatrix2.
 * @param K1 The constant of the formula.
 * @param result The matrix to storage the result.
 * @param rr The number of rows of result.
 * @param cr The number of columns of result.
 * @param ror The row where the submatrix of result starts.
 * @param cor The column where the submatrix of result starts.
 * @param nCores The total number of threads.
 * @param posIni Variable to know what task to do by this thread.
 * @param numTh The thread identifier.
 */

void LNProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,double K1,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh);


/**
 * @brief This function performs a product of a lower triangular submatrix of the transpose of matrix m1
 * and a submatrix of m2 in parallel.
 *
 * Begin subm1 a lower triangular submatrix of the transpose of matrix m1 and subm2 a submatrix of m2,
 * this funtion execute K1(subm1*subm2) and saves the result in a different matrix.
 *
 * @param m1 The first matrix.
 * @param r1 The number of rows of the first matrix.
 * @param c1 The number of columns of the first matrix.
 * @param ro1 The row where the submatrix of m1 starts.
 * @param co1 The column where the submatrix of m1 starts.
 * @param m2 The second matrix.
 * @param r2 The number of rows of matrix.
 * @param c2 The number of columns of matrix.
 * @param ro2 The row where the submatrix starts.
 * @param co2 The column where the submatrix starts.
 * @param n1 The number of rows of the submatrix1.
 * @param n2 The number of columns of the submatrix2.
 * @param K1 The constant of the formula.
 * @param result The matrix to storage the result.
 * @param rr The number of rows of result.
 * @param cr The number of columns of result.
 * @param ror The row where the submatrix of result starts.
 * @param cor The column where the submatrix of result starts.
 * @param nCores The total number of threads.
 * @param posIni Variable to know what task to do by this thread.
 * @param numTh The thread identifier.
 */

void LTNProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,double K1,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh);

/**
 * @brief This function performs a product of a submatrix of matrix m1 and a triangular submatrix 
 * of m2 in parallel.
 *
 * Begin subm1 a submatrix of m1 and subm2 a triangular submatrix of m2,
 * this funtion execute K1(subm1*subm2) and saves the result in a different matrix.
 *
 * @param m1 The first matrix.
 * @param r1 The number of rows of the first matrix.
 * @param c1 The number of columns of the first matrix.
 * @param ro1 The row where the submatrix of m1 starts.
 * @param co1 The column where the submatrix of m1 starts.
 * @param m2 The second matrix.
 * @param r2 The number of rows of matrix.
 * @param c2 The number of columns of matrix.
 * @param ro2 The row where the submatrix starts.
 * @param co2 The column where the submatrix starts.
 * @param n1 The number of rows of the submatrix1.
 * @param n2 The number of columns of the submatrix1 (and rows and columns of submatrix2).
 * @param K1 The constant of the formula.
 * @param result The matrix to storage the result.
 * @param rr The number of rows of result.
 * @param cr The number of columns of result.
 * @param ror The row where the submatrix of result starts.
 * @param cor The column where the submatrix of result starts.
 * @param nCores The total number of threads.
 * @param posIni Variable to know what task to do by this thread.
 * @param numTh The thread identifier.
 */

void NLProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,double K1,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh);


/**
 * @brief This function performs a product of a submatrix of matrix m1 and a triangular submatrix 
 * of the transpose of m2 in parallel.
 *
 * Begin subm1 a submatrix of m1 and subm2 a triangular submatrix of the transpose of m2,
 * this funtion execute K1(subm1*subm2) and saves the result in a different matrix.
 *
 * @param m1 The first matrix.
 * @param r1 The number of rows of the first matrix.
 * @param c1 The number of columns of the first matrix.
 * @param ro1 The row where the submatrix of m1 starts.
 * @param co1 The column where the submatrix of m1 starts.
 * @param m2 The second matrix.
 * @param r2 The number of rows of matrix.
 * @param c2 The number of columns of matrix.
 * @param ro2 The row where the submatrix starts.
 * @param co2 The column where the submatrix starts.
 * @param n1 The number of rows of the submatrix1.
 * @param n2 The number of columns of the submatrix1 (and rows and columns of submatrix2).
 * @param K1 The constant of the formula.
 * @param result The matrix to storage the result.
 * @param rr The number of rows of result.
 * @param cr The number of columns of result.
 * @param ror The row where the submatrix of result starts.
 * @param cor The column where the submatrix of result starts.
 * @param nCores The total number of threads.
 * @param posIni Variable to know what task to do by this thread.
 * @param numTh The thread identifier.
 */

void NLTProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,double K1,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh);


/**
 * @brief It moves a submatrix from matrix m1 to matrix m2.
 *
 * This function takes a submatrix form matrix m2 and allocated it in another submatrix of matrix m2.
 * @param m1 The matrix origin.
 * @param r1 The number of rows of the origin matrix.
 * @param c1 The number of columns of the origin matrix.
 * @param ro1 The row of the origin matrix where the submatrix starts.
 * @param co1 The column of the origin where the submatrix starts.
 * @param m2 The destination matrix.
 * @param r2 The number of rows of the destination matrix.
 * @param c2 The number of columns of the destination matrix.
 * @param ro2 The row of the destination matrix where the submatrix starts.
 * @param co2 The column of the destination matrix where the submatrix starts.
 * @param n1 The number of rows of the submatrix.
 * @param n2 The number of columns of the submatrix.
 * @param numTh The thread identifier.
 * @param nCores The total number of threads.
 * @param posIni Variable to know what task to do by this thread.
 */

void MoveMatrix(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2, int n1, int n2, int nCores,int posIni,int numTh);

/**
 * @endcond
 */

#endif


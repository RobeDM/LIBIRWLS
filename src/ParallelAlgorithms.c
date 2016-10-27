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
 * @brief Functions to perform some parallel linear algebra tasks.
 *
 * Parallel procedures to solve linear systems, cholesky factorization,
 * matrix products or triangular matrix inversion.
 *
 * @file ParallelAlgorithms.c
 * @author Roberto Diaz Morales
 * @date 23 Aug 2016
 * @see ParallelAlgorithms.h
 */


#include "ParallelAlgorithms.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>



/**
 * @cond
 */


extern void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double
                   *alpha, double *a, int *lda, double *b, int *ldb, double *beta, double *c,
                   int *ldc );

extern void dpotrs_(char *uplo, int *n, int *nrhs, double *A, int *lda,
                    double *B, int *ldb, int *info);

extern void dtrtri_(char *uplo, char *diag, int *n, double  *a, int *lda,
                    int *info);

extern void dpotrf_(char *uplo, int *n, double *A, int *lda, int *info);

extern void dsyrk_(char   *uplo, char   *trans, int    *n, int    *k,
                  double *alpha, double *a, int    *lda, double *beta,
                  double *c, int    *ldc);

extern void dtrmm_(char *side, char *uplo, char *transA, char *diag,
                   int *m, int *n, double *alpha, double *A, int *ldA,
                   double *B, int *ldB);


/** @brief Auxiliar memory to perform temporal results in the parallel algebra operations. */
double **auxmemory1;
/** @brief Auxiliar memory to perform temporal results in the parallel algebra operations. */
double **auxmemory2;
/** @brief Auxiliar memory to perform temporal results in the parallel algebra operations. */
double **auxmemory3;

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

void initMemory(int Threads, int size){

    auxmemory1=(double **) calloc(Threads,sizeof(double*));
    auxmemory2=(double **) calloc(Threads,sizeof(double*));
    auxmemory3=(double **) calloc(Threads,sizeof(double*));

    int i;
    for(i=0;i<Threads;i++){
        auxmemory1[i]=(double *) calloc(pow(ceil(1.0*(size)),2),sizeof(double));
        auxmemory2[i]=(double *) calloc(pow(ceil(1.0*(size)),2),sizeof(double));
        auxmemory3[i]=(double *) calloc(pow(ceil(1.0*(size)),2),sizeof(double));
    }            

}


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

void freeMemory(int Threads){

    int i;

    for(i=0;i<Threads;i++){
        free(auxmemory1[i]);
        free(auxmemory2[i]);
        free(auxmemory3[i]);
    } 

    free(auxmemory1);
    free(auxmemory2);
    free(auxmemory3);
}




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

void updateMemory(int Threads, int size){

    int i;
    for(i=0;i<Threads;i++){
        auxmemory1[i]=realloc(auxmemory1[i],pow(ceil(1.0*(size)),2)*sizeof(double));
        auxmemory2[i]=realloc(auxmemory2[i],pow(ceil(1.0*(size)),2)*sizeof(double));
        auxmemory3[i]=realloc(auxmemory3[i],pow(ceil(1.0*(size)),2)*sizeof(double));
    }

}

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

void getSubMatrix(double *matrix,int size1,int size2,int O1,int O2,double *A, int size3,int size4,int nCores){
    
    int j;
    
    for(j=0;j<size4;j++){
        memcpy(&A[j*size3],&matrix[(j+O2)*size1+O1],size3*sizeof(double));
    }    
}

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

void putSubMatrix(double *matrix,int size1,int size2,int O1,int O2,double *A, int size3,int size4,int nCores){
    
    int j;
    
    for(j=0;j<size4;j++){
        memcpy(&matrix[(j+O2)*size1+O1],&A[j*size3],size3*sizeof(double));
    }

}

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

void ParallelChol(double *matrix,int r,int c, int ro, int co, int n,int nCores, int deep){    
    double *memaux = (double *)calloc(2*pow(ceil(0.5*n),2),sizeof(double));
    int blockSize = pow(ceil(0.5*n),2)/nCores;    
    
    int i;
    #pragma omp parallel default(shared) private(i)
    {    
    #pragma omp for schedule(static)    
    for (i=0;i<nCores;i++){
        Chol(matrix,r,c,ro,co,n,nCores,i,deep,0,memaux,blockSize);
    }
    }
    free(memaux);        
}

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

void Chol(double *matrix,int r,int c, int ro, int co, int n,int nCores,int numTh, int deep,int posIni,double *memaux, int blockSize){    
    if(deep<=1 | n < 8){    
        if(numTh==posIni & n>0){
           
            double *m=auxmemory1[numTh];        
            getSubMatrix(matrix,r,c,ro,co,m, n,n,1);
            int info;
            char s='L';
            dpotrf_(&s,&n, m, &n,&info);
            
            if(info != 0 & posIni==2){
                printf("Error en dpotrf %d ------------------------------\n",info);
                exit(0);
            }

            int i,j;
            for(i=1;i<n;i++){
                for(j=0;j<i;j++){
                    m[i*n+j]=0.0;
                }
            }
            
            putSubMatrix(matrix,r,c,ro,co,m, n,n,1);

        }
    }else{

        int size1=ceil(0.5*n);
        int size2=n-size1;

        Chol(matrix,r,c,ro,co,size1,nCores,numTh,deep-1,posIni,memaux,blockSize);

        #pragma omp barrier            
        MoveMatrix(matrix,r,ro,c,co,&memaux[posIni*blockSize],size1,0,size1,0,size1,size1, nCores,posIni,numTh);

        #pragma omp barrier                
        TriangleInversion(&memaux[posIni*blockSize],size1, size1, 0, 0, size1, nCores,posIni,numTh,&memaux[size1*size1],blockSize);

        #pragma omp barrier            
        MoveMatrix(matrix,r,ro+size1,c,co,&memaux[posIni*blockSize+size1*size1],size2,0,size1,0,size2,size1, nCores,posIni,numTh);

        #pragma omp barrier            
        NLTProduct(&memaux[posIni*blockSize],size1,0,size1,0,&memaux[posIni*blockSize+size1*size1],size2,0,size1,0,size1,size2,1.0,matrix,r,ro+size1,c, co, nCores,posIni,numTh);

        #pragma omp barrier            
        AATProduct(matrix,r,ro+size1,c,co,size2,size1,-1.0,1.0,matrix,r,ro+size1,c, co+size1, nCores,posIni,numTh);

        #pragma omp barrier            
        if(numTh==posIni){
            double *Zeroes = (double *) malloc(size1*size2*sizeof(double));
            putSubMatrix(matrix,r,c,ro,co+size1,Zeroes, size1,size2,1);
            free(Zeroes);
        }

        #pragma omp barrier            
        Chol(matrix,r,c,ro+size1,co+size1,size2,nCores,numTh,deep-1,posIni,memaux,blockSize);        
        
        #pragma omp barrier                            
        
    }
}

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


void ParallelLinearSystem(double *matrix1,int r1,int c1, int ro1, int co1,double *matrix2,int r2,int c2, int ro2, int co2,int n, int m,double *result,int rr,int cr, int ror, int cor, int nCores){

    if(n>nCores){
    
        double *memaux = (double *)calloc(2*pow(ceil(0.5*n),2),sizeof(double));
        int blockSize = pow(ceil(0.5*n),2)/nCores;    
        
        int i;
    
        #pragma omp parallel default(shared) private(i)
        {    
        #pragma omp for schedule(static)    
        for (i=0;i<nCores;i++){
            LinearSystem(matrix1,r1,c1,ro1,co1,matrix2,r2,c2,ro2,co2,n,m,result,rr,cr,ror,cor,nCores,i,0,memaux,blockSize);
        }
        }
        free(memaux);
    }else{

        int info;
        char s='L';
        int cols=1;
        dpotrf_(&s,&n, matrix1, &n,&info);
        memcpy(result,matrix2,n*sizeof(double));
        dpotrs_(&s,&n,&cols, matrix1, &n, result,&n,&info);
        
    }
}

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

void LinearSystem(double *matrix1,int r1,int c1, int ro1, int co1,double *matrix2,int r2,int c2, int ro2, int co2,int n, int m,double *result,int rr,int cr, int ror, int cor, int nCores, int numTh,int posIni,double *memaux, int blockSize){    
         

    int deep=2;
    Chol(matrix1,r1,c1,ro1,co1,n,nCores,numTh,deep,posIni,memaux,blockSize);

    #pragma omp barrier    
        
    int info;
    char s='L';
    int ncols = 1;
    int j,k;
    for(j=0;j<n;j++){
        for(k=0;k<j;k++){
            matrix1[j*n+k]=0.0;
        }
    }

  
    if(numTh==0){

        memcpy(&result[0],&matrix2[0],n*sizeof(double));
        dpotrs_(&s,&n,&ncols, &matrix1[0], &n, &result[0],&n,&info);
    }
       
}

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


void TriangleInversion(double *matrix,int r, int c, int ro, int co, int n,int nCores,int posIni, int numTh,double *memaux,int blockSize){

    #pragma omp barrier            
        
    DiagInversion(matrix,r,c,ro,co,n,nCores,posIni,numTh);
        
    #pragma omp barrier            
    
    int deep=log(nCores)/log(2);
    int o;    
    
    #pragma omp barrier            
    
    for (o=deep;o>=1;o--){
            
        InversionNLProducts(matrix,r,c,ro,co,n,nCores,posIni,o,numTh,memaux,blockSize);
            
        #pragma omp barrier            
            
        InversionLNProducts(matrix,r,c,ro,co,n,nCores,posIni,o,numTh,memaux,blockSize);
            
        #pragma omp barrier            
    }    

    
}

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

void DiagInversion(double *matrix,int r, int c, int ro, int co, int n,int nCores,int posIni,int numTh){
    if(nCores <= 1){        
        if(n>0){
            double *m=auxmemory1[numTh];
            getSubMatrix(matrix,r,c,ro,co,m, n,n,1);
            int info;
            char s1='L';
            char s2='N';            
            dtrtri_(&s1,&s2,&n, m, &n,&info);
            if(info != 0){printf("Error en dtrtri %d ------------------------------\n",info);}            
            putSubMatrix(matrix,r,c,ro,co,m, n,n,1);            
        }
    }else{
        int size1=ceil(0.5*n);
        int size2=n-size1;

        if(numTh<posIni+nCores/2)
            DiagInversion(matrix,r,c,ro,co,size1,nCores/2,posIni,numTh);
        else
            DiagInversion(matrix,r,c,ro+size1,co+size1, size2,nCores/2,posIni+nCores/2,numTh);
    }    
    
}

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

void InversionNLProducts(double *matrix,int r, int c, int ro, int co, int n,int nCores,int posIni,int deep,int numTh, double *memaux, int blockSize){

        int size1=ceil(0.5*n);
        int size2=n-size1;
        
        if(deep==1){
            double *C1=&memaux[posIni*blockSize];
            NLProduct(matrix,r,ro,c,co,matrix,r,ro+size1,c,co,size1,size2,-1.0,C1,size2,0,size1,0, nCores,posIni,numTh);
        }else{
            if(numTh<posIni+nCores/2)
                InversionNLProducts(matrix,r,c,ro,co,size1,nCores/2,posIni,deep-1,numTh,memaux,blockSize);
            else
                InversionNLProducts(matrix,r,c,ro+size1,co+size1,size2,nCores/2,posIni+nCores/2,deep-1,numTh,memaux,blockSize);
        }    

}

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

void InversionLNProducts(double *matrix,int r, int c, int ro, int co, int n,int nCores,int posIni,int deep,int numTh, double *memaux, int blockSize){

        int size1=ceil(0.5*n);
        int size2=n-size1;
        
        if(deep==1){
            double *C1=&memaux[posIni*blockSize];
            LNProduct(matrix,r,ro+size1,c,co+size1,C1,size2,0,size1,0,size2,size1,1.0,matrix,r,ro+size1,c,co, nCores,posIni,numTh);
        }else{
            if(numTh<posIni+nCores/2)
                InversionLNProducts(matrix,r,c,ro,co,size1,nCores/2,posIni,deep-1,numTh,memaux,blockSize);
            else
                InversionLNProducts(matrix,r,c,ro+size1,co+size1,size2,nCores/2,posIni+nCores/2,deep-1,numTh,memaux,blockSize);
        }    

}

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

void NNProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,int n3,double K1,double K2,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh,int orientation){
    if(nCores <= 1){
        if(n1 >0 & n3>0){            
            double *mresultT=auxmemory2[numTh];            
            getSubMatrix(result,rr,cr,ror,cor,mresultT, n1,n3,nCores);
            
            if(n2 > 0){
                                            
                double *m1T=auxmemory1[numTh];
                double *m2T=auxmemory3[numTh];

                getSubMatrix(m1,r1,c1,ro1,co1,m1T, n1,n2,nCores);                
                getSubMatrix(m2,r2,c2,ro2,co2,m2T, n2,n3,nCores);   

                char transN = 'N';
                char transY = 'N';
  
                dgemm_(&transN, &transY, &(n1), &(n3), &(n2),&(K1), m1T, &(n1), m2T, &(n2), &(K2), mresultT, &(n1));                     
                //cblas_dgemm (CblasColMajor, CblasNoTrans, CblasNoTrans, n1,n3,n2,K1, m1T, n1, m2T, n2, K2, mresultT, n1);                
                
            }else{
                int i;
                for(i=0;i<n2*n3;i++) mresultT[i]=K2*mresultT[i];                
            }        
            putSubMatrix(result,rr,cr,ror,cor,mresultT, n1,n3,nCores);
            //free(mresultT);
        }
        
    }else{
        int rows1A=ceil(0.5*n1);
        int rows1B=n1-rows1A;
        int cols1A=ceil(0.5*n2);
        int cols1B=n2-cols1A;
        int rows2A=ceil(0.5*n2);
        int rows2B=n2-rows2A;
        int cols2A=ceil(0.5*n3);
        int cols2B=n3-cols2A;                



        if(numTh<posIni+nCores/2){
                NNProduct(m1,r1,ro1,c1,co1,m2,r2,ro2,c2,co2,rows1A,cols1A,cols2A,K1,K2,result,rr,ror,cr,cor, nCores/2,posIni,numTh,orientation);            
                NNProduct(m1,r1,ro1,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2,rows1A,cols1B,cols2A,K1,1.0,result,rr,ror,cr,cor, nCores/2,posIni,numTh,orientation);

                if(orientation==1){
                    NNProduct(m1,r1,ro1+rows1A,c1,co1,m2,r2,ro2,c2,co2,rows1B,cols1A,cols2A,K1,K2,result,rr,ror+rows1A,cr,cor,nCores/2,posIni,numTh,orientation);
                    NNProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2,rows1B,cols1B,cols2A,K1,1.0,result,rr,ror+rows1A,cr,cor, nCores/2,posIni,numTh,orientation);                
                }else{
                    NNProduct(m1,r1,ro1,c1,co1,m2,r2,ro2,c2,co2+cols2A,rows1A,cols1A,cols2B,K1,K2,result,rr,ror,cr,cor+cols2A, nCores/2,posIni,numTh,orientation);
                    NNProduct(m1,r1,ro1,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2+cols2A,rows1A,cols1B,cols2B,K1,1.0,result,rr,ror,cr,cor+cols2A, nCores/2,posIni,numTh,orientation);
                }
                                
        }else{
                if(orientation==2){
                    NNProduct(m1,r1,ro1+rows1A,c1,co1,m2,r2,ro2,c2,co2,rows1B,cols1A,cols2A,K1,K2,result,rr,ror+rows1A,cr,cor,nCores/2,posIni+nCores/2,numTh,orientation);
                    NNProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2,rows1B,cols1B,cols2A,K1,1.0,result,rr,ror+rows1A,cr,cor, nCores/2,posIni+nCores/2,numTh,orientation);                
                }else{
                    NNProduct(m1,r1,ro1,c1,co1,m2,r2,ro2,c2,co2+cols2A,rows1A,cols1A,cols2B,K1,K2,result,rr,ror,cr,cor+cols2A, nCores/2,posIni+nCores/2,numTh,orientation);
                    NNProduct(m1,r1,ro1,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2+cols2A,rows1A,cols1B,cols2B,K1,1.0,result,rr,ror,cr,cor+cols2A, nCores/2,posIni+nCores/2,numTh,orientation);
                }
                
                NNProduct(m1,r1,ro1+rows1A,c1,co1,m2,r2,ro2,c2,co2+cols2A,rows1B,cols1A,cols2B,K1,K2,result,rr,ror+rows1A,cr,cor+cols2A, nCores/2,posIni+nCores/2,numTh,orientation);
                NNProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2+cols2A,rows1B,cols1B,cols2B,K1,1.0,result,rr,ror+rows1A,cr,cor+cols2A, nCores/2,posIni+nCores/2,numTh,orientation);
            
        }
    }
}


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

void MoveMatrix(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2, int n1, int n2, int nCores,int posIni,int numTh){
    if(nCores <= 1){
        if(n1 >0 & n2>0){            
            double *mmv=auxmemory2[numTh];
            getSubMatrix(m1,r1,c1,ro1,co1,mmv, n1,n2,nCores);
            putSubMatrix(m2,r2,c2,ro2,co2,mmv, n1,n2,nCores);
        }        
    }else{
        int rows1A=ceil(0.5*n1);
        int rows1B=n1-rows1A;
        int cols1A=ceil(0.5*n2);
        int cols1B=n2-cols1A;

        if(numTh<posIni+nCores/2){
                MoveMatrix(m1,r1,ro1,c1,co1,m2,r2,ro2,c2,co2,rows1A,cols1A,nCores/2,posIni,numTh);            
                MoveMatrix(m1,r1,ro1+rows1A,c1,co1,m2,r2,ro2+rows1A,c2,co2,rows1B,cols1A,nCores/2,posIni,numTh);                                            
        }else{
                MoveMatrix(m1,r1,ro1,c1,co1+cols1A,m2,r2,ro2,c2,co2+cols1A,rows1A,cols1B,nCores/2,posIni+nCores/2,numTh);            
                MoveMatrix(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2+rows1A,c2,co2+cols1A,rows1B,cols1B,nCores/2,posIni+nCores/2,numTh);                                            
        }
    }
}


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

void TNNProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,int n3,double K1,double K2,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh){
    if(nCores <= 1){
        if(n2 >0 & n3>0){    
                                            
            double *mresultT=auxmemory2[numTh];        
            getSubMatrix(result,rr,cr,ror,cor,mresultT, n2,n3,nCores);
            
            if(n1 >0){
                                            
                double *m1T=auxmemory1[numTh];
                double *m2T=auxmemory3[numTh];        
                getSubMatrix(m1,r1,c1,ro1,co1,m1T, n1,n2,nCores);            
                getSubMatrix(m2,r2,c2,ro2,co2,m2T, n1,n3,nCores); 
                
                char transN = 'N';
                char transY = 'N';
                dgemm_(&transN, &transY, &(n2), &(n3), &(n1),&(K1), m1T, &(n1), m2T, &(n1), &(K2), mresultT, &(n2));                                    
                //cblas_dgemm (CblasColMajor, CblasTrans, CblasNoTrans, n2,n3,n1,K1, m1T, n1, m2T, n1, K2, mresultT, n2);                

            
            }else{
                int i;
                for(i=0;i<n2*n3;i++) mresultT[i]=K2*mresultT[i];                
            }            
            putSubMatrix(result,rr,cr,ror,cor,mresultT, n2,n3,nCores);            

        }
    }else{
        int rows1A=ceil(0.5*n1);
        int rows1B=n1-rows1A;
        int cols1A=ceil(0.5*n2);
        int cols1B=n2-cols1A;
        int rows2A=ceil(0.5*n1);
        int rows2B=n1-rows1A;
        int cols2A=ceil(0.5*n3);
        int cols2B=n3-cols2A;
        
        if(numTh<posIni+nCores/2){
                TNNProduct(m1,r1,ro1,c1,co1,m2,r2,ro2,c2,co2,rows1A,cols1A,cols2A,K1,K2,result,rr,ror,cr,cor, nCores/2,posIni,numTh);
                TNNProduct(m1,r1,ro1+rows1A,c1,co1,m2,r2,ro2+rows2A,c2,co2,rows1B,cols1A,cols2A,K1,1.0,result,rr,ror,cr,cor, nCores/2,posIni,numTh);

                TNNProduct(m1,r1,ro1,c1,co1+cols1A,m2,r2,ro2,c2,co2,rows1A,cols1B,cols2A,K1,K2,result,rr,ror+cols1A,cr,cor,nCores/2,posIni,numTh);
                TNNProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2,rows1B,cols1B,cols2A,K1,1.0,result,rr,ror+cols1A,cr,cor, nCores/2,posIni,numTh);
        }else{
                TNNProduct(m1,r1,ro1,c1,co1,m2,r2,ro2,c2,co2+cols2A,rows1A,cols1A,cols2B,K1,K2,result,rr,ror,cr,cor+cols2A, nCores/2,posIni+nCores/2,numTh);
                TNNProduct(m1,r1,ro1+rows1A,c1,co1,m2,r2,ro2+rows2A,c2,co2+cols2A,rows1B,cols1A,cols2B,K1,1.0,result,rr,ror,cr,cor+cols2A, nCores/2,posIni+nCores/2,numTh);

                TNNProduct(m1,r1,ro1,c1,co1+cols1A,m2,r2,ro2,c2,co2+cols2A,rows1A,cols1B,cols2B,K1,K2,result,rr,ror+cols1A,cr,cor+cols2A, nCores/2,posIni+nCores/2,numTh);
                TNNProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2+cols2A,rows1B,cols1B,cols2B,K1,1.0,result,rr,ror+cols1A,cr,cor+cols2A, nCores/2,posIni+nCores/2,numTh);
            
        }                

    }
}

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

void NNTProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,int n3,double K1,double K2,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh){
    if(nCores <= 1){
        if(n1 >0 & n3>0){                                
            double *mresultT=auxmemory2[numTh];        
            getSubMatrix(result,rr,cr,ror,cor,mresultT, n1,n3,nCores);
            if(n2 > 0){
                                            
                double *m1T=auxmemory1[numTh];
                double *m2T=auxmemory3[numTh];    
                getSubMatrix(m1,r1,c1,ro1,co1,m1T, n1,n2,nCores);
                getSubMatrix(m2,r2,c2,ro2,co2,m2T, n3,n2,nCores);
                
                char transN = 'N';
                char transY = 'T';
                dgemm_(&transN, &transY, &(n1), &(n3), &(n2),&(K1), m1T, &(n1), m2T, &(n3), &(K2), mresultT, &(n1));                     
                //cblas_dgemm (CblasColMajor, CblasNoTrans, CblasTrans, n1,n3,n2,K1, m1T, n1, m2T, n3, K2, mresultT, n1);

            }else{
                int i;
                for(i=0;i<n1*n3;i++) mresultT[i]=K2*mresultT[i];
            }                        
            putSubMatrix(result,rr,cr,ror,cor,mresultT, n1,n3,nCores);
        }
    }else{

        int rows1A=ceil(0.5*n1);
        int rows1B=n1-rows1A;
        int cols1A=ceil(0.5*n2);
        int cols1B=n2-cols1A;
        int rows2A=ceil(0.5*n3);
        int rows2B=n3-rows2A;
        int cols2A=ceil(0.5*n2);
        int cols2B=n2-cols2A;
    
        if(numTh<posIni+nCores/2){
                NNTProduct(m1,r1,ro1,c1,co1,m2,r2,ro2,c2,co2,rows1A,cols1A,rows2A,K1,K2,result,rr,ror,cr,cor, nCores/2,posIni,numTh);
                NNTProduct(m1,r1,ro1,c1,co1+cols1A,m2,r2,ro2,c2,co2+cols2A,rows1A,cols1B,rows2A,K1,1.0,result,rr,ror,cr,cor, nCores/2,posIni,numTh);

                NNTProduct(m1,r1,ro1,c1,co1,m2,r2,ro2+rows2A,c2,co2,rows1A,cols1A,rows2B,K1,K2,result,rr,ror,cr,cor+rows2A, nCores/2,posIni,numTh);
                NNTProduct(m1,r1,ro1,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2+cols2A,rows1A,cols1B,rows2B,K1,1.0,result,rr,ror,cr,cor+rows2A, nCores/2,posIni,numTh);
                
        }else{                

                NNTProduct(m1,r1,ro1+rows1A,c1,co1,m2,r2,ro2+rows2A,c2,co2,rows1B,cols1A,rows2B,K1,K2,result,rr,ror+rows1A,cr,cor+rows2A, nCores/2,posIni+nCores/2,numTh);
                NNTProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2+cols2A,rows1B,cols1B,rows2B,K1,1.0,result,rr,ror+rows1A,cr,cor+rows2A, nCores/2,posIni+nCores/2,numTh);
                
                NNTProduct(m1,r1,ro1+rows1A,c1,co1,m2,r2,ro2,c2,co2,rows1B,cols1A,rows2A,K1,K2,result,rr,ror+rows1A,cr,cor,nCores/2,posIni+nCores/2,numTh);
                NNTProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2,c2,co2+cols2A,rows1B,cols1B,rows2A,K1,1.0,result,rr,ror+rows1A,cr,cor, nCores/2,posIni+nCores/2,numTh);                
            
        }            
    }
}

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

void AATProduct(double *m1,int r1,int ro1,int c1, int co1,int n1,int n2,double K1,double K2,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh){
    if(nCores <= 1){        
        if(n1 >0 & n2>0){
        double *m1T = (double *) malloc(n1*n2*sizeof(double));            
        double *mresultT = (double *) malloc(n1*n1*sizeof(double));                
        getSubMatrix(m1,r1,c1,ro1,co1,m1T, n1,n2,nCores);
        getSubMatrix(result,rr,cr,ror,cor,mresultT, n1,n1,nCores);

        char transN = 'L';
        char transY = 'N';
        dsyrk_(&transN, &transY, &(n1), &(n2), &(K1), m1T, &(n1), &(K2), mresultT, &(n1));                     
        //cblas_dsyrk (CblasColMajor, CblasLower, CblasNoTrans, n1,n2,K1, m1T, n1, K2, mresultT, n1);
        int i,j;
        for(i=0;i<n1;i++){
            for(j=i;j<n1;j++){
                mresultT[j*n1+i]=mresultT[i*n1+j];
            }
        }                
        putSubMatrix(result,rr,cr,ror,cor,mresultT, n1,n1,nCores);
        free(m1T);
        free(mresultT);
        }
    }else{
        int rows1A=ceil(0.5*n1);
        int rows1B=n1-rows1A;
        int cols1A=ceil(0.5*n2);
        int cols1B=n2-cols1A;
        
        if(numTh<posIni+nCores/2){
                AATProduct(m1,r1,ro1,c1,co1,rows1A,cols1A,K1,K2,result,rr,ror,cr,cor, nCores/2,posIni,numTh);
                AATProduct(m1,r1,ro1,c1,co1+cols1A,rows1A,cols1B,K1,1.0,result,rr,ror,cr,cor, nCores/2,posIni,numTh);
        }else{
                AATProduct(m1,r1,ro1+rows1A,c1,co1,rows1B,cols1A,K1,K2,result,rr,ror+rows1A,cr,cor+rows1A, nCores/2,posIni+nCores/2,numTh);
                AATProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,rows1B,cols1B,K1,1.0,result,rr,ror+rows1A,cr,cor+rows1A, nCores/2,posIni+nCores/2,numTh);
            
        }                
        NNTProduct(m1,r1,ro1+rows1A,c1,co1,m1,r1,ro1,c1,co1,rows1B,cols1A,rows1A,K1,K2,result,rr,ror+rows1A,cr,cor,nCores,posIni,numTh);
        NNTProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m1,r1,ro1,c1,co1+cols1A,rows1B,cols1B,rows1A,K1,1.0,result,rr,ror+rows1A,cr,cor, nCores,posIni,numTh);
        
    }
}

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

void LNProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,double K1,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh){
    if(nCores <= 1){
        if(n1 >0 & n2>0){
            double *m1T=auxmemory2[numTh];
            double *mresultT=auxmemory1[numTh];
            getSubMatrix(m1,r1,c1,ro1,co1,m1T, n1,n1,nCores);        
            getSubMatrix(m2,r2,c2,ro2,co2,mresultT, n1,n2,nCores); 
            char side = 'L';
            char uplo = 'L';
            char transa = 'N';
            char diag = 'N';
            dtrmm_(&side, &uplo, &transa, &diag, &(n1), &(n2), &(K1), m1T, &(n1), mresultT, &(n1));
            //cblas_dtrmm (CblasColMajor, CblasLeft,CblasLower,CblasNoTrans,CblasNonUnit,n1,n2,K1, m1T, n1, mresultT, n1);
            putSubMatrix(result,rr,cr,ror,cor,mresultT, n1,n2,nCores);
         }
    }else{
        
        int rows1A=ceil(0.5*n1);
        int rows1B=n1-rows1A;
        int cols1A=ceil(0.5*n1);
        int cols1B=n1-cols1A;
        int rows2A=ceil(0.5*n1);
        int rows2B=n1-cols1A;        
        int cols2A=ceil(0.5*n2);
        int cols2B=n2-cols2A;
        
        
        if(numTh<posIni+nCores/2){            
                LNProduct(m1,r1,ro1,c1,co1,m2,r2,ro2,c2,co2,rows1A,cols2A,K1,result,rr,ror,cr,cor, nCores/2,posIni,numTh);
                
                LNProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2,rows1B,cols2A,K1,result,rr,ror+rows1A,cr,cor, nCores/2,posIni,numTh);
                NNProduct(m1,r1,ro1+rows1A,c1,co1,m2,r2,ro2,c2,co2,rows1B,cols1A,cols2A,K1,1.0,result,rr,ror+rows1A,cr,cor,nCores/2,posIni,numTh,1);
        }else{
                LNProduct(m1,r1,ro1,c1,co1,m2,r2,ro2,c2,co2+cols2A,rows1A,cols2B,K1,result,rr,ror,cr,cor+cols2A, nCores/2,posIni+nCores/2,numTh);
                
                LNProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2+cols2A,rows1B,cols2B,K1,result,rr,ror+rows1A,cr,cor+cols2A, nCores/2,posIni+nCores/2,numTh);
                NNProduct(m1,r1,ro1+rows1A,c1,co1,m2,r2,ro2,c2,co2+cols2A,rows1B,cols1A,cols2B,K1,1,result,rr,ror+rows1A,cr,cor+cols2A, nCores/2,posIni+nCores/2,numTh,1);
            
        }        
            
    }
}

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


void LTNProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,double K1,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh){    
    if(nCores <= 1){
        if(n1 >0 & n2>0){
            int in;
            double *m1T=auxmemory1[numTh];
            double *m2T=auxmemory2[numTh];
            getSubMatrix(m1,r1,c1,ro1,co1,m1T, n1,n1,1);
            getSubMatrix(result,rr,cr,ror,cor,m2T, n1,n2,1);
            char side = 'L';
            char uplo = 'L';
            char transa = 'T';
            char diag = 'N';
            dtrmm_(&side, &uplo, &transa, &diag, &(n1), &(n2), &(K1), m1T, &(n1), m2T, &(n1));
            //cblas_dtrmm (CblasColMajor, CblasLeft,CblasLower,CblasTrans,CblasNonUnit,n1,n2,K1, m1T, n1, m2T, n1);
            putSubMatrix(result,rr,cr,ror,cor,m2T, n1,n2,1);
        }
    }else{
        int rows1A=ceil(0.5*n1);
        int rows1B=n1-rows1A;
        int cols1A=ceil(0.5*n1);
        int cols1B=n1-cols1A;
        int rows2A=ceil(0.5*n1);
        int rows2B=n1-cols1A;        
        int cols2A=ceil(0.5*n2);
        int cols2B=n2-cols2A;
        
                        
        if(numTh<posIni+nCores/2){                            
                LTNProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2,rows1B,cols2A,K1,result,rr,ror+cols1A,cr,cor, nCores/2,posIni,numTh);                
                
                LTNProduct(m1,r1,ro1,c1,co1,m2,r2,ro2,c2,co2,rows1A,cols2A,K1,result,rr,ror,cr,cor, nCores/2,posIni,numTh);
                TNNProduct(m1,r1,ro1+rows1A,c1,co1,m2,r2,ro2+rows2A,c2,co2,rows1B,cols1A,cols2A,K1,1.0,result,rr,ror,cr,cor,nCores/2,posIni,numTh);            

        }else{    
                LTNProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2+cols2A,rows1B,cols2B,K1,result,rr,ror+cols1A,cr,cor+cols2A, nCores/2,posIni+nCores/2,numTh);
                
                LTNProduct(m1,r1,ro1,c1,co1,m2,r2,ro2,c2,co2+cols2A,rows1A,cols2B,K1,result,rr,ror,cr,cor+cols2A, nCores/2,posIni+nCores/2,numTh);            
                TNNProduct(m1,r1,ro1+rows1A,c1,co1,m2,r2,ro2+rows2A,c2,co2+cols2A,rows1B,cols1A,cols2B,K1,1.0,result,rr,ror,cr,cor+cols2A,nCores/2,posIni+nCores/2,numTh);                                    
                

        }                
    }
}

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

void NLProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,double K1,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh){
    if(nCores <= 1){
        if(n1 >0 & n2>0){
            double *m1T=auxmemory1[numTh];
            double *mresultT=auxmemory2[numTh];
            getSubMatrix(m1,r1,c1,ro1,co1,m1T, n1,n1,nCores);        
            getSubMatrix(m2,r2,c2,ro2,co2,mresultT, n2,n1,nCores);
            char side = 'R';
            char uplo = 'L';
            char transa = 'N';
            char diag = 'N';
            
            dtrmm_(&side, &uplo, &transa, &diag, &(n2), &(n1), &(K1), m1T, &(n1), mresultT, &(n2));
            //cblas_dtrmm (CblasColMajor, CblasRight,CblasLower,CblasNoTrans,CblasNonUnit,n2,n1,K1, m1T, n1, mresultT, n2);
            putSubMatrix(result,rr,cr,ror,cor,mresultT, n2,n1,nCores);
        }
    }else{

        int rows1A=ceil(0.5*n1);
        int rows1B=n1-rows1A;
        int cols1A=ceil(0.5*n1);
        int cols1B=n1-cols1A;
        int rows2A=ceil(0.5*n2);
        int rows2B=n2-rows2A;        
        int cols2A=ceil(0.5*n1);
        int cols2B=n1-cols2A;
        
        
        if(numTh<posIni+nCores/2){
                NLProduct(m1,r1,ro1,c1,co1,m2,r2,ro2,c2,co2,rows1A,rows2A,K1,result,rr,ror,cr,cor, nCores/2,posIni,numTh);
                NNProduct(m2,r2,ro2,c2,co2+cols2A,m1,r1,ro1+rows1A,c1,co1,rows2A,cols2B,cols1A,K1,1,result,rr,ror,cr,cor, nCores/2,posIni,numTh,2);
                
                NLProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2,c2,co2+cols2A,rows1B,rows2A,K1,result,rr,ror,cr,cor+cols1A, nCores/2,posIni,numTh);                
        }else{
                NLProduct(m1,r1,ro1,c1,co1,m2,r2,ro2+rows2A,c2,co2,rows1A,rows2B,K1,result,rr,ror+rows2A,cr,cor,nCores/2,posIni+nCores/2,numTh);
              NNProduct(m2,r2,ro2+rows2A,c2,co2+cols2A,m1,r1,ro1+rows1A,c1,co1,rows2B,cols2B,cols1A,K1,1,result,rr,ror+rows2A,cr,cor, nCores/2,posIni+nCores/2,numTh,2);                        
                
                NLProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2+cols2A,rows1B,rows2B,K1,result,rr,ror+rows2A,cr,cor+cols1A, nCores/2,posIni+nCores/2,numTh);            
        }                        

    }
}

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

void NLTProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,double K1,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh){
    if(nCores <= 1){
        if(n1 >0 & n2>0){
            double *m1T=auxmemory1[numTh];
            double *m2T=auxmemory2[numTh];
            getSubMatrix(m1,r1,c1,ro1,co1,m1T, n1,n1,nCores);        
            getSubMatrix(m2,r2,c2,ro2,co2,m2T, n2,n1,nCores);
            char side = 'R';
            char uplo = 'L';
            char transa = 'T';
            char diag = 'N';
            
            dtrmm_(&side, &uplo, &transa, &diag, &(n2), &(n1), &(K1), m1T, &(n1), m2T, &(n2));
            //cblas_dtrmm (CblasColMajor, CblasRight,CblasLower,CblasTrans,CblasNonUnit,n2,n1,K1, m1T, n1, m2T, n2);
            putSubMatrix(result,rr,cr,ror,cor,m2T, n2,n1,nCores);
        }
    }else{
        int rows1A=ceil(0.5*n1);
        int rows1B=n1-rows1A;
        int cols1A=ceil(0.5*n1);
        int cols1B=n1-cols1A;
        int rows2A=ceil(0.5*n2);
        int rows2B=n2-rows2A;
        int cols2A=ceil(0.5*n1);
        int cols2B=n1-cols2A;

        if(numTh<posIni+nCores/2){
                NLTProduct(m1,r1,ro1,c1,co1,m2,r2,ro2,c2,co2,rows1A,rows2A,K1,result,rr,ror,cr,cor,nCores/2,posIni,numTh);
                
                NLTProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2,c2,co2+cols2A,rows1B,rows2A,K1,result,rr,ror,cr,cor+rows1A,nCores/2,posIni,numTh);
                NNTProduct(m2,r2,ro2,c2,co2,m1,r1,ro1+rows1A,c1,co1,rows2A,cols2A,rows1B,K1,1.0,result,rr,ror,cr,cor+rows1A,nCores/2,posIni,numTh);


        }else{            

                NLTProduct(m1,r1,ro1,c1,co1,m2,r2,ro2+rows2A,c2,co2,rows1A,rows2B,K1,result,rr,ror+rows2A,cr,cor, nCores/2,posIni+nCores/2,numTh);        

                NLTProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2+cols2A,rows1B,rows2B,K1,result,rr,ror+rows2A,cr,cor+rows1A, nCores/2,posIni+nCores/2,numTh);                
                NNTProduct(m2,r2,ro2+rows2A,c2,co2,m1,r1,ro1+rows1A,c1,co1,rows2B,cols2A,rows1B,K1,1.0,result,rr,ror+rows2A,cr,cor+rows1A, nCores/2,posIni+nCores/2,numTh);                                        
                                    
                
            
        }        
        
        
        
    }
}

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

void ParallelVectorMatrixT(double *m1,int size,double *m2,double *result, int numThreads){
    int i;
    char transN = 'N';
    char transY = 'T';
    int aux=1;
    double aux2=0;
    double factor=1.0;
    #pragma omp parallel default(shared) private(i)
    {                
    #pragma omp for schedule(static)    
    for (i=0;i<numThreads;i++){
        int InitCol=round(i*size/numThreads);
        int FinalCol=round((i+1)*size/numThreads)-1;            
        int lenghtCol=FinalCol-InitCol+1;
        if(lenghtCol>0){
            dgemm_(&transN, &transN, &(lenghtCol), &(aux), &(size),&factor, &m2[InitCol], &(size), m1, &size, &aux2, &result[InitCol], &(size));
        }
    }
    }
}

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

void ParallelVectorMatrix(double *m1,int size,double *m2,double *result, int numThreads){
    int i;
    char transN = 'N';
    char transY = 'N';
    int aux=1;
    double aux2=0;
    double factor=1.0;
    #pragma omp parallel default(shared) private(i)
    {                
    #pragma omp for schedule(static)    
    for (i=0;i<numThreads;i++){
        int InitCol=round(i*size/numThreads);
        int FinalCol=round((i+1)*size/numThreads)-1;            
        int lenghtCol=FinalCol-InitCol+1;
        if(lenghtCol>0){
            dgemm_(&transN, &transY, &aux, &(lenghtCol), &(size),&factor, m1, &aux, &m2[size*InitCol], &size, &aux2, &result[InitCol], &aux);
        }
    }
    }    
}

/**
 * @endcond
 */

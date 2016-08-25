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
 * @file PSIRWLS-train.h
 * @author Roberto Diaz Morales
 * @date 23 Aug 2016
 * @brief Functions to train a semi parametric SVM using the IRWLS algorithm.
 *
 * Functions to train a semi parametric SVM using the IRWLS algorithm.
 */


#ifndef PSIRWLSTRAIN_
#define PSIRWLSTRAIN_

#include "IOStructures.h"

/**
 * @brief Sparse Greedy Matrix Approximation algorithm
 *
 * Sparse Greedy Matrix Approximation algorithm to select the basis elements of the semi parametric model.
 */

int* SGMA(svm_dataset dataset,properties props);

/**
 * @brief Iterative Re-Weighted Least Squares Algorithm.
 *
 * IRWLS procedure to obtain the weights of the semi parametric model.
 */

double* IRWLSpar(svm_dataset dataset, int* indexes,properties props);



/**
 * @brief It shows PSIRWLS-train command line instructions in the standard output.
 *
 *  It shows PSIRWLS-train command line instructions in the standard output.
 */

void printPSIRWLSInstructions() ;


/**
 * @brief It parses input command line to extract the parameters.
 *
 * It parses input command line to extract the parameters.
 */

properties parseTrainParameters(int* argc, char*** argv);

/**
 * @brief It converts the result into a model strut.
 *
 * It converts the result into a model strut.
 */

model calculatePSIRWLSModel(properties props, svm_dataset dataset, int *centroids, double * beta );

#endif



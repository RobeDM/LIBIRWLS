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
 * @file kernels.h
 * @author Roberto Diaz Morales
 * @date 23 Aug 2016
 * @brief kernel function of non linear SVM.
 *
 * Implements the kernel function to use in the non linear SVM.
 */


#ifndef KERNELS_
#define KERNELS_


/**
 * @brief Kernel function among two elements of the dataset.
 *
 * This function returns the kernel function among two elements of the dataset.
 */

double kernel(svm_dataset dataset, int index1, int index2, properties props);

/**
 * @brief Kernel function among two elements of two different datasets.
 *
 * This function returns the kernel function among two elements of two different datasets.
 */

double kernelTest(svm_dataset dataset, int index1, model mymodel, int index2);


#endif


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
 * @brief Defition of the kernel functions used in the non linear SVM.
 *
 * It defines the kernel function to use in the non linear SVM in this library.
 */


#ifndef KERNELS_
#define KERNELS_

#include "IOStructures.h"

/**
 * @brief Radial Basis Function of two elements of the dataset.
 *
 * This function returns the kernel function among two elements of the same dataset.
 *
 * It returns exp(-gamma||x1-x2||^2)
 *
 * x1 and x2 are two elements of the dataset and gamma is a parameter whose value can be found
 * in the struct props. 
 *
 * @param dataset The strut that contains the dataset information.
 * @param index1 The index of the first element of the dataset.
 * @param index2 The index of the second element of the dataset.
 * @param props The list of properties to extract the kernel parameters.
 * @return The value of the Radial Basis Function of both elements.
 */

double kernelFunction(svm_dataset dataset, int index1, int index2, properties props);

/**
 * @brief Radial Basis Function of one element of the dataset and Support Vectro of a trained model.
 *
 * This method returns the RBF Kernel function of one element of the dataset and Support Vectro of a trained model.
 * 
 * It returns exp(-gamma||x1-x2||^2)
 *
 * x1 is an element of the dataset and x2 is a support vector of a trained model, gamma is a parameter whose value can be found
 * in the struct props. 
 *
 * @param dataset The strut that contains the dataset information.
 * @param index1 The index of the sample of the dataset.
 * @param mymodel The trained SVM model.
 * @param index2 The index of one of the Support Vectors of the trained model.
 * @return The value of the Radial Basis Function of both elements.
 */

double kernelTest(svm_dataset dataset, int index1, model mymodel, int index2);


#endif


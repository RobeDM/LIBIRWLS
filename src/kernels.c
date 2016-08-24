
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

#include "../include/kernels.h”

double kernel(svm_dataset dataset, int index1, int index2, properties props){

    double sum = 0.0;   
    
    svm_sample *x=dataset.x[index1];
    svm_sample *y=dataset.x[index2];

    if (dataset.sparse==0){
        int maxdim = dataset.maxdim;
        int i;
        for(i=0;i<maxdim;i++){
	    sum+=pow((x->value) - (y->value),2);
	    ++x;
	    ++y;
	}

    }else{

        sum += (dataset.quadratic_value[index1])+(dataset.quadratic_value[index2]);
        while(x->index !=-1 && y->index !=-1) {
            if(x->index == y->index){
                sum += -2.0 * (x->value) * (y->value);
                ++x;
                ++y;
            }else{
                if((x->index) < (y->index)){
                    ++x;
                }else{
                    ++y;
                }
            }
        }
    }

    return exp(-(props.Kgamma)*sum);
}



double kernelTest(svm_dataset dataset, int index1, model mymodel, int index2){

    double sum = 0.0;

    svm_sample *x=dataset.x[index1];
    svm_sample *y=mymodel.x[index2];

    if (mymodel.sparse==0){
        int maxdim = mymodel.maxdim;
        int i;
        for(i=0;i<maxdim;i++){
            sum+=pow((x->value)-(y->value) ,2);
            ++x;
            ++y;
        }
    }else{

        sum += (dataset.quadratic_value[index1])+(mymodel.quadratic_value[index2]);
        while(x->index !=-1 && y->index !=-1) {
            if(x->index == y->index){
                sum += -2.0 * (x->value) * (y->value);
                ++x;
                ++y;
            }else{
                if((x->index) < (y->index)){
                    ++x;
                }else{
                    ++y;
                }
            }
        }
    }

    return exp(-(mymodel.Kgamma)*sum);
}


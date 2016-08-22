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


#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include<sys/time.h>
#include <string.h>


#ifdef USE_MKL
#include "mkl_cblas.h"
#include "mkl_blas.h"
#include "mkl.h"
#else
#include "cblas.h"
#endif

#include "./lib/tools.c"
#include "./lib/kernels.c"



double *test(svm_dataset dataset, model mymodel,predictProperties props){

    int i,j;		
    double *predictions=(double *) malloc((dataset.l)*sizeof(double));

    #pragma omp parallel default(shared) private(i,j)
    {	
    #pragma omp for schedule(static)			
    for (i=0;i<dataset.l;i++){
        double pred=mymodel.bias;
        for (j=0;j<mymodel.nSVs;j++){
            pred+=(mymodel.weights[j])*kernelTest(dataset, i,  mymodel, j);
        }
        predictions[i]=pred;
    }	
    }

    double aciertos=0.0;
    double total=(double)dataset.l;
    if(props.Labels==1){
        for (i=0;i<dataset.l;i++){
            if(predictions[i]>0 & dataset.y[i]>0) aciertos++;
            if(predictions[i]<=0 & dataset.y[i]<=0) aciertos++;
        }
        printf("Accuracy: %f\n",aciertos/total);
    }		
    return predictions;
}


void writeOutput (char fileoutput[], double *predictions, int size){
	
     FILE *Archivo;
     Archivo = fopen(fileoutput,"w+"); 
         
     if(Archivo !=0){
          int i;
          for(i=0;i<size;i++){
      	      fprintf(Archivo,"%lf\n",predictions[i]);
          }
     }
     fclose(Archivo);
}

int main(int argc, char** argv)
{

    predictProperties props = PredictParameters(&argc, &argv);
  
    if (argc != 4) {
        printInstructionsPredict();
        return 4;
    }

    char * data_file = argv[1];
    char * data_model = argv[2];
    char * output_file = argv[3];
  
    model  mymodel;
    
    //////////////////////////////
    FILE *In = fopen(data_model, "r+");
    readModel(&mymodel, In);
    fclose(In);

    printf("\nModel Loaded from file: %s\nSupport Vectors: %d\n\n",data_model,mymodel.nSVs);

    svm_dataset dataset;
	  
    if(props.Labels==0){
        dataset=readUnlabeledFile(data_file);
    }else{
        dataset=readTrainFile(data_file);			
    }

    printf("Dataset Loaded from file: %s\nTraining samples: %d\nNumber of features: %d\n\n",data_file, dataset.l,dataset.maxdim);

    omp_set_num_threads(props.Threads);
	
    double *predictions=test(dataset,mymodel,props);
	
    printf("\nWriting output in file: %s \n\n",output_file);
    writeOutput (output_file, predictions,dataset.l);
    return 0;

}

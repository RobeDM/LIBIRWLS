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

#include "./lib/ParallelAlgorithms.h"
#include "./lib/ParallelAlgorithms.c"
#include "./lib/tools.c"
#include "./lib/kernels.c"


int randomNum;

void ParallelVectorMatrixT(double *m1,int size,double *m2,double *result, properties props){
    int i;
    char transN = 'N';
    char transY = 'T';
    int aux=1;
    double aux2=0;
    double factor=1.0;
    #pragma omp parallel default(shared) private(i)
    {				
    #pragma omp for schedule(static)	
    for (i=0;i<props.Threads;i++){
        int InitCol=round(i*size/props.Threads);
        int FinalCol=round((i+1)*size/props.Threads)-1;			
        int lenghtCol=FinalCol-InitCol+1;
        if(lenghtCol>0){
            dgemm_(&transN, &transN, &(lenghtCol), &(aux), &(size),&factor, &m2[InitCol], &(size), m1, &size, &aux2, &result[InitCol], &(size));
        }
    }
    }
}

void ParallelVectorMatrix(double *m1,int size,double *m2,double *result, properties props){
    int i;
    char transN = 'N';
    char transY = 'N';
    int aux=1;
    double aux2=0;
    double factor=1.0;
    #pragma omp parallel default(shared) private(i)
    {				
    #pragma omp for schedule(static)	
    for (i=0;i<props.Threads;i++){
        int InitCol=round(i*size/props.Threads);
        int FinalCol=round((i+1)*size/props.Threads)-1;			
        int lenghtCol=FinalCol-InitCol+1;
        if(lenghtCol>0){
            dgemm_(&transN, &transY, &aux, &(lenghtCol), &(size),&factor, m1, &aux, &m2[size*InitCol], &size, &aux2, &result[InitCol], &aux);
        }
    }
    }	
}



int* SGMA(svm_dataset dataset,properties props){

    //TO STORE ERROR DESCENT AND SAMPLE INDEX
    double *descE=(double *) malloc(64*sizeof(double));	
    int *indexes=(int *) malloc(64*sizeof(int));
    int *centroids=(int *) malloc((props.size)*sizeof(int));

    //Memory to every thread
    double **KNC = (double **) malloc(64*sizeof(double *));	
    double **KSM = (double **) malloc(64*sizeof(double *));	
    double *eta=(double *) malloc(64*sizeof(double));	
    double *KSC = (double *) malloc((dataset.l)*(props.size)*sizeof(double));	
    double **Z = (double **) malloc(64*sizeof(double *));	

    //Cholesky decomposition and inverse
    double *iKC = (double *) calloc((props.size)*(props.size),sizeof(double));	
    double *invKC = (double *) calloc((props.size)*(props.size),sizeof(double));	
    double *iKCTmp = (double *) calloc((props.size)*(props.size),sizeof(double));	
    double *invKCTmp = (double *) calloc((props.size)*(props.size),sizeof(double));	
    double *L2 = (double *) calloc((props.size),sizeof(double));	
    double *IL2 = (double *) calloc((props.size),sizeof(double));	

    int size = 0;
    int i,e,bestBasis,ncols=1,info=1;
    double factor=-1;
    double factorA=1.0;
    char s = 'L';
    char trans = 'N';
    double *miKSM;
    double *miKNC;
    double *miZ;
    double value,L3,IL3;
    double *tmp1,*tmp2;
    int indexSample=0;

    while(size<props.size){
        if(size>1){
        #pragma omp parallel default(shared) private(i,e,miKSM,miKNC,miZ,value)
        {
        #pragma omp for schedule(static)	
        for(i=0;i<64;i++){
            indexSample=rand()%(dataset.l);
            while(dataset.y[indexSample] != ((i%2)*2.0-1)){
                indexSample=rand()%(dataset.l);
               
            }

            indexes[i]=indexSample;
            				
            if(size==2 && i>0){
                KNC[i]=(double *) malloc((props.size)*sizeof(double));
                KSM[i]=(double *) malloc((dataset.l)*sizeof(double));
                Z[i]=(double *) malloc((props.size)*sizeof(double));
            }

            miKNC=KNC[i];
            miKSM=KSM[i];
            miZ=Z[i];

            for(e=0;e<dataset.l;e++) miKSM[e]=kernel(dataset,indexes[i],e,props);            

            for(e=0;e<size;e++){
                value=kernel(dataset,indexes[i],centroids[e],props);
                miKNC[e]=value;
                miZ[e]=value;
            }

            value=1.0;
            if(size==0){
                eta[i]=value;
            }else{
                dpotrs_(&s,&size,&ncols, iKC, &size, miZ,&size,&info);
                for(e=0;e<size;e++) value = value - (miKNC[e]*miZ[e]);
                eta[i]=value;
                dgemm_(&trans, &trans, &(dataset.l), &ncols, &size,&factorA, KSC, &(dataset.l), miZ, &size, &factor, miKSM, &(dataset.l));
            }
            value=0.0;
            for(e=0;e<dataset.l;e++) value +=miKSM[e]*miKSM[e];
            if(eta[i]>0.0){
                descE[i]=(1.0/eta[i])*value;
            }else{
                descE[i]=0.0;
            }

        }
        }

        value=descE[0];
        bestBasis=0;
        for(i=1;i<64;i++){
            if(descE[i]>value){
                value=descE[i];
                bestBasis=i;
            }
        }
        centroids[size]=indexes[bestBasis];


        }else{
            if(size==0){
                KNC[0]=(double *) malloc((props.size)*sizeof(double));
                KSM[0]=(double *) malloc((dataset.l)*sizeof(double));
                Z[0]=(double *) malloc((props.size)*sizeof(double));
                centroids[size]=dataset.l;
            }else{
                centroids[size]=dataset.l+1;
                KNC[0][0]=kernel(dataset,centroids[0],centroids[1],props);
            }
            value=1.0;
            bestBasis=0;
            
            
        }
        

        printf("Best Error Descent %f, Index %d, Sample %d\n",value,centroids[size],size);

        #pragma omp parallel default(shared) private(i)
        {
        #pragma omp for schedule(static)	
        for(i=0;i<dataset.l;i++) KSC[size*(dataset.l)+i]=kernel(dataset,i,centroids[size],props);
        }

        if(size==0){
            iKCTmp[0]=pow(kernel(dataset,centroids[size],centroids[size],props)+0.000001,0.5);
            invKCTmp[0]=1.0/iKCTmp[0];
        }else{
            ParallelVectorMatrixT(KNC[bestBasis],size,invKC,L2,props);
            L3=kernel(dataset,centroids[size],centroids[size],props)+0.00001;
            for(i=0;i<size;i++) L3 = L3 - (L2[i]*L2[i]);
            L3=pow(L3,0.5);
            IL3=1.0/L3;
            ParallelVectorMatrix(L2,size,invKC,IL2,props);

            for(i=0;i<size;i++){
                for(e=0;e<size;e++){
                    iKCTmp[(i*(size+1))+e]=iKC[(i*size)+e];
                    invKCTmp[(i*(size+1))+e]=invKC[(i*size)+e];
                }                
            }
            iKCTmp[(size+1)*(size+1)-1]=L3;
            invKCTmp[(size+1)*(size+1)-1]=IL3;
            for(i=0;i<size;i++){
                iKCTmp[(i*(size+1))+size]=L2[i];            
                invKCTmp[(i*(size+1))+size]=-IL3*IL2[i];            
            }

        }
 
        tmp1=&iKC[0];
        tmp2=&invKC[0];
        iKC=&iKCTmp[0];
        invKC=&invKCTmp[0];
        iKCTmp=&tmp1[0];
        invKCTmp=&tmp2[0];
        ++size;

    }
    
    if(size>=2){
        for(i=0;i<64;i++){
    	    free(KNC[i]);
    	    free(KSM[i]);
    	    free(Z[i]);
        }
    }
    
    free(KNC);
    free(KSM);
    free(eta);
    free(Z);

    free(KSC);
    free(iKC);	
    free(invKC);	
    free(iKCTmp);	
    free(invKCTmp);	
    free(L2);	
    free(IL2);	    
    
    free(indexes);
    free(descE);	
    
    return centroids;
}



double* IRWLSpar(svm_dataset dataset, int* indexes,properties props){

    int i,j;
    double kernelvalue;


    int nSVs=dataset.l;
    double *KC=(double *) calloc(props.size*props.size,sizeof(double));
    double *KSC=(double *) calloc(dataset.l*props.size,sizeof(double));
    double *KSCA=(double *) calloc(dataset.l*props.size,sizeof(double));
    double *Da=(double *) calloc(dataset.l,sizeof(double));
    double *Day=(double *) calloc(dataset.l,sizeof(double));


    #pragma omp parallel default(shared) private(i,j)
    {
    #pragma omp for schedule(static)
    for (i=0;i<props.size;i++){
        int j=0;
        for (j=0;j<props.size;j++){
            KC[i*(props.size)+j]=kernel(dataset,indexes[i], indexes[j], props);
            if(i==j) KC[i*(props.size)+j]+=pow(10,-5);
        }
    }
    }

    double M=10000.0;

    #pragma omp parallel default(shared) private(i,j,kernelvalue)
    {
    #pragma omp for schedule(static)
    for (i=0;i<dataset.l;i++){
        Da[i]=M;
        Day[i]=dataset.y[i]*M;
        for (j=0;j<props.size;j++){
            kernelvalue=kernel(dataset,i, indexes[j], props);
            KSC[i*(props.size)+j]=kernelvalue;
            KSCA[i*(props.size)+j]=kernelvalue;
        }
    }
    }

    //Stop conditions
    int  iter=0, max_iter=500,cambios=100;
    double deltaW = 1e9, normW = 1.0;

    double *K1 = (double *) calloc(props.size*props.size,sizeof(double));
    double *K2 = (double *) calloc(props.size,sizeof(double));
    double *beta = (double *) calloc(props.size,sizeof(double));
    double *betaNew = (double *) calloc(props.size,sizeof(double));
    double *betaBest = (double *) calloc(props.size,sizeof(double));
    double *e = (double *) calloc(dataset.l,sizeof(double));
    int *indKSCA = (int *) calloc(dataset.l,sizeof(int));


    char notrans='N';
    char trans='T';
    int row = 1;
    double factor=1.0;
    double nfactor=-1.0;
    double zfactor=0.0;


    double val;

    double oldnorm=0.0;

    
    int itersSinceBestDW=0;
    double bestDW=1e9;
    
    while( (iter<max_iter) && (deltaW/normW > 1e-6) && (itersSinceBestDW<5) ){

        memcpy(K1,KC,(props.size)*(props.size)*sizeof(double));
        
        #pragma omp parallel default(shared) private(i)
        {				
        #pragma omp for schedule(static)	
        for (i=0;i<props.Threads;i++){
            int InitCol=round(i*props.size/props.Threads);
            int FinalCol=round((i+1)*props.size/props.Threads)-1;			
            int lengthCol=FinalCol-InitCol+1;
            if(lengthCol>0){
                dgemm_(&notrans, &notrans, &(lengthCol), &(row), &(nSVs), &factor, &KSCA[InitCol], &(props.size), Day, &nSVs, &zfactor, &K2[InitCol], &(props.size));
                dgemm_(&notrans, &trans, &(lengthCol), &(props.size), &(nSVs), &factor, &KSCA[InitCol], &(props.size), KSCA, &props.size, &factor, &K1[InitCol], &(props.size));
            }
        }
        }


        memset(betaNew,0.0,props.size*sizeof(double));
        ParallelLinearSystem(K1,props.size,props.size,0,0,K2,props.size,1,0,0,props.size,1,betaNew,props.size,1,0,0,props.Threads);


        deltaW=0.0;        
        normW=0.0;

        for (i=0;i<props.size;i++){
            deltaW += pow(betaNew[i]-beta[i],2);
            normW += pow(betaNew[i],2);
            beta[i]=betaNew[i];
        }


        memcpy(e,dataset.y,dataset.l*sizeof(double));

        #pragma omp parallel default(shared) private(i)
        {				
        #pragma omp for schedule(static)	
        for (i=0;i<props.Threads;i++){
            int InitCol=round(i*dataset.l/props.Threads);
            int FinalCol=round((i+1)*dataset.l/props.Threads)-1;			
            int lengthCol=FinalCol-InitCol+1;
            if(lengthCol>0){
                dgemm_(&notrans, &notrans, &(row), &(lengthCol), &(props.size), &nfactor, beta, &row, &KSC[InitCol*props.size], &props.size, &factor, &e[InitCol], &(row));
            }
        }
        }


        
        double alpha,chi;
        #pragma omp parallel default(shared) private(i,val,chi,alpha)
        {				
        #pragma omp for schedule(static)	
        for(i=0;i<dataset.l;i++){
	    
           if(e[i]*dataset.y[i]<0.0){
                Da[i]=0.0;
           }else{
           	Da[i]=1.0*props.C/(dataset.y[i]*e[i]);
	       }
	       if(Da[i]>M) Da[i]=M;
        }
        }


        nSVs=0;
        for(i=0;i<dataset.l;i++){
            if(Da[i]!=0.0){
                indKSCA[nSVs]=i;
                ++nSVs;
            }
        }


        #pragma omp parallel default(shared) private(i,j)
        {
        #pragma omp for schedule(static)
        for (i=0;i<nSVs;i++){
            for (j=0;j<props.size;j++){
                KSCA[i*(props.size)+j]=sqrt(Da[indKSCA[i]])*KSC[indKSCA[i]*(props.size)+j];
            }
            Day[i]=sqrt(Da[indKSCA[i]])*dataset.y[indKSCA[i]];
        }
        }         

        ++iter;
        printf("Iteration %d, nSVs %d, ||deltaW||^2/||W||^2=%f\n",iter,nSVs,deltaW/normW);
    
        if(iter>10 && deltaW/normW>100*oldnorm) M=M/10.0;
        oldnorm=deltaW/normW;
        
        if(deltaW/normW<bestDW){
            bestDW=deltaW/normW;
            itersSinceBestDW=0;
            memcpy(betaBest,betaNew,(props.size)*sizeof(double));
        }else{
            itersSinceBestDW+=1;
        }
    }

    return betaBest;
}

  
int main(int argc, char** argv)
{

    //srand(getpid());	
    //srand48(getpid());

    srand(0);	
    srand48(0);

    properties props = TrainParameters(&argc, &argv,1);
  
    if (argc != 3) {
        printPSIRWLSInstructions();
        return 4;
    }

    char * data_file = argv[1];
    char * data_model = argv[2];
  	

    svm_dataset dataset = readTrainFile(data_file);
    printf("\nDataset Loaded from file: %s\nTraining samples: %d\nNumber of features: %d\n\n",data_file, dataset.l,dataset.maxdim);

    struct timeval tiempo1, tiempo2;
    omp_set_num_threads(props.Threads);

    printf("Running SGMA\n");	
    gettimeofday(&tiempo1, NULL);


    initMemory(props.Threads,props.size);
    //omp_set_num_threads(1);

    int * centroids=SGMA(dataset,props);

    omp_set_num_threads(props.Threads);

    double * W = IRWLSpar(dataset,centroids,props);
	


    gettimeofday(&tiempo2, NULL);
    printf("Weights calculated in %ld\n\n",((tiempo2.tv_sec-tiempo1.tv_sec)*1000+(tiempo2.tv_usec-tiempo1.tv_usec)/1000));

    model modelo = calculatePSIRWLSModel(props, dataset,centroids, W);

    printf("Saving model in file: %s\n\n",data_model);	
 
    FILE *Out = fopen(data_model, "w+");
    storeModel(&modelo, Out);
    fclose(Out);

    return 0;
}

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

int * rpermute(int n) {
    int *a = (int *) malloc(n*sizeof(int));
    int k;
    for (k = 0; k < n; k++)
	a[k] = k;
        for (k = n-1; k > 0; k--) {
		int j = rand() % (k+1);
		int temp = a[j];
		a[j] = a[k];
		a[k] = temp;
        }
    return a;
}


double* subIRWLS(svm_dataset dataset,properties props, double *GIN, double *e, double *beta){
    

    //Auxiliary variables of the elements of the training set
    double *a = (double *) calloc(dataset.l,sizeof(double));
    int *elementGroup = (int *) calloc(dataset.l,sizeof(int));
    
    //Classifier weights
    double *betaNew=(double *) calloc((dataset.l+1),sizeof(double));
    double *betaAux=(double *) calloc((dataset.l+1),sizeof(double));    
    double *betaBest=(double *) calloc((dataset.l+1),sizeof(double));

    //Max y min weight
    double maxbeta=0.0;
    double minbeta=0.0;
    
    //Classes of samples
    int *S1comp = (int *) calloc((dataset.l+1),sizeof(int));
    int *S3comp = (int *) calloc((dataset.l),sizeof(int));
    
    //Stop conditions
    int  iter=0, max_iter=100;
    double deltaW = 1e9, normW = 1.0, errW=0.0;
    int itersSinceBestDW=0;
    double bestDW=1e9;
    //Variables to iterate
    int i, o, ind=0, ind2=0,nS1=0, nS3=0, thLS=0;
    
    //Variables for least square problems
    double *H   = (double *) calloc((dataset.l+1)*(dataset.l+1),sizeof(double));
    double *et  = (double *) calloc((dataset.l+1),sizeof(double));
    double *G13 = (double *) calloc((dataset.l+1),sizeof(double));    
    
    //Initialization

    for (i=0;i<dataset.l;i++){
        
        if(e[i]*dataset.y[i]<0){
            a[i]=0.0;
        }else{
            a[i]=1.0*dataset.y[i]*((double)props.C)/e[i];
        }
        
        if(a[i]==0){
            elementGroup[i]=2;
        }else if(beta[i]==dataset.y[i]*((double)props.C)){
            S3comp[ind2]=i;
            ind2++;
            nS3++;
            elementGroup[i]=3;
        }else{
            S1comp[ind]=i;
            ind++;
            nS1++;
            elementGroup[i] = 1;
        }
        
    }

    S1comp[nS1]=dataset.l;


    while(((iter<5) || ((minbeta<0.0) || (maxbeta>((double)props.C)) ) && (iter<1000) && (itersSinceBestDW<5) && (deltaW/normW > 1e-6)) ){
        
        iter++;
        
        ///////////////////////////////////////////////////////
        //GENERATING MATRIX H and VECTOR FOR THE LINEAR SYSTEM
        ///////////////////////////////////////////////////////
        memset(betaAux,0.0,(nS1+1)*sizeof(double));        
        memset(et,0.0,(nS1+1)*sizeof(double));        
        memset(H,0.0,(nS1+1)*(nS1+1)*sizeof(double));        
        
        #pragma omp parallel default(shared) private(i)
        {
        #pragma omp for schedule(static)
            for (i=0;i<nS1;i++){
                int j;
                H[i*(nS1+1)+nS1]=dataset.y[S1comp[i]];
                H[nS1*(nS1+1)+i]=dataset.y[S1comp[i]];
                et[i]=1.0-G13[i]-GIN[S1comp[i]];
                for (j=0;j<nS1;j++){
                    H[i*(nS1+1)+j]=kernel(dataset,S1comp[i], S1comp[j], props)*dataset.y[S1comp[i]]*dataset.y[S1comp[j]];
                    if(i==j) H[i*(nS1+1)+j]+=(1.0/(a[S1comp[i]]));
                }
            }
        }


        H[nS1*(nS1+1)+(nS1)]=0.0;
        et[nS1]=-G13[nS1]-GIN[dataset.l];
       
       
        ///////////////////////////////////////////////////////
        //SOLVING THE LINEAR SYSTEM
        ///////////////////////////////////////////////////////
        thLS=pow(2,floor(log(props.Threads)/log(2.0)));
        if(nS1<thLS) thLS=pow(2,floor(log(nS1)/log(2.0)));
        if(thLS<1) thLS=1;
        
        omp_set_num_threads(thLS);
        ParallelLinearSystem(H,(nS1+1),(nS1+1),0,0,et,(nS1+1),1,0,0,(nS1+1),1,betaAux,(nS1+1),1,0,0,thLS);
        omp_set_num_threads(props.Threads);

        ///////////////////////////////////////////////////////
        //UPDATING SVM WEIGHTS
        ///////////////////////////////////////////////////////
        
        deltaW=0.0;
        normW=0.0;

        maxbeta=0.0;
        minbeta=0.0;

        
        memset(betaNew,0.0,(dataset.l+1)*sizeof(double));
        

        for (i=0;i<nS1;i++){
        	if (betaAux[i]> maxbeta) maxbeta=betaAux[i];
                if (betaAux[i]< minbeta) minbeta=betaAux[i];

                betaNew[S1comp[i]]=betaAux[i]*dataset.y[S1comp[i]];
        }
        
        for (i=0;i<nS3;i++){
            betaNew[S3comp[i]]=((double)props.C)*dataset.y[S3comp[i]];
        }
        
        betaNew[dataset.l]=betaAux[nS1];
        
        for (i=0;i<dataset.l+1;i++){        	
            deltaW+=pow(betaNew[i]-beta[i],2);
            normW+=pow(beta[i],2);
            
        }
        
        

        ////////////////////////////////////////
        //UPDATING THE ERROR OF THE TRAINING SET
        ////////////////////////////////////////
        
        #pragma omp parallel default(shared) private(i,randomNum)
        {
        #pragma omp for schedule(static)
            for (i=0;i<dataset.l;i++){
                int j;
                for (j=0;j<dataset.l;j++){
                    if(betaNew[j] != beta[j]){
                        e[i]=e[i]-kernel(dataset,i,j,props)*(betaNew[j]-beta[j]);
                    }
                }
                e[i]=e[i]-(betaNew[dataset.l]-beta[dataset.l]);

            }
        }
        
        if(deltaW/normW<bestDW){
            bestDW=deltaW/normW;
            itersSinceBestDW=0;
            memcpy(betaBest,betaNew,(dataset.l+1)*sizeof(double));
        }else{
            itersSinceBestDW+=1;
        }

        
        /////////////////////////
        //ADDING EVERY DATA TO ITS GROUP
        /////////////////////////
        

        #pragma omp parallel default(shared) private(i,randomNum)
        {
        #pragma omp for schedule(static)
            for (i=0;i<dataset.l;i++){
                
                
                if(e[i]*dataset.y[i]<0.0 ){
                    a[i]=0.0;                    
                }else if(e[i]*dataset.y[i]<(1.0/10000)){
                    a[i]=((double)props.C)*10000.0;
                }else{
                    a[i]=1.0*dataset.y[i]*((double)props.C)/e[i];
                }

                
                if(e[i]*dataset.y[i]<0.0 && elementGroup[i] != 2){
                    elementGroup[i] = 2;
                }
                
                if(elementGroup[i]==1  && dataset.y[i]*betaNew[i]>=0.99*((double)props.C) && dataset.y[i]*betaNew[i]<=1.01*((double)props.C) ){
                    elementGroup[i]=3;
                }

                
                if(a[i]==0.0 && elementGroup[i] == 1){
                    elementGroup[i] = 2;
                }
                
                if(elementGroup[i]==2 && a[i] != 0.0){
                        elementGroup[i]=1;
                }
                
                beta[i]=betaNew[i];
            }
        }
        
        beta[dataset.l]=betaNew[dataset.l];
        
        nS1=0;
        nS3=0;
        for (i=0;i<dataset.l;i++){
            if (elementGroup[i]==1) ++nS1;
            if (elementGroup[i]==3) ++nS3;
        }
        
        //printf("Iteration %d, SVs %d, bSVs %d, nonasigned %d, ||deltaW||^2/||W||^2=%f, Dnor %f , nor %f , Changes %d\n",iter,nS1,nS3,nonasigned,deltaW/normW,deltaW,normW,cambios);
        
        //////////////////
        //UPDATING H13
        /////////////////
        
        ind=0;
        ind2=0;
        for (i=0;i<dataset.l;i++){
            if (elementGroup[i]==1){
                S1comp[ind]=i;
                ind++;
            }else if (elementGroup[i]==3){
                S3comp[ind2]=i;
                et[ind2]=((double)props.C);
                ind2++;
            }
        }
        S1comp[nS1]=dataset.l;
        
        memset(G13,0.0,(nS1+1)*sizeof(double));
        
        if(nS3>0){
            #pragma omp parallel default(shared) private(i,o)
            {
            #pragma omp for schedule(static)
                for (i=0;i<(nS1+1);i++){
                    int o;
                    if(i<nS1){
                        for (o=0;o<nS3;o++) G13[i] += et[o]*kernel(dataset,S1comp[i], S3comp[o], props)*dataset.y[S1comp[i]]*dataset.y[S3comp[o]];
                    }else{
                        for (o=0;o<nS3;o++) G13[nS1]+=et[o]*dataset.y[S3comp[o]];	
                        
                    }
                }
            }
            
        }

    }

    free(a);
    free(elementGroup);

    free(betaAux);
    free(betaNew);

    free(S3comp);
    free(S1comp);
        
    free(et);
    free(G13);
    free(H);

    return betaBest;
}


double* trainFULL(svm_dataset dataset,properties props){

    printf("\n");
    int MaxWorkingSize = props.MaxSize;
    double epsilon=1e6;
    double epsilonTmp=0.0;
    double epsilonThreshold=0.001;

    svm_dataset subdataset;
    subdataset.sparse = dataset.sparse;
    subdataset.maxdim = dataset.maxdim;
    subdataset.y=(double *) calloc(MaxWorkingSize,sizeof(double));
    subdataset.quadratic_value=(double *) calloc(MaxWorkingSize,sizeof(double));
    subdataset.x = (svm_sample **) calloc(MaxWorkingSize,sizeof(svm_sample *));
    
    int found10,found11, found12, found00, found01, found02;
    
    double *e = (double *) calloc(dataset.l,sizeof(double));
    double *beta=(double *) calloc((dataset.l+1),sizeof(double));
    double *betaNew=(double *) calloc((dataset.l+1),sizeof(double));
    double *betaBest=(double *) calloc((dataset.l+1),sizeof(double));
    double *betaTmp;

    int *SW = (int *) calloc(MaxWorkingSize,sizeof(int));
    int *SIN = (int *) calloc(dataset.l,sizeof(int));
    int *SC = (int *) calloc(dataset.l,sizeof(int));

    double *GIN=(double *) calloc((MaxWorkingSize+1),sizeof(double));
    double *esub=(double *) calloc((MaxWorkingSize+1),sizeof(double));
    double *betasub=(double *) calloc((MaxWorkingSize+1),sizeof(double));

    int nSW=0, nSIn=0, nSC=0;
    int i, o, ind=0, ind2=0;

    double lambeq, mil, mal;
    int neq=0;


    for (i=0;i<dataset.l;i++){		
        int randomNum=i%10;
        if (randomNum<1 && nSW<MaxWorkingSize){
            SW[ind]=i;
            ind++;
            nSW++;
        }else{
            SIN[ind2]=i;
            ind2++;
            nSIn++;
        }
        e[i]=dataset.y[i];		
    }	


    int iter=0;
    int endNorm=0;
    double bestNorm=1e20;
    int SinceBest=0;

    while( (endNorm==0) && (SinceBest<300)){
        iter+=1;

        // CONSTRUCT GIN AND GBIN
        
        if(nSIn>0){

            memset(GIN,0.0,(nSW+1)*sizeof(double));
        	  
            #pragma omp parallel default(shared) private(i,o)
            {
            #pragma omp for schedule(static)
                for (i=0;i<(nSW+1);i++){
                    int o;
                    if(i<nSW){
                        for (o=0;o<nSIn;o++) if (betaNew[SIN[o]] != 0.0){
                            GIN[i] += betaNew[SIN[o]]*kernel(dataset,SW[i], SIN[o], props)*dataset.y[SW[i]];
                        }
                        
                    }else{
                        for (o=0;o<nSIn;o++) GIN[nSW]+=betaNew[SIN[o]];
                    }
                }
            }
            
        }

        ////////////////////
        // CREATE SUBDATASET
        ///////////////////
        
        subdataset.l = nSW;
        for(i=0;i<nSW;i++){
            subdataset.y[i]=dataset.y[SW[i]];
            subdataset.quadratic_value[i]=dataset.quadratic_value[SW[i]];
            subdataset.x[i]=dataset.x[SW[i]];
            betasub[i]=beta[SW[i]];
            esub[i]=e[SW[i]];

        }


        betasub[nSW]=beta[dataset.l];

        /////////////////
        // CALL TO IRWLS
        /////////////////

        double *betaTmp = subIRWLS(subdataset,props, GIN, esub, betasub);
        

        /////////////////
        // UPDATING ERROR
        /////////////////
        

        memcpy(betaNew,beta,(dataset.l+1)*sizeof(double));

	
        for (i=0;i<nSW;i++){
            betaNew[SW[i]]=betaTmp[i];
        }

        betaNew[dataset.l]=betaTmp[subdataset.l];

        #pragma omp parallel default(shared) private(i,randomNum)
        {	
        #pragma omp for schedule(static)	
        for (i=0;i<dataset.l;i++){
            int j;

            for (j=0;j<nSW;j++){  
                e[i]=e[i]-kernel(dataset,i,SW[j],props)*(betaNew[SW[j]]-beta[SW[j]]);
                
            }
            e[i]=e[i]-(betaNew[dataset.l]-beta[dataset.l]);

        }
        }        

        free(betaTmp); 

        double deltaW=0.0;
	double normW=0.0;
        for (i=0;i<dataset.l+1;i++){
	    deltaW=deltaW+pow(beta[i]-betaNew[i],2.0);
	    normW=normW+pow(beta[i],2.0);
	}

        //printf("El valor es %f\n",deltaW/normW);

        if(deltaW/normW<props.Eta){
	    endNorm=1;
	}

        memcpy(beta,betaNew,(dataset.l+1)*sizeof(double));

	if(deltaW/normW<bestNorm){
	    bestNorm=deltaW/normW;
	    SinceBest=0;
	    memcpy(betaBest,betaNew,(dataset.l+1)*sizeof(double));
	}else{
	    SinceBest+=1;
	}


        ///////////////////////////////
        // UPDATING STOPPING CONDITIONS
        ///////////////////////////////

        found00=0,found01=0,found02=0,found10=0,found11=0,found12=0;

        nSW=0;
        nSIn=0;
        nSC=0;

        
        for (i=0;i<dataset.l;i++){
               
            if(betaNew[i]*dataset.y[i]==((double)props.C)){
                epsilonTmp=e[i]*dataset.y[i];

                if(epsilonTmp<-1.0*epsilonThreshold){
                    if(dataset.y[i]==-1 && found02==0){
                        SW[nSW]=i;
                        nSW+=1;
                        found02=1;
                    }else if(dataset.y[i]==1 && found12==0){
                        SW[nSW]=i;
                        nSW+=1;
                        found12=1;                        
                    }else{
                        SC[nSC]=i;
                        nSC+=1;
                    }
                }else{
                    SIN[nSIn]=i;
                    nSIn+=1;
                }

            }else if(betaNew[i]==0.0){
                epsilonTmp=e[i]*dataset.y[i];
                
                if(epsilonTmp>epsilonThreshold){
                    if(dataset.y[i]==-1 && found00==0){
                        SW[nSW]=i;
                        nSW+=1;
                        found00=1;
                    }else if(dataset.y[i]==1 && found10==0){
                        SW[nSW]=i;
                        nSW+=1;
                        found10=1;                        
                    }else{
                        SC[nSC]=i;
                        nSC+=1;
                    }
                }else{
                    SIN[nSIn]=i;
                    nSIn+=1;
                }

            }else if((betaNew[i]*dataset.y[i]!=0.0) && (betaNew[i]*dataset.y[i]!=(double)props.C)){
               
                epsilonTmp=fabs(e[i]*dataset.y[i]);

                if(epsilonTmp>epsilonThreshold){
                    if(dataset.y[i]==-1 && found01==0){
                        SW[nSW]=i;
                        nSW+=1;
                        found01=1;
                    }else if(dataset.y[i]==1 && found11==0){
                        SW[nSW]=i;
                        nSW+=1;
                        found11=1;                        
                    }else{
                        SC[nSC]=i;
                        nSC+=1;
                    }
                }else{
                    SC[nSC]=i;
                    nSC+=1;
                }

            }

        }
        
        
        //printf("ITERS SINCE BEST, %d %d %d\n",itersSinceBest,fullfill1+fullfill2+fullfill3,bestFulfill);
        printf("%s", ".");
        fflush(stdout);

        //////////////////////
        // SELECT WORKING SET
        //////////////////////
  
        if(nSC<(MaxWorkingSize-nSW)){
            for(i=0;i<nSC;i++){
                SW[nSW]=SC[i];
                nSW+=1;

            }
        }else{
              int *perm = rpermute(nSC);
              int space = (MaxWorkingSize-nSW);
            	for(i=0;i<nSC;i++){
                    if (i<space){
                        SW[nSW]=SC[perm[i]];
                        nSW+=1;
                    }else{
                        SIN[nSIn]=SC[perm[i]];
                        nSIn+=1;                	
                    }
            	}
            	free(perm);
        }
	
        //memcpy(beta,betaNew,dataset.l*sizeof(double));
    }

    printf("\n");
  
    return betaNew;

}

  
int main(int argc, char** argv)
{

    //srand(getpid());	
    //srand48(getpid());

    srand(0);	
    srand48(0);

    properties props = TrainParameters(&argc, &argv);
  
    if (argc != 3) {
        printInstructions();
        return 4;
    }

    char * data_file = argv[1];
    char * data_model = argv[2];
  	

    svm_dataset dataset = readTrainFile(data_file);
    printf("\nDataset Loaded from file: %s\n\nTraining samples: %d\nNumber of features: %d\n\n",data_file, dataset.l,dataset.maxdim);

    struct timeval tiempo1, tiempo2;
    omp_set_num_threads(props.Threads);

    printf("Running IRWLS\n");	
    gettimeofday(&tiempo1, NULL);

    initMemory(props.Threads,(props.MaxSize+1));
    double * W = trainFULL(dataset,props);

    gettimeofday(&tiempo2, NULL);
    printf("\nWeights calculated in %ld\n\n",((tiempo2.tv_sec-tiempo1.tv_sec)*1000+(tiempo2.tv_usec-tiempo1.tv_usec)/1000));

    model modelo = calculateModel(props, dataset, W);

    printf("Saving model in file: %s\n\n",data_model);	
 
    FILE *Out = fopen(data_model, "w+");
    storeModel(&modelo, Out);
    fclose(Out);

    return 0;
}

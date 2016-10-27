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
 * @brief Implementation of the training functions of the PSIRWLS algorithm.
 *
 * See PSIRWLS-train.h for a detailed description of its functions and parameters.
 * 
 * For a detailed description of the algorithm and its parameters read the following paper:
 *
 * Díaz-Morales, R., & Navia-Vázquez, Á. (2016). Efficient parallel implementation of kernel methods. Neurocomputing, 191, 175-186.
 *
 * @file PSIRWLS-train.c
 * @author Roberto Diaz Morales
 * @date 23 Aug 2016
 * @see PSIRWLS-train.h
 * 
 */

#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include "PSIRWLS-train.h"
#include "kernels.h"
#include "ParallelAlgorithms.h"


/**
 * @cond
 */


extern void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double
                   *alpha, double *a, int *lda, double *b, int *ldb, double *beta, double *c,
                   int *ldc );

extern void dpotrs_(char *uplo, int *n, int *nrhs, double *A, int *lda,
                    double *B, int *ldb, int *info);


/**
 * @brief Random selection of centroids for the semiparametric model
 *
 * It creates a random permutation and selects the first elements to be the indexes of the centroids of the semiparametric model.
 *
 * @param dataset The training set.
 * @param props The struct with the training parameters.
 */

int* randomCentroids(svm_dataset dataset,properties props){

    int* permut = malloc(dataset.l * sizeof(int));
    int i;
    // initial range of numbers
    for(i=0;i<dataset.l;++i){
        permut[i]=i;
    }
    
    for (i = dataset.l-1; i >= 0; --i){
        //generate a random number [0, n-1]
        int j = rand() % (i+1);
        //swap the last element with element at random index
        int temp = permut[i];
        permut[i] = permut[j];
        permut[j] = temp;
    }
    
    int* centroids = malloc(props.size * sizeof(int));
    
    for (i = 0; i < props.size; i++){
        centroids[i]=permut[i];
    }
    
    free(permut);
    return centroids;
    
}


/**
 * @brief Sparse Greedy Matrix Approximation algorithm
 *
 * Sparse Greedy Matrix Approximation algorithm to select the basis elements of the semi parametric model. For a detailed description read:
 *
 * Díaz-Morales, R., & Navia-Vázquez, Á. (2016). Efficient parallel implementation of kernel methods. Neurocomputing, 191, 175-186.
 *
 * @param dataset The training set.
 * @param props The struct with the training parameters.
 */

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

    for(i=0;i<64;i++){
            KNC[i]=(double *) malloc((props.size)*sizeof(double));
            KSM[i]=(double *) malloc((dataset.l)*sizeof(double));
            Z[i]=(double *) malloc((props.size)*sizeof(double));
    }

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
            	
            /*			
            if(size==2 && i>0){
                KNC[i]=(double *) malloc((props.size)*sizeof(double));
                KSM[i]=(double *) malloc((dataset.l)*sizeof(double));
                Z[i]=(double *) malloc((props.size)*sizeof(double));
            }*/

            miKNC=KNC[i];
            miKSM=KSM[i];
            miZ=Z[i];

            for(e=0;e<dataset.l;e++) miKSM[e]=kernelFunction(dataset,indexes[i],e,props);            

            for(e=0;e<size;e++){
                value=kernelFunction(dataset,indexes[i],centroids[e],props);
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
                //KNC[0]=(double *) malloc((props.size)*sizeof(double));
                //KSM[0]=(double *) malloc((dataset.l)*sizeof(double));
                //Z[0]=(double *) malloc((props.size)*sizeof(double));
                centroids[size]=dataset.l;
            }else{
                centroids[size]=dataset.l+1;
                KNC[0][0]=kernelFunction(dataset,centroids[0],centroids[1],props);
            }
            value=1.0;
            bestBasis=0;
            
            
        }
        
        if(size==0) printf("Best Error Descent %f, Average of positive data is centroid %d\n",value,size);
        if(size==1) printf("Best Error Descent %f, Average of negative data is centroid %d\n",value,size);
        if(size>1) printf("Best Error Descent %f, Data with index %d is centroid %d\n",value,centroids[size],size);

        #pragma omp parallel default(shared) private(i)
        {
        #pragma omp for schedule(static)	
        for(i=0;i<dataset.l;i++) KSC[size*(dataset.l)+i]=kernelFunction(dataset,i,centroids[size],props);
        }

        if(size==0){
            iKCTmp[0]=pow(kernelFunction(dataset,centroids[size],centroids[size],props)+0.000001,0.5);
            invKCTmp[0]=1.0/iKCTmp[0];
        }else{
            ParallelVectorMatrixT(KNC[bestBasis],size,invKC,L2,props.Threads);
            L3=kernelFunction(dataset,centroids[size],centroids[size],props)+0.00001;
            for(i=0;i<size;i++) L3 = L3 - (L2[i]*L2[i]);
            L3=pow(L3,0.5);
            IL3=1.0/L3;
            ParallelVectorMatrix(L2,size,invKC,IL2,props.Threads);

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
    

    for(i=0;i<64;i++){
        free(KNC[i]);
        free(KSM[i]);
        free(Z[i]);
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

/**
 * @brief Iterative Re-Weighted Least Squares Algorithm.
 *
 * IRWLS procedure to obtain the weights of the semi parametric model. For a detailed description of the algorithm and parallelization:
 *
 * Díaz-Morales, R., & Navia-Vázquez, Á. (2016). Efficient parallel implementation of kernel methods. Neurocomputing, 191, 175-186.
 *
 * @param dataset The training set.
 * @param indexes The indexes of the centroids selected by the SGMA algorithm.
 * @param props The struct with the training parameters.
 * @return The weights of every centroid.
 */

double* IRWLSpar(svm_dataset dataset, int* indexes,properties props){

    int i;
    double kernelvalue;

    double *KC=(double *) calloc(props.size*props.size,sizeof(double));
    double *KSC=(double *) calloc(dataset.l*props.size,sizeof(double));
    double *KSCA=(double *) calloc(dataset.l*props.size,sizeof(double));
    double *Da=(double *) calloc(dataset.l,sizeof(double));
    double *Day=(double *) calloc(dataset.l,sizeof(double));


    #pragma omp parallel for
    for (i=0;i<props.size;i++){
        int j=0;
        for (j=0;j<props.size;j++){
            KC[i*(props.size)+j]=kernelFunction(dataset,indexes[i], indexes[j], props);
            if(i==j) KC[i*(props.size)+j]+=pow(10,-5);
        }
    }


    double M=10000.0;

    #pragma omp parallel for
    for (i=0;i<dataset.l;i++){
        Da[i]=M;
        Day[i]=dataset.y[i]*M;
        int j = 0;
        for (j=0;j<props.size;j++){
            kernelvalue=kernelFunction(dataset,i, indexes[j], props);
            KSC[i*(props.size)+j]=kernelvalue;
            KSCA[i*(props.size)+j]=kernelvalue;
        }
    }
    

    //Stop conditions
    int  iter=0, max_iter=500,cambios=100, trueSVs=0;
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

    int thLS=(int) pow(2,floor(log(props.Threads)/log(2.0)));
    if(props.size<thLS) thLS=pow(2,floor(log(props.size)/log(2.0)));
    if(thLS<1) thLS=1;        
	
	int tamDgemm = props.Threads;
	if (props.size<props.Threads) tamDgemm = props.size;
    
	trueSVs=dataset.l;
	
    while( (iter<max_iter) && (deltaW/normW > 1e-6) && (itersSinceBestDW<5) ){

        memcpy(K1,KC,(props.size)*(props.size)*sizeof(double));

		if(trueSVs>0){
			#pragma omp parallel for
			for (i=0;i<tamDgemm;i++){
				int InitCol=round(i*props.size/tamDgemm);
				int FinalCol=round((i+1)*props.size/tamDgemm)-1;			
				int lengthCol=FinalCol-InitCol+1;
				if(lengthCol>0){
					dgemm_(&notrans, &notrans, &(lengthCol), &(row), &(trueSVs), &factor, &KSCA[InitCol], &(props.size), Day, &trueSVs, &zfactor, &K2[InitCol], &(props.size));
					dgemm_(&notrans, &trans, &(lengthCol), &(props.size), &(trueSVs), &factor, &KSCA[InitCol], &(props.size), KSCA, &props.size, &factor, &K1[InitCol], &(props.size));
				}
			}
		}else{
			memset(K2,0.0,props.size*sizeof(double));
		}

        memset(betaNew,0.0,props.size*sizeof(double));

        omp_set_num_threads(thLS);
        ParallelLinearSystem(K1,props.size,props.size,0,0,K2,props.size,1,0,0,props.size,1,betaNew,props.size,1,0,0,thLS);
        omp_set_num_threads(props.Threads);
        deltaW=0.0;        
        normW=0.0;

        for (i=0;i<props.size;i++){
            deltaW += pow(betaNew[i]-beta[i],2);
            normW += pow(betaNew[i],2);
            beta[i]=betaNew[i];
        }


        memcpy(e,dataset.y,dataset.l*sizeof(double));

        #pragma omp parallel for
        for (i=0;i<tamDgemm;i++){
            int InitCol=round(i*dataset.l/tamDgemm);
            int FinalCol=round((i+1)*dataset.l/tamDgemm)-1;			
            int lengthCol=FinalCol-InitCol+1;
            if(lengthCol>0){                
                dgemm_(&notrans, &notrans, &(row), &(lengthCol), &(props.size), &nfactor, beta, &row, &KSC[InitCol*props.size], &props.size, &factor, &e[InitCol], &(row));
            }
        }


        double alpha,chi;

        #pragma omp parallel for
        for(i=0;i<dataset.l;i++){

           if(e[i]*dataset.y[i]<0.0){
                Da[i]=0.0;
           }else{
           	Da[i]=1.0*props.C/(dataset.y[i]*e[i]);
	       }
	       if(Da[i]>M) Da[i]=M;
        }

        trueSVs=0;
        for(i=0;i<dataset.l;i++){
            if(Da[i]!=0.0){
                indKSCA[trueSVs]=i;
                ++trueSVs;
            }
        }

        #pragma omp parallel for
        for (i=0;i<trueSVs;i++){
            int j = 0;
            for (j=0;j<props.size;j++){
                KSCA[i*(props.size)+j]=sqrt(Da[indKSCA[i]])*KSC[indKSCA[i]*(props.size)+j];
            }
            Day[i]=sqrt(Da[indKSCA[i]])*dataset.y[indKSCA[i]];
        }

        ++iter;
        printf("Iteration %d, nSVs %d, ||deltaW||^2/||W||^2=%f\n",iter,trueSVs,deltaW/normW);
    
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

    free(KC);
    free(KSC);
    free(KSCA);
    free(Da);
    free(Day);

    free(K1);
    free(K2);
    free(beta);
    free(betaNew);
    free(e);
    free(indKSCA);

    return betaBest;
}

/**
 * @brief It converts the result into a model struct.
 *
 * After the training of a SVM using the PSIRWLS procedure, this function build a struct with the information and returns it.
 *
 * @param props The training parameters.
 * @param dataset The training set.
 * @param centroids of the selected centroids by the SGMA algorithm.
 * @param beta The weights of every centroid obtained with the IRWLS algorithm.
 * @return The struct that storages all the information of the classifier.
 */

model calculatePSIRWLSModel(properties props, svm_dataset dataset, int *centroids, double * beta ){
    model classifier;
    classifier.Kgamma = props.Kgamma;
    classifier.sparse = dataset.sparse;
    classifier.maxdim = dataset.maxdim;
    classifier.nSVs = props.size;
    classifier.bias=0.0;
    classifier.kernelType = props.kernelType;
        
    int nElem=0;
    svm_sample *iteratorSample;
    svm_sample *classifierSample;
    int i;
    for (i =0;i<props.size;i++){
        iteratorSample = dataset.x[centroids[i]];
        while (iteratorSample->index != -1){
        	  ++iteratorSample;
            ++nElem;
        }
        ++nElem;
    }

    classifier.nElem = nElem;
    classifier.weights = (double *) calloc(props.size,sizeof(double));
	memcpy(classifier.weights,beta,props.size*sizeof(double));	
    classifier.quadratic_value = (double *) calloc(props.size,sizeof(double));

    classifier.x = (svm_sample **) calloc(props.size,sizeof(svm_sample *));
    classifier.features = (svm_sample *) calloc(nElem,sizeof(svm_sample));
    
    int indexIt=0;
    int featureIt=0;
    for (i =0;i<props.size;i++){
        classifier.quadratic_value[i]=dataset.quadratic_value[centroids[i]];
        classifier.x[i] = &classifier.features[featureIt];
        iteratorSample = dataset.x[centroids[i]];
        classifierSample = classifier.x[i];
        while (iteratorSample->index != -1){
            classifierSample->index = iteratorSample->index;
            classifierSample->value = iteratorSample->value;
            ++classifierSample;
            ++iteratorSample;
            ++featureIt;
        }

        classifierSample->index = iteratorSample->index;
            
        ++featureIt;
    }

    return classifier;
}

/**
 * @brief It parses input command line to extract the parameters of the PSIRWLS algorithm.
 *
 * It parses input command line to extract the parameters.
 * @param argc The number of words of the command line.
 * @param argv The list of words of the command line.
 * @return A struct that contains the values of the training parameters of the PSIRWLS algorithm.
 */

properties parseTrainParameters(int* argc, char*** argv) {

    properties props;
    props.Kgamma = 1.0;
    props.C = 1.0;
    props.Threads=1;
    props.MaxSize=500;
    props.Eta=0.001;
    props.size=10;
    props.algorithm=1;
    props.kernelType=1;

    int i,j;
    for (i = 1; i < *argc; ++i) {
        if ((*argv)[i][0] != '-') break;
        if (++i >= *argc) {
            printPSIRWLSInstructions();
            exit(1);
        }

        char* param_name = &(*argv)[i-1][1];
        char* param_value = (*argv)[i];
        if (strcmp(param_name, "g") == 0) {    	
            props.Kgamma = atof(param_value);
        } else if (strcmp(param_name, "c") == 0) {
            props.C = atof(param_value);
        } else if (strcmp(param_name, "e") == 0) {
            props.Eta = atof(param_value);
        }else if (strcmp(param_name, "k") == 0) {
            props.kernelType = atoi(param_value);
        }else if (strcmp(param_name, "t") == 0) {
            props.Threads = atoi(param_value);
        } else if (strcmp(param_name, "w") == 0) {
            props.MaxSize = atoi(param_value);
        } else if (strcmp(param_name, "s") == 0) {
            props.size = atoi(param_value);
        } else if (strcmp(param_name, "a") == 0) {
            props.algorithm = atoi(param_value);
        } else {
            fprintf(stderr, "Unknown parameter %s\n",param_name);
            printPSIRWLSInstructions();
            exit(2);
        }
    }
  
    for (j = 1; i + j - 1 < *argc; ++j) {
        (*argv)[j] = (*argv)[i + j - 1];
    }
    *argc -= i - 1;

    return props;

}

/**
 * @brief Print Instructios.
 *
 *  It shows PSIRWLS-train command line instructions in the standard output.
 */

void printPSIRWLSInstructions(void) {
    fprintf(stderr, "PSIRWLS-train: This software train the sparse SVM on the given training set ");
    fprintf(stderr, "and generages a model for futures prediction use.\n\n");
    fprintf(stderr, "Usage: PSIRWLS-train [options] training_set_file model_file\n\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -k kernel type: (default 1)\n");
    fprintf(stderr, "       0 -- Linear kernel u'*v\n");
    fprintf(stderr, "       1 -- radial basis function: exp(-gamma*|u-v|^2)\n");
    fprintf(stderr, "  -g gamma: set gamma in radial basis kernel function (default 1)\n");
    fprintf(stderr, "       radial basis K(u,v)= exp(-gamma*|u-v|^2)\n");
    fprintf(stderr, "  -c Cost: set SVM Cost (default 1)\n");
    fprintf(stderr, "  -t Threads: Number of threads (default 1)\n");
    fprintf(stderr, "  -s Classifier size: Size of the classifier (default 1)\n");
    fprintf(stderr, "  -a Algorithm: Algorithm for centroids selection (default 1)\n");
    fprintf(stderr, "       0 -- Random Selection\n");
    fprintf(stderr, "       1 -- SGMA (Sparse Greedy Matrix Approximation)\n");
}

/**
 * @endcond
 */
